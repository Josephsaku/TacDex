"""Two-stage training with auditable checkpoints and simulation-only probes."""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SCHEMA_VERSION, EpisodeDataset, Normalizer, load_episode, manifest_digest, read_manifest
from .losses import pretrain_loss
from .models import FrozenProbe, ModelConfig, TactSpace


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Stable single-GPU debug runs. Exact reproducibility can still vary by release.
    torch.backends.cudnn.benchmark = False


def device_for(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    return torch.device(name)


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def output_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Use a new run directory to preserve previous results")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path, **values):
    temp = Path(str(path) + ".tmp")
    torch.save(values, temp)
    temp.replace(path)


def load_checkpoint(path, device="cpu"):
    # No arbitrary pickle objects: checkpoints contain tensors and builtin values.
    return torch.load(path, map_location=device, weights_only=True)


def make_loader(dataset, cfg, shuffle=False, contrastive=False):
    batch = min(int(cfg.get("batch_size", 32)), len(dataset))
    if contrastive and batch < 2:
        raise ValueError("Contrastive loader needs at least two samples")
    # Avoid singleton tails in train AND validation. Documented in metrics.
    drop = contrastive and len(dataset) % batch == 1
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle,
                      num_workers=0, drop_last=drop)


def dataset(manifest, split, mc, norm, cfg, paired, task="depth"):
    return EpisodeDataset(manifest, split, mc, norm, paired=paired, task=task,
                          max_history_gap_s=cfg.get("max_history_gap_s", 0.25))


def pretrain(manifest_path, config_path, out, device="auto"):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    seed_all(cfg.get("seed", 7))
    device = device_for(device)
    mc = ModelConfig(**cfg["model"])
    m = read_manifest(manifest_path)
    train_episodes = [load_episode(r["resolved_path"]) for r in m["episodes"] if r["split"] == "align_train"]
    norm = Normalizer.fit(train_episodes)
    loaders = {split: make_loader(dataset(m, split, mc, norm, cfg, True), cfg,
                                  split == "align_train", True) for split in ("align_train", "align_val")}
    model = TactSpace(mc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("weight_decay", 1e-4))
    out = output_directory(out)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    best = float("inf")
    history = []
    epochs = int(cfg.get("epochs", 100))
    if epochs < 1:
        raise ValueError("epochs must be positive")
    for epoch in range(epochs):
        row = {"epoch": epoch + 1}
        for split, loader in loaders.items():
            train = split == "align_train"
            model.train(train)
            totals, count = {}, 0
            zs = {"real": [], "sim": []}
            for batch in loader:
                batch = to_device(batch, device)
                with torch.set_grad_enabled(train):
                    z, decoded = model(batch)
                    loss, metrics = pretrain_loss(batch, z, decoded, **cfg.get("loss", {}))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite pretraining loss")
                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0), error_if_nonfinite=True)
                        opt.step()
                size = len(batch["sim"])
                count += size
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0) + value.detach().item() * size
                if not train:
                    for domain in zs:
                        zs[domain].append(z[domain].detach().cpu())
            row[split] = {key: value / count for key, value in totals.items()}
            row[split]["evaluated_windows"] = count
            if not train:
                row[split]["latent_std"] = {k: torch.cat(v).std(0, unbiased=False).mean().item() for k, v in zs.items()}
                row[split]["paired_cosine"] = (torch.cat(zs["real"]) * torch.cat(zs["sim"])).sum(-1).mean().item()
        history.append(row)
        state = dict(stage="alignment", schema_version=SCHEMA_VERSION, model_config=asdict(mc), model=model.state_dict(),
                     normalization=norm.stats, config=cfg, epoch=epoch + 1,
                     optimizer=opt.state_dict(), manifest_sha256=manifest_digest(manifest_path),
                     geometry_id=train_episodes[0]["metadata"]["geometry_id"],
                     geometry=torch.from_numpy(train_episodes[0]["geometry"]),
                     source_groups=[e["metadata"]["group_id"] for e in train_episodes],
                     torch_version=str(torch.__version__), device=str(device))
        save_checkpoint(out / "last.pt", **state)
        if row["align_val"]["total"] < best:
            best = row["align_val"]["total"]
            save_checkpoint(out / "best.pt", **state)
        (out / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"alignment epoch={epoch+1} train={row['align_train']['total']:.4f} val={row['align_val']['total']:.4f}", flush=True)
    return history


def restore_encoder(checkpoint, device):
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Checkpoint predates sim-force inputs; retrain with schema v2")
    model = TactSpace(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device)


def verify_geometry(checkpoint, manifest):
    for row in manifest["episodes"]:
        e = load_episode(row["resolved_path"])
        if e["metadata"]["geometry_id"] != checkpoint["geometry_id"] or not np.array_equal(e["geometry"], checkpoint["geometry"].cpu().numpy()):
            raise ValueError("Checkpoint geometry/taxel order differs from input")


def task_loss(pred, batch, task):
    if task == "classification":
        return nn.functional.cross_entropy(pred, batch["target"].long())
    target = batch["target"]
    if target.ndim == 1 and pred.shape[-1] == 1:
        target = target[:, None]
    if target.shape != pred.shape:
        raise ValueError("Regression target and task output shape differ")
    if task == "depth":
        return (pred - target).square()[batch["sim_mask"]].mean()
    return nn.functional.mse_loss(pred, target)


def train_probe(manifest_path, pretrained, config_path, out, device="auto"):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    seed_all(cfg.get("seed", 7))
    device = device_for(device)
    ckpt = load_checkpoint(pretrained)
    m = read_manifest(manifest_path)
    if manifest_digest(manifest_path) != ckpt["manifest_sha256"]:
        raise ValueError("Use the same audited manifest across stages to prevent split leakage")
    verify_geometry(ckpt, m)
    encoder = restore_encoder(ckpt, device)
    norm = Normalizer(ckpt["normalization"])
    task = cfg.get("task", "depth")
    if task not in ("depth", "classification", "regression"):
        raise ValueError("Unsupported downstream task")
    outputs = encoder.config.taxels if task == "depth" else int(cfg["outputs"])
    probe = FrozenProbe(encoder, outputs, cfg.get("hidden", 128)).to(device)
    frozen = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
    opt = torch.optim.AdamW(probe.head.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
    loaders = {s: make_loader(dataset(m, s, encoder.config, norm, cfg, False, task), cfg,
                              s == "downstream_train") for s in ("downstream_train", "downstream_val")}
    out = output_directory(out)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    history, best = [], float("inf")
    epochs = int(cfg.get("epochs", 100))
    if epochs < 1:
        raise ValueError("epochs must be positive")
    for epoch in range(epochs):
        row = {"epoch": epoch + 1}
        for split, loader in loaders.items():
            train = split == "downstream_train"
            probe.train(train)
            total, count = 0, 0
            for batch in loader:
                batch = to_device(batch, device)
                # Dataset is deliberately constructed without any real observations.
                if "real" in batch or "real_force" in batch:
                    raise RuntimeError("Real input leaked into simulation-only probe training")
                with torch.set_grad_enabled(train):
                    pred = probe(batch, "sim")
                    loss = task_loss(pred, batch, task)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite downstream loss")
                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(probe.head.parameters(), 1.0, error_if_nonfinite=True)
                        opt.step()
                total += loss.item() * len(pred)
                count += len(pred)
            row[split] = total / count
        history.append(row)
        if any(not torch.equal(v.cpu(), frozen[k]) for k, v in encoder.state_dict().items()):
            raise RuntimeError("Frozen encoder weights changed")
        state = dict(stage="probe", encoder=ckpt, head=probe.head.state_dict(),
                     outputs=outputs, task=task, config=cfg, epoch=epoch + 1,
                     optimizer=opt.state_dict(), encoders_unchanged=True)
        save_checkpoint(out / "last.pt", **state)
        if row["downstream_val"] < best:
            best = row["downstream_val"]
            save_checkpoint(out / "best.pt", **state)
        (out / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"probe epoch={epoch+1} train={row['downstream_train']:.4f} val={row['downstream_val']:.4f}", flush=True)
    return history


@torch.no_grad()
def evaluate(manifest_path, checkpoint, split="test_id", device="auto"):
    if split not in {"test_id", "test_ood_size", "test_ood_shape"}:
        raise ValueError("Evaluation requires an explicit test partition")
    device = device_for(device)
    ckpt = load_checkpoint(checkpoint)
    base = ckpt["encoder"]
    m = read_manifest(manifest_path)
    if manifest_digest(manifest_path) != base["manifest_sha256"]:
        raise ValueError("Test manifest must match audited training splits")
    verify_geometry(base, m)
    encoder = restore_encoder(base, device)
    probe = FrozenProbe(encoder, ckpt["outputs"], ckpt["config"].get("hidden", 128)).to(device).eval()
    probe.head.load_state_dict(ckpt["head"])
    norm = Normalizer(base["normalization"])
    ds = dataset(m, split, encoder.config, norm, ckpt["config"], True, ckpt["task"])
    loader = make_loader(ds, ckpt["config"])
    predictions, targets, masks, groups = {"sim": [], "real": []}, [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        for domain in predictions:
            predictions[domain].append(probe(batch, domain).cpu())
        targets.append(batch["target"].cpu())
        masks.append(batch["sim_mask"].cpu())
        groups.extend(batch["episode_id"])
    target, mask = torch.cat(targets), torch.cat(masks)
    result = {"split": split, "samples": len(ds), "episodes": len(ds.episodes),
              "synthetic": all(e["metadata"]["synthetic"] for e in ds.episodes),
              "encoders_unchanged": ckpt["encoders_unchanged"], "task": ckpt["task"]}
    for domain, values in predictions.items():
        pred = torch.cat(values)
        if ckpt["task"] == "classification":
            result[domain] = {"accuracy": (pred.argmax(-1) == target.long()).float().mean().item()}
        else:
            if target.ndim == 1 and pred.shape[-1] == 1:
                target = target[:, None]
            if ckpt["task"] == "depth":
                # Nonnegative physical output; same inference transform in both domains.
                pred = pred.clamp_min(0)
                error = (pred - target).abs() * norm.stats["sim"][0] * 1e6
                sample_error = (error * mask).sum(1) / mask.sum(1)
                contact = mask & (target > 1e-6 / norm.stats["sim"][0])
                contact_mae = error[contact].mean().item() if contact.any() else None
                unit = "um"
            else:
                sample_error = (pred - target).abs().reshape(len(pred), -1).mean(1)
                contact_mae, unit = None, "target_units"
            episode_values = [sample_error[torch.tensor([g == name for g in groups])].mean().item() for name in sorted(set(groups))]
            result[domain] = {"mae": sample_error.mean().item(), "units": unit,
                              "contact_mae": contact_mae, "p20": sample_error.quantile(.2).item(),
                              "p80": sample_error.quantile(.8).item(),
                              "episode_macro_mae": float(np.mean(episode_values))}
    if ckpt["task"] == "depth":
        result["zero_depth_baseline_mae_um"] = ((target.abs() * mask).sum() / mask.sum() * norm.stats["sim"][0] * 1e6).item()
        result["caveat"] = "Simulation depth reconstruction is an autoencoding baseline; real evaluation requires verified paired geometric targets. Synthetic results do not establish real transfer."
    return result
