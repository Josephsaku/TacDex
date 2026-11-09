"""Strict episode schema, causal windows, split audits and train-only scaling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = {"align_train", "align_val", "downstream_train", "downstream_val",
          "test_id", "test_ood_size", "test_ood_shape"}
SCHEMA_VERSION = 2


def load_episode(path):
    with np.load(path, allow_pickle=False) as source:
        e = {key: source[key] for key in source.files}
    e["metadata"] = json.loads(str(e["metadata"]))
    return e


def validate_episode(e, paired=False):
    meta = e["metadata"]
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Episode schema v2 required: export saved depth and direct sensor resultant; missing fields cannot be filled with zeros")
    for key in ("episode_id", "group_id", "geometry_id", "sensor_id", "units",
                "depth_method", "synthetic", "sim_force_source", "force_frame"):
        if key not in meta:
            raise ValueError(f"Missing metadata: {key}")
    if meta["sensor_id"] != 3 or meta["units"] != "m,N,s":
        raise ValueError("Only ID3 with explicit SI units is supported")
    allowed_depth = {"taxel_normal_raycast", "taxel_collision_depth"}
    allowed_force = {"mujoco_contact_solver"}
    if meta["synthetic"] is True:
        allowed_depth.add("analytic_synthetic")
        allowed_force.add("synthetic_fixture")
    if meta["depth_method"] not in allowed_depth:
        raise ValueError("Contact distance / RBF force is not a dense depth map")
    if meta["sim_force_source"] not in allowed_force:
        raise ValueError("sim_force_N must be the saved direct MuJoCo sensor resultant, not inferred taxel forces")
    if meta["force_frame"] != "sensor_local":
        raise ValueError("Forces must be expressed in the calibrated sensor-local frame")
    t = e["time_s"]
    depth = e["sim_depth_m"]
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
        raise ValueError("time_s must be finite and strictly increasing")
    if depth.ndim != 2 or len(depth) != len(t):
        raise ValueError("sim_depth_m must have shape [T,N]")
    n = depth.shape[1]
    for key in ("geometry",):
        if e[key].shape != (n, 6) or not np.isfinite(e[key]).all():
            raise ValueError("geometry must be finite [N,6]: xyz(m), unit normal")
    if not np.allclose(np.linalg.norm(e["geometry"][:, 3:], axis=1), 1, atol=1e-3):
        raise ValueError("Taxel normals must be unit length")
    for key in ("sim_mask", "real_mask") if paired else ("sim_mask",):
        if e[key].shape != (n,) or e[key].dtype != np.bool_ or not e[key].any():
            raise ValueError(f"{key} must contain valid taxels as boolean [N]")
    if not np.isfinite(depth[:, e["sim_mask"]]).all() or (depth[:, e["sim_mask"]] < 0).any():
        raise ValueError("Depth must be finite and nonnegative on valid taxels")
    if e["stimulus_id"].shape != t.shape or e["stimulus_id"].dtype.kind not in "US":
        raise ValueError("stimulus_id must be a string array [T]")
    if e["frame_valid"].shape != t.shape or e["frame_valid"].dtype != np.bool_:
        raise ValueError("frame_valid must be boolean [T]")
    if not e["frame_valid"].any():
        raise ValueError("Episode has no valid frames")
    if "sim_force_N" not in e or e["sim_force_N"].shape != (len(t), 3):
        raise ValueError("sim_force_N is required with shape [T,3]")
    if not np.isfinite(e["sim_force_N"][e["frame_valid"]]).all():
        raise ValueError("Nonfinite solver force in valid frames")
    if paired:
        if not meta.get("pairing_verified") or not meta.get("axis_calibrated"):
            raise ValueError("Paired training requires verified stimulus pairing and calibrated axes")
        real = e["real_taxel_N"]
        if real.shape != (len(t), n, 3):
            raise ValueError("real_taxel_N must be [T,N,3]; never fabricate taxels from net force")
        if e["real_force_N"].shape != (len(t), 3):
            raise ValueError("real_force_N must be [T,3]")
        valid = e["frame_valid"]
        if not np.isfinite(real[valid][:, e["real_mask"]]).all() or not np.isfinite(e["real_force_N"][valid]).all():
            raise ValueError("Missing real readings in valid frames")
        event = e["raw_event_id"]
        if event.shape != t.shape or event.dtype.kind not in "iu" or (event < 0).any():
            raise ValueError("raw_event_id must be nonnegative integer [T]")
        if len(np.unique(event)) != len(event):
            raise ValueError("Repeated tactile packets: deduplicate before constructing history")
        if e["pair_error_s"].shape != t.shape or not np.isfinite(e["pair_error_s"]).all():
            raise ValueError("pair_error_s must be finite [T]")
        tolerance = meta.get("pair_tolerance_s", 0.02)
        if not np.isfinite(tolerance) or tolerance <= 0 or np.abs(e["pair_error_s"][valid]).max() > tolerance:
            raise ValueError("Pairing timestamp tolerance exceeded")
        # Force thresholds are quality gates, not assumptions of force equivalence.
        contact = np.linalg.norm(e["real_force_N"], axis=1) > meta.get("contact_force_threshold_N", 0.1)
        sim_contact = depth[:, e["sim_mask"]].max(1) > meta.get("contact_depth_threshold_m", 1e-6)
        mismatch = np.mean(contact[valid] != sim_contact[valid])
        if mismatch > meta.get("max_contact_mismatch_fraction", 0.1):
            raise ValueError(f"ID3 real/sim contact mismatch: {mismatch:.1%}")
    return {"frames": len(t), "taxels": n, "paired": paired,
            "valid_frames": int(e["frame_valid"].sum())}


def read_manifest(path):
    path = Path(path)
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("schema_version") != SCHEMA_VERSION or not m.get("episodes"):
        raise ValueError("Expected schema_version=2 and nonempty episodes")
    groups, episodes, locations, stimuli = {}, set(), {}, {}
    geometry_id, geometry = None, None
    for row in m["episodes"]:
        if row["split"] not in SPLITS:
            raise ValueError(f"Unknown split: {row['split']}")
        e = load_episode(path.parent / row["path"])
        paired = "real_taxel_N" in e
        validate_episode(e, paired)
        if row["split"].startswith("align") and not paired:
            raise ValueError("Alignment requires measured paired data")
        meta = e["metadata"]
        if meta["episode_id"] in episodes:
            raise ValueError("Duplicate episode in manifest")
        episodes.add(meta["episode_id"])
        for registry, key in ((groups, meta["group_id"]),
                              (locations, meta.get("location_id", meta["group_id"]))):
            if key in registry and registry[key] != row["split"]:
                raise ValueError("Trajectory/location leakage across splits")
            registry[key] = row["split"]
        for stimulus in e["stimulus_id"]:
            if str(stimulus) in stimuli and stimuli[str(stimulus)] != row["split"]:
                raise ValueError("Physical stimulus leakage across splits")
            stimuli[str(stimulus)] = row["split"]
        if geometry is None:
            geometry, geometry_id = e["geometry"], meta["geometry_id"]
        elif geometry_id != meta["geometry_id"] or not np.array_equal(geometry, e["geometry"]):
            raise ValueError("All episodes must share the same verified ID3 geometry and taxel order")
        row["resolved_path"] = str((path.parent / row["path"]).resolve())
    return m


class Normalizer:
    """Per-channel zero-centered RMS preserves the physical no-contact zero."""
    def __init__(self, stats):
        self.stats = stats

    @classmethod
    def fit(cls, episodes):
        sums, counts = {}, {}
        for e in episodes:
            valid = e["frame_valid"]
            streams = {"sim": e["sim_depth_m"][valid][:, e["sim_mask"]].reshape(-1, 1),
                       "sim_force": e["sim_force_N"][valid]}
            if "real_taxel_N" in e:
                streams["real"] = e["real_taxel_N"][valid][:, e["real_mask"]].reshape(-1, e["real_taxel_N"].shape[-1])
                streams["real_force"] = e["real_force_N"][valid]
            for key, values in streams.items():
                sums[key] = sums.get(key, 0) + np.square(values.astype(np.float64)).sum(0)
                counts[key] = counts.get(key, 0) + len(values)
        if set(sums) != {"sim", "real", "real_force", "sim_force"}:
            raise ValueError("Fit normalization only on paired alignment training episodes")
        return cls({k: np.maximum(np.sqrt(sums[k] / counts[k]), 1e-8).tolist() for k in sums})

    def transform(self, x, key):
        return x / np.asarray(self.stats[key], dtype=np.float32)


class EpisodeDataset(Dataset):
    def __init__(self, manifest, split, model_config, normalizer, *, paired,
                 max_history_gap_s=0.25, task="depth"):
        self.config, self.normalizer, self.paired, self.task = model_config, normalizer, paired, task
        self.episodes = [load_episode(r["resolved_path"]) for r in manifest["episodes"] if r["split"] == split]
        if not self.episodes:
            raise ValueError(f"No episodes for {split}")
        self.index = []
        for ei, e in enumerate(self.episodes):
            validate_episode(e, paired)
            if e["sim_depth_m"].shape[1] != model_config.taxels:
                raise ValueError("Configured taxel count differs from data")
            if paired and e["real_taxel_N"].shape[-1] != model_config.real_channels:
                raise ValueError("Configured real channel count differs from measured data")
            h = max(model_config.history if paired else 1, model_config.sim_history)
            for i in range(h - 1, len(e["time_s"])):
                start = i - h + 1
                if e["frame_valid"][start:i + 1].all() and np.all(np.diff(e["time_s"][start:i + 1]) <= max_history_gap_s):
                    self.index.append((ei, i))
        if not self.index:
            raise ValueError("No complete causal history windows; check cadence and validity")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        ei, i = self.index[index]
        e, c = self.episodes[ei], self.config
        geometry = e["geometry"].astype(np.float32).copy()
        # Fixed physical scale (1 cm); NOT fitted to test data.
        geometry[:, :3] /= 0.01
        sim = e["sim_depth_m"][i - c.sim_history + 1:i + 1, :, None].copy()
        sim[:, ~e["sim_mask"]] = 0
        result = {"sim": self.normalizer.transform(sim, "sim"), "geometry": geometry,
                  "sim_force": self.normalizer.transform(e["sim_force_N"][i - c.sim_history + 1:i + 1], "sim_force"),
                  "sim_mask": e["sim_mask"], "stimulus_id": str(e["stimulus_id"][i]),
                  "episode_id": e["metadata"]["episode_id"]}
        if self.task == "depth":
            result["target"] = self.normalizer.transform(sim[-1, :, 0], "sim")
        else:
            key = "shape_label" if self.task == "classification" else "target"
            if key not in e:
                raise ValueError(f"Missing downstream label {key}")
            result["target"] = e[key][i]
        if self.paired:
            real = e["real_taxel_N"][i - c.history + 1:i + 1].copy()
            real[:, ~e["real_mask"]] = 0
            result.update(real=self.normalizer.transform(real, "real"),
                          real_force=self.normalizer.transform(e["real_force_N"][i - c.history + 1:i + 1], "real_force"),
                          real_mask=e["real_mask"])
        return {k: torch.from_numpy(np.asarray(v).copy()).float() if isinstance(v, (np.ndarray, np.number)) and np.asarray(v).dtype.kind != "b"
                else torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for k, v in result.items()}


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
