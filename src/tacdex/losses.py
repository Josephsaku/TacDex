"""Symmetric InfoNCE and all four self/cross reconstruction paths."""
import torch
from torch.nn import functional as F


def alignment_loss(real, sim, stimulus_ids, temperature=0.1, mode="infonce"):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if mode == "none":
        return (real.sum() + sim.sum()) * 0
    if mode == "mse":
        return F.mse_loss(real, sim)
    if mode != "infonce":
        raise ValueError(f"Unknown alignment: {mode}")
    if len(set(stimulus_ids)) < 2:
        raise ValueError("InfoNCE needs at least two distinct physical stimuli")
    logits = F.normalize(real, dim=-1) @ F.normalize(sim, dim=-1).T / temperature
    positive = torch.tensor([[a == b for b in stimulus_ids] for a in stimulus_ids],
                            dtype=torch.bool, device=logits.device)
    # Multi-positive extension: repeated measurements of the SAME certified
    # stimulus are never treated as negatives. Unique IDs give ordinary InfoNCE.
    def direction(scores, targets):
        return -(scores.log_softmax(1) * targets).sum(1).div(targets.sum(1)).mean()
    return (direction(logits, positive) + direction(logits.T, positive.T)) / 2


def masked_mse(pred, target, mask):
    valid = mask[:, None, :, None].expand_as(target)
    if not valid.any():
        raise ValueError("Reconstruction has no valid taxels")
    return (pred - target).square()[valid].mean()


def pretrain_loss(batch, embeddings, decoded, *, align_weight=1.0,
                  recon_weight=1.0, force_weight=1.0, temperature=0.1,
                  alignment="infonce", cross_reconstruction=True):
    align = alignment_loss(embeddings["real"], embeddings["sim"],
                           batch["stimulus_id"], temperature, alignment)
    if min(align_weight, recon_weight, force_weight) < 0:
        raise ValueError("Loss weights must be nonnegative")
    terms, details = {}, {}
    for source in ("real", "sim"):
        for target in ("real", "sim"):
            if source != target and not cross_reconstruction:
                continue
            value = masked_mse(decoded[source][target], batch[target], batch[f"{target}_mask"])
            details[f"{source}_to_{target}_taxels"] = value
            force_key = f"{target}_force"
            if force_key in decoded[source]:
                force_error = F.mse_loss(decoded[source][force_key], batch[force_key])
                details[f"{source}_to_{target}_force"] = force_error
                value = (value + force_weight * force_error) / (1 + force_weight)
            terms[f"{source}_to_{target}"] = value
    reconstruction = torch.stack(list(terms.values())).mean()
    total = align_weight * align + recon_weight * reconstruction
    return total, {"total": total, "alignment": align, "reconstruction": reconstruction, **terms, **details}
