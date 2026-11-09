"""Synthetic fixtures exercise software only; they are NOT real ID3 data."""
import json
from pathlib import Path

import numpy as np


def generate(root, frames=40, seed=7):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError("Use a fresh synthetic output directory")
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(-0.01, 0.01, 12), np.linspace(-0.009, 0.009, 10))
    geometry = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(120), np.zeros(120), np.zeros(120), np.ones(120)]).astype(np.float32)
    rows = []
    splits = ["align_train"] * 6 + ["align_val"] * 2 + ["downstream_train"] * 4 + ["downstream_val"] * 2 + ["test_id"] * 2
    for episode, split in enumerate(splits):
        name = f"synthetic_{episode:03d}"
        center = rng.uniform(-0.006, 0.006, 2)
        sigma = rng.uniform(0.002, 0.005)
        phase = np.linspace(0, 2 * np.pi, frames)
        locations = center + 0.002 * np.column_stack([np.cos(phase), np.sin(phase)])
        distance = geometry[None, :, :2] - locations[:, None, :]
        depth = ((0.0004 + 0.0002 * np.sin(phase))[:, None] * np.exp(-(distance ** 2).sum(-1) / (2 * sigma ** 2))).astype(np.float32)
        # Deterministic analytic transduction, not a claimed ID3 physical model.
        taxel = np.stack([depth * 35, depth * -20, depth * 250], axis=-1)
        force = taxel.sum(1)
        # Software fixture only. This is NOT solver output and is explicitly
        # rejected for production records (synthetic=False).
        sim_force = force * np.array([0.8, 1.2, 0.9], dtype=np.float32)
        meta = dict(schema_version=2, episode_id=name, group_id=name, location_id=name,
                    geometry_id="synthetic_flat_120_v1", sensor_id=3, units="m,N,s",
                    depth_method="analytic_synthetic", synthetic=True,
                    sim_force_source="synthetic_fixture", force_frame="sensor_local",
                    pairing_verified=True, axis_calibrated=True, pair_tolerance_s=0.02,
                    contact_force_threshold_N=0.01, max_contact_mismatch_fraction=0.0)
        arrays = dict(metadata=np.array(json.dumps(meta)), geometry=geometry,
                      time_s=np.arange(frames, dtype=float) * 0.1,
                      sim_depth_m=depth, sim_force_N=sim_force, sim_mask=np.ones(120, dtype=bool),
                      frame_valid=np.ones(frames, dtype=bool),
                      stimulus_id=np.array([f"{name}:{i}" for i in range(frames)]))
        # Downstream training/validation genuinely have NO real branch present.
        if not split.startswith("downstream"):
            arrays.update(real_taxel_N=taxel, real_force_N=force,
                          real_mask=np.ones(120, dtype=bool), raw_event_id=np.arange(frames, dtype=np.int64),
                          pair_error_s=np.zeros(frames))
        np.savez_compressed(root / f"{name}.npz", **arrays)
        rows.append(dict(path=f"{name}.npz", split=split))
    (root / "manifest.json").write_text(json.dumps({"schema_version": 2, "synthetic": True, "episodes": rows}, indent=2), encoding="utf-8")
    return root / "manifest.json"
