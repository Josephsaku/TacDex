"""Explicit field mapping for SDK exports and diagnosis of old replay files."""
import csv
import json
from pathlib import Path

import numpy as np


def read_geometry_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(r["taxel_id"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Taxel IDs must be ordered and contiguous from zero")
    xyz = np.array([[float(r[k]) * 1e-3 for k in ("x_mm", "y_mm", "z_mm")] for r in rows])
    normals = np.array([[float(r[k]) for k in ("nx", "ny", "nz")] for r in rows])
    if (np.linalg.norm(normals, axis=1) < 1e-8).any():
        raise ValueError("Invalid geometry normals")
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return np.concatenate([xyz, normals], axis=1).astype(np.float32)


def extract_id3(sensor_ids, module_force, taxel_force, *, scale_to_newton,
                sdk_to_sensor, taxel_order):
    """Canonical decoded packet adapter; no assumed proprietary wire format.

    module_force: [modules,3]; taxel_force: mapping sensor_id -> [N,3].
    taxel_order maps geometry taxel order to the SDK packet taxel indices.
    Forces use one common module frame, not individual taxel tangent frames.
    """
    ids = list(sensor_ids)
    if ids.count(3) != 1 or 3 not in taxel_force:
        raise ValueError("Exactly one ID3 module and its measured taxels are required")
    rotation = np.asarray(sdk_to_sensor, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError("sdk_to_sensor must be a calibrated proper rotation")
    values = np.asarray(taxel_force[3], dtype=float)
    order = np.asarray(taxel_order, dtype=int)
    if values.ndim != 2 or values.shape[1] != 3 or sorted(order.tolist()) != list(range(len(values))):
        raise ValueError("Explicit complete taxel permutation required")
    if not np.isfinite(scale_to_newton) or scale_to_newton <= 0:
        raise ValueError("Explicit positive force unit conversion required")
    module = np.asarray(module_force, dtype=float)
    if module.shape != (len(ids), 3) or not np.isfinite(values).all() or not np.isfinite(module[ids.index(3)]).all():
        raise ValueError("Invalid measured ID3 values")
    return {"real_taxel_N": (values[order] @ rotation.T * scale_to_newton).astype(np.float32),
            "real_force_N": (rotation @ module[ids.index(3)] * scale_to_newton).astype(np.float32)}


def inspect_replay(path):
    import h5py
    with h5py.File(path, "r") as f:
        fields = []
        f.visititems(lambda name, obj: fields.append(name) if isinstance(obj, h5py.Dataset) else None)
        required = ["real/tactile/taxel_force_sensor_N", "sim/tactile/taxel_depth_m",
                    "sim/tactile/module_force_sensor_N"]
        missing = [key for key in required if key not in fields]
        return {"file": str(path), "fields": fields, "missing_for_alignment": missing,
                "trainable_as_is": False,
                "reason": "Export stored taxel depth and the direct MuJoCo ID3 resultant into schema v2. Existing module_force_sensor_N can be reused if accumulated before RBF; missing depth must be supplied upstream. Do not import RBF taxel forces."}


def pair_nearest(real_time, sim_time, tolerance_s):
    """Offline nearest pairing; never uses -1 as an array index."""
    r, s = np.asarray(real_time), np.asarray(sim_time)
    if r.ndim != 1 or s.ndim != 1 or len(s) == 0 or not np.isfinite(r).all() or not np.isfinite(s).all() or np.any(np.diff(s) <= 0):
        raise ValueError("Finite times and strictly increasing simulation clock required")
    if tolerance_s <= 0:
        raise ValueError("Positive pairing tolerance required")
    right = np.clip(np.searchsorted(s, r), 0, len(s) - 1)
    left = np.maximum(right - 1, 0)
    idx = np.where(np.abs(s[left] - r) <= np.abs(s[right] - r), left, right)
    delta = s[idx] - r
    valid = (r >= s[0]) & (r <= s[-1]) & (np.abs(delta) <= tolerance_s)
    return np.where(valid, idx, -1), delta, valid
