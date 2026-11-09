import copy
import json

import numpy as np
import pytest
import torch

from tacdex.adapters import extract_id3, pair_nearest
from tacdex.data import EpisodeDataset, Normalizer, load_episode, read_manifest, validate_episode
from tacdex.losses import alignment_loss, pretrain_loss
from tacdex.models import FrozenProbe, ModelConfig, TactSpace
from tacdex.synthetic import generate


@pytest.fixture
def fixture_data(tmp_path):
    path = generate(tmp_path / "data", frames=18)
    m = read_manifest(path)
    episodes = [load_episode(r["resolved_path"]) for r in m["episodes"] if r["split"] == "align_train"]
    return path, m, episodes, Normalizer.fit(episodes)


def test_windows_are_causal_and_no_real_in_probe(fixture_data):
    _, m, episodes, norm = fixture_data
    config = ModelConfig(width=16, heads=2, layers=1, latent=8)
    ds = EpisodeDataset(m, "align_train", config, norm, paired=True)
    assert ds[0]["real"].shape == (10, 120, 3)
    np.testing.assert_allclose(ds[0]["real_force"], norm.transform(episodes[0]["real_force_N"][:10], "real_force"))
    np.testing.assert_allclose(ds[0]["sim_force"], norm.transform(episodes[0]["sim_force_N"][9:10], "sim_force"))
    probe = EpisodeDataset(m, "downstream_train", config, norm, paired=False)
    assert "real" not in probe[0] and "real_force" not in probe[0]
    assert probe[0]["sim_force"].shape == (1, 3)
    # Each history stays inside an episode; first valid index is exactly H-1.
    assert all(i >= 9 for _, i in ds.index)


@pytest.mark.parametrize("corruption", ["duplicate", "axes", "mismatch", "negative_depth", "nan", "missing_taxels", "missing_sim_force", "nan_sim_force", "rbf_force", "wrong_frame", "scalar_taxels"])
def test_bad_pairs_rejected(fixture_data, corruption):
    e = copy.deepcopy(fixture_data[2][0])
    if corruption == "duplicate": e["raw_event_id"][1] = e["raw_event_id"][0]
    elif corruption == "axes": e["metadata"]["axis_calibrated"] = False
    elif corruption == "mismatch": e["sim_depth_m"][:] = 0
    elif corruption == "negative_depth": e["sim_depth_m"][0, 0] = -1
    elif corruption == "nan": e["real_taxel_N"][0, 0, 0] = np.nan
    elif corruption == "missing_sim_force": del e["sim_force_N"]
    elif corruption == "nan_sim_force": e["sim_force_N"][0, 0] = np.nan
    elif corruption == "rbf_force": e["metadata"]["sim_force_source"] = "rbf_taxel_sum"
    elif corruption == "wrong_frame": e["metadata"]["force_frame"] = "world"
    elif corruption == "scalar_taxels": e["real_taxel_N"] = e["real_taxel_N"][..., :1]
    else: e["real_taxel_N"] = np.zeros((18, 3))
    with pytest.raises(ValueError): validate_episode(e, paired=True)


def test_split_leakage_rejected(fixture_data):
    path, _, _, _ = fixture_data
    m = json.loads(path.read_text())
    m["episodes"].append(dict(m["episodes"][0], split="test_id"))
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError): read_manifest(path)


def test_alignment_positives_and_collapsed_batch():
    z = torch.eye(4)
    ids = list("abcd")
    assert alignment_loss(z, z, ids) < alignment_loss(z, z.roll(1, 0), ids)
    with pytest.raises(ValueError): alignment_loss(z, z, ["same"] * 4)
    assert torch.isfinite(alignment_loss(z, z, ["a", "a", "b", "b"]))


def test_cross_reconstruction_and_freezing(fixture_data):
    _, m, _, norm = fixture_data
    c = ModelConfig(width=16, heads=2, layers=1, latent=8, dropout=.3)
    ds = EpisodeDataset(m, "align_train", c, norm, paired=True)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4)))
    model = TactSpace(c)
    z, decoded = model(batch)
    loss, terms = pretrain_loss(batch, z, decoded)
    assert all(k in terms for k in ("real_to_real", "real_to_sim", "sim_to_real", "sim_to_sim"))
    loss.backward()
    assert model.encoders["real"].value.weight.grad.abs().sum() > 0
    assert model.encoders["sim"].value.weight.grad.abs().sum() > 0
    for domain in ("real", "sim"):
        assert model.encoders[domain].force.weight.grad.abs().sum() > 0
        assert model.decode_forces[domain].weight.grad.abs().sum() > 0
        for source in ("real", "sim"):
            assert f"{source}_to_{domain}_force" in terms
    model.zero_grad(set_to_none=True)
    probe = FrozenProbe(model, 120).train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(probe.head.parameters(), lr=.01)
    probe(batch, "sim").square().mean().backward()
    opt.step()
    assert not model.training and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_pairing_rejects_out_of_range():
    idx, delta, valid = pair_nearest([-1, .49, 3], [0, .5, 1], .05)
    assert idx.tolist() == [-1, 1, -1]
    assert valid.tolist() == [False, True, False]


def test_id3_isolation_units_and_rotation():
    force = np.array([[999, 999, 999], [10, 20, 30]])
    result = extract_id3([6, 3], force, {3: [[1, 2, 3], [4, 5, 6]]},
                        scale_to_newton=.1, sdk_to_sensor=np.eye(3), taxel_order=[1, 0])
    np.testing.assert_allclose(result["real_force_N"], [1, 2, 3])
    np.testing.assert_allclose(result["real_taxel_N"][0], [.4, .5, .6])


def test_masks_exclude_bad_taxels(fixture_data):
    e = copy.deepcopy(fixture_data[2][0])
    e["real_mask"][0] = False
    e["real_taxel_N"][:, 0] = np.nan
    validate_episode(e, paired=True)
    stats = Normalizer.fit([e])
    assert np.isfinite(stats.stats["real"]).all()


def test_sim_force_changes_embedding_and_is_required(fixture_data):
    _, m, _, norm = fixture_data
    c = ModelConfig(width=16, heads=2, layers=1, latent=8, sim_history=3)
    ds = EpisodeDataset(m, "align_train", c, norm, paired=True)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4)))
    assert batch["sim_force"].shape == (4, 3, 3)
    model = TactSpace(c).eval()
    before = model.encode(batch, "sim")
    modified = dict(batch, sim_force=batch["sim_force"] + 5)
    assert not torch.allclose(before, model.encode(modified, "sim"))
    del batch["sim_force"]
    with pytest.raises(ValueError, match="force"):
        model.encode(batch, "sim")


def test_four_force_reconstruction_paths_have_gradients(fixture_data):
    _, m, _, norm = fixture_data
    c = ModelConfig(width=16, heads=2, layers=1, latent=8)
    batch = next(iter(torch.utils.data.DataLoader(EpisodeDataset(m, "align_train", c, norm, paired=True), batch_size=4)))
    model = TactSpace(c)
    z, decoded = model(batch)
    for source in decoded:
        for target in ("real_force", "sim_force"):
            decoded[source][target].retain_grad()
    loss, _ = pretrain_loss(batch, z, decoded)
    loss.backward()
    for source in decoded:
        for target in ("real_force", "sim_force"):
            assert decoded[source][target].grad.abs().sum() > 0


def test_no_force_ablation_remains_available(fixture_data):
    _, m, _, norm = fixture_data
    c = ModelConfig(width=16, heads=2, layers=1, latent=8, use_module_force=False)
    batch = next(iter(torch.utils.data.DataLoader(EpisodeDataset(m, "align_train", c, norm, paired=True), batch_size=4)))
    model = TactSpace(c)
    z, decoded = model(batch)
    assert set(decoded["sim"]) == {"real", "sim"}
    assert torch.isfinite(pretrain_loss(batch, z, decoded)[0])


def test_synthetic_provenance_cannot_be_used_as_real_data(fixture_data):
    e = copy.deepcopy(fixture_data[2][0])
    e["metadata"]["synthetic"] = False
    with pytest.raises(ValueError):
        validate_episode(e, paired=True)


def test_saved_direct_force_can_be_reused_from_rbf_replay(fixture_data):
    e = copy.deepcopy(fixture_data[2][0])
    # Provenance describes the SELECTED saved field, not every upstream output.
    e["metadata"].update(synthetic=False, depth_method="taxel_normal_raycast",
                         sim_force_source="mujoco_contact_solver", upstream_also_exported_rbf=True)
    validate_episode(e, paired=True)
