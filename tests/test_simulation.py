import numpy as np
import pytest

from tacdex.simulation import MujocoDepthSampler


def test_mujoco4_rays_measure_each_taxel_and_ignore_other_geoms():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <site name="a" pos="0 0 0"/><site name="b" pos="0.01 0 0"/>
      <geom name="object" type="sphere" size="0.005" pos="0 0 0.004"/>
      <geom name="other_finger" type="sphere" size="0.005" pos="0.01 0 0.004"/>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    read = MujocoDepthSampler(model, ["a", "b"], ["object"]).read(data)
    np.testing.assert_allclose(read["depth_m"], [.001, 0], atol=1e-8)
    assert read["ray_hit"].tolist() == [True, False]


def test_noncontact_ray_reports_zero():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <site name="a"/><geom name="object" type="box" size=".01 .01 .01" pos="0 0 .02"/>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert MujocoDepthSampler(model, ["a"], ["object"]).read(data)["depth_m"].item() == 0
