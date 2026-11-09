"""MuJoCo normal-ray depth proxy; independent of solver contact distances.

The ray starts behind the undeformed skin surface and points outward. Only
explicit indenter geoms are intersected. Over-depth, inside-solid or missed
rays are not interpreted as trustworthy contact measurements.
"""
import numpy as np


class MujocoDepthSampler:
    def __init__(self, model, site_names, object_geom_names, max_depth_m=0.003):
        import mujoco
        self.mj, self.model = mujoco, model
        if not object_geom_names or max_depth_m <= 0:
            raise ValueError("Explicit object geometry names and positive ray depth required")
        self.site_ids = [model.site(name).id for name in site_names]
        self.geom_ids = [model.geom(name).id for name in object_geom_names]
        if len(set(self.site_ids)) != len(self.site_ids):
            raise ValueError("Duplicate taxel sites")
        self.max_depth = float(max_depth_m)

    def read(self, data):
        depth = np.zeros(len(self.site_ids), dtype=np.float32)
        hit = np.zeros(len(self.site_ids), dtype=bool)
        # mj_ray does not accept an arbitrary allow-list. mj_rayMesh / mju_rayGeom
        # operate on individual geoms without mutating the user's collision groups.
        for j, sid in enumerate(self.site_ids):
            normal = np.asarray(data.site_xmat[sid]).reshape(3, 3)[:, 2]
            origin = data.site_xpos[sid] - self.max_depth * normal
            distances = []
            for gid in self.geom_ids:
                typ = int(self.model.geom_type[gid])
                if typ == int(self.mj.mjtGeom.mjGEOM_MESH):
                    distance = self.mj.mj_rayMesh(self.model, data, gid, origin, normal)
                elif typ in {int(self.mj.mjtGeom.mjGEOM_PLANE), int(self.mj.mjtGeom.mjGEOM_SPHERE),
                             int(self.mj.mjtGeom.mjGEOM_CAPSULE), int(self.mj.mjtGeom.mjGEOM_ELLIPSOID),
                             int(self.mj.mjtGeom.mjGEOM_CYLINDER), int(self.mj.mjtGeom.mjGEOM_BOX)}:
                    distance = self.mj.mju_rayGeom(data.geom_xpos[gid], data.geom_xmat[gid],
                                    self.model.geom_size[gid], origin, normal, typ)
                else:
                    raise ValueError(f"Unsupported ray geometry type {typ}; no silent approximation")
                if distance >= 0:
                    distances.append(distance)
            if distances:
                distance = min(distances)
                depth[j] = max(0, self.max_depth - distance)
                hit[j] = True
        return {"depth_m": depth, "ray_hit": hit}


def id3_sites(model, expected_taxels=120):
    names = [f"taxel_03_{i:03d}" for i in range(expected_taxels)]
    for name in names:
        model.site(name)  # Fail before recording on missing/mismatched taxel order.
    return names
