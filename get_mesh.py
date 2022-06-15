from nerf_grasping.sim import ig_objects

from nerf_grasping import grasp_opt, grasp_utils, mesh_utils
import os
import scipy.spatial
import torch
import trimesh
import numpy as np

def main(
    obj_name="banana",
    outfile=None,
    level_set=50,
):
    if obj_name == "banana":
        obj = ig_objects.Banana
    elif obj_name == "box":
        obj = ig_objects.Box
    elif obj_name == "teddy_bear":
        obj = ig_objects.TeddyBear
        obj.use_centroid = True
    elif obj_name == "powerdrill":
        obj = ig_objects.PowerDrill

    if outfile is None:
        outfile = obj_name

    object_nerf = ig_objects.load_nerf(obj.workspace, obj.bound, obj.scale)

    verts, faces, normals, _ = mesh_utils.marching_cubes(object_nerf, level_set=level_set)
    approx_mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals)

    T = np.eye(4)
    R = scipy.spatial.transform.Rotation.from_euler('Y', [-np.pi / 2]).as_matrix()
    R = R @ scipy.spatial.transform.Rotation.from_euler('X',
                                                    [-np.pi / 2]).as_matrix()
    # Apply inverse transform to map approximate mesh -> ig frame.
    T[:3, :3] = R.reshape(3,3).T
    approx_mesh.apply_transform(T)

    approx_mesh = mesh_utils.poisson_mesh(approx_mesh)

    approx_mesh.export(f'grasp_data/meshes/{outfile}_{level_set:.0f}.obj')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_name", "--o", help="object to use", default="banana")
    parser.add_argument("--level_set", "--l", default=50., type=float)
    parser.add_argument("--outfile", "--out", default=None)

    args = parser.parse_args()

    print(args)

    main(**vars(args))