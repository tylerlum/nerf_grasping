from nerf_grasping.sim import ig_objects

from nerf_grasping import config, grasp_opt, grasp_utils, mesh_utils

import dcargs
import os
import scipy.spatial
import torch
import trimesh
import numpy as np


def main(exp_config: config.Experiment):
    obj = ig_objects.load_object(exp_config)
    outfile = config.mesh_file(exp_config)

    object_nerf = ig_objects.load_nerf(obj.workspace, obj.bound, obj.scale)

    verts, faces, normals, _ = mesh_utils.marching_cubes(
        object_nerf, level_set=exp_config.level_set
    )
    approx_mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals)

    # Attempt to center mesh at its centroid.
    approx_mesh.apply_translation(-approx_mesh.centroid)

    # Apply inverse transform to map approximate mesh -> ig frame.
    T = np.eye(4)
    R = scipy.spatial.transform.Rotation.from_euler("Y", [-np.pi / 2]).as_matrix()
    R = R @ scipy.spatial.transform.Rotation.from_euler("X", [-np.pi / 2]).as_matrix()
    T[:3, :3] = R.reshape(3, 3).T
    approx_mesh.apply_transform(T)

    # Run poisson reconstruction.
    approx_mesh = mesh_utils.poisson_mesh(approx_mesh)
    approx_mesh.export(outfile)

    print(f"Saving mesh to: {outfile}")


if __name__ == "__main__":
    exp_config = dcargs.cli(config.Experiment)
    main(exp_config)
