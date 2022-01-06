"""
Module implementing some mesh utilities for NeRFs, including
a wrapper for marching cubes and some IoU calculations.
"""
import numpy as np
import torch
import trimesh

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pypoisson import poisson_reconstruction
from skimage import measure

def marching_cubes(nerf,
                   lower=-0.25*np.ones(3),
                   upper=0.25*np.ones(3),
                   level_set=0.5,
                   num_points=50):
    """
    Generates a mesh of a desired density level set of a NeRF.
    Assumes mesh is origin-centered.

    Args:
        nerf: a nerf_shared.nerf.NeRF model to be meshed.
        lower: a numpy array specifying the lower corner of the meshing box.
        upper: numpy array specifying upper corner.
        level_set: float specifying the desired level set.
        num_points: number of points per axis.
    """
    # Create grid points for each axis.
    grid_x = np.linspace(lower[0], upper[0], num=num_points)
    grid_y = np.linspace(lower[1], upper[1], num=num_points)
    grid_z = np.linspace(lower[2], upper[2], num=num_points)

    dx, dy, dz = (upper-lower) / (num_points - 1)

    # Meshgrid the points to get all query points.
    xv, yv, zv = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

    # Load points into torch, flatten.
    test_points = torch.stack(
        [torch.from_numpy(arr) for arr in (xv, yv, zv)], axis=-1).reshape(
        -1, num_points, 3).to(next(nerf.parameters()).device,
                              next(nerf.parameters()).dtype)

    # Query the network for density values at each point and reshape.
    test_densities = torch.nn.ReLU()(nerf.get_density(test_points).reshape(
        num_points, num_points, num_points).cpu()).numpy()

    verts, faces, normals, values = measure.marching_cubes(
        test_densities, level_set, spacing=(dx,dy,dz))

    verts -= upper

    return verts, faces, normals, values

def get_axis(lower=-0.25*np.ones(3), upper=0.25*np.ones(3), scale=1.1):
    fig = plt.figure(figsize=(6,6))
    ax = ax = Axes3D(fig)

    # set plot view angle
    ax.view_init(elev=30,azim=0)

    ax.set_xlim(lower[0]*scale, upper[0]*scale)
    ax.set_ylim(lower[1]*scale, upper[1]*scale)
    ax.set_zlim(lower[2]*scale, upper[2]*scale)

    return fig, ax

def plot_mesh(ax, verts, faces):
    """
    Plots a mesh on a desired 3D axis.
    """
    ax.add_collection3d(Poly3DCollection(verts[faces]))

def iou(x, y):
    """
    Computes the IoU
    """
    intersection = trimesh.boolean.intersection([x,y], engine='blender')
    union = trimesh.boolean.union([x,y], engine='blender')
    int_normals = trimesh.geometry.weighted_vertex_normals(
                len(intersection.vertices),
                intersection.faces,
                intersection.face_normals,
                intersection.face_angles)

    union_normals = trimesh.geometry.weighted_vertex_normals(
                    len(union.vertices),
                    union.faces,
                    union.face_normals,
                    union.face_angles)

    union_faces, union_verts = poisson_reconstruction(union.vertices,
                                                      union_normals, depth=5)
    int_faces, int_verts = poisson_reconstruction(intersection.vertices,
                                                  int_normals, depth=5)

    union_poisson = trimesh.Trimesh(union_verts, union_faces)
    int_poisson = trimesh.Trimesh(int_verts, int_faces)

    trimesh.repair.fix_inversion(union_poisson)
    trimesh.repair.fix_winding(union_poisson)
    trimesh.repair.fix_inversion(int_poisson)
    trimesh.repair.fix_winding(int_poisson)

    return int_poisson.volume / union_poisson.volume