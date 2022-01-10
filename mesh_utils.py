"""
Module implementing some mesh utilities for NeRFs, including
a wrapper for marching cubes and some IoU calculations.
"""
import numpy as np
import pypoisson
import torch
import trimesh

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

def poisson_mesh(mesh, depth=10, samples_per_node=5.):
    """
    Performs Poisson reconstruction to generate a (hopefully) watertight
    version of a mesh. Note that it can fail sometimes, so make sure to check output.
    """
    trimesh.repair.fix_inversion(mesh)
    faces, verts = pypoisson.poisson_reconstruction(
        np.array(mesh.triangles_center),
        np.array(mesh.face_normals), depth=depth,
        samples_per_node=samples_per_node)

    return trimesh.Trimesh(verts, faces)

def iou(x, y):
    """
    Computes the IoU
    """
    intersection = x.intersection(y, engine='scad')
    union = x.union(y, engine='scad')

    trimesh.repair.fix_inversion(union)
    trimesh.repair.fix_winding(union)
    trimesh.repair.fix_inversion(intersection)
    trimesh.repair.fix_winding(intersection)

    union_poisson = poisson_mesh(union)
    int_poisson = poisson_mesh(intersection)

    trimesh.repair.fix_inversion(union_poisson)
    trimesh.repair.fix_winding(union_poisson)
    trimesh.repair.fix_inversion(int_poisson)
    trimesh.repair.fix_winding(int_poisson)

    print(int_poisson.is_watertight, union_poisson.is_watertight)

    success = int_poisson.is_watertight and union_poisson.is_watertight

    return int_poisson.volume / union_poisson.volume, success

def plot_density_contours(nerf,
                          level=0.,
                          plane='xy',
                          urange=np.linspace(-0.25, 0.25, num=500),
                          vrange=np.linspace(-0.25, 0.25, num=500),
                          ax=None,
                          log_scale=False,
                          colorbar=True):
    """
    Generates a contour plot in an x-y plane of a NeRF's density channel
    over a desired range.

    Args:
        nerf: a nerf_shared.NeRF object whose density will be plotted.
        z_level: the z-coordinate of the plane to be plotted.
        xrange: a numpy array defining the x coordinates to be queried.
        yrange: a numpy array defining the y coordinates to be queried.
        ax: (optional) argument allowing plotting on a predefined pyplot axis.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(10,10))

    if plane == 'xy':
        X, Y = np.meshgrid(urange, vrange)
        Z = level * np.ones_like(X)
    elif plane == 'yz':
        Y, Z = np.meshgrid(urange, vrange)
        X = level * np.ones_like(Y)
    elif plane == 'zx':
        Z, X = np.meshgrid(urange, vrange)
        Y = level * np.ones_like(Z)
    else:
        raise ValueError('plane must be \'xy\', \'yz\', or \'zx\'.')

    device, dtype = next(nerf.parameters()).device, next(nerf.parameters()).dtype

    query_points = torch.from_numpy(np.stack([X,Y,Z], axis=-1)).to(device, dtype)

    densities = torch.nn.ReLU()(nerf.get_density(query_points))

    if log_scale:
        levels = np.logspace(-1, 2.7, num=25)
    else:
        levels = np.linspace(0.1, 500, num=25)

    if plane == 'xy':
        c = ax.contour(X, Y, densities.cpu().numpy(), levels=levels)
    elif plane == 'yz':
        c = ax.contour(Y, Z, densities.cpu().numpy(), levels=levels)
    elif plane == 'zx':
        c = ax.contour(Z, X, densities.cpu().numpy(), levels=levels)

    if colorbar:
        plt.colorbar(c)

    return ax
