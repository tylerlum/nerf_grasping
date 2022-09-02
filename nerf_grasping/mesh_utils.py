"""
Module implementing some mesh utilities for NeRFs, including
a wrapper for marching cubes and some IoU calculations.
"""
import numpy as np
import pypoisson
import torch
import trimesh

from nerf_grasping import grasp_utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def marching_cubes(
    nerf,
    lower=np.array([-0.1, 0.01, -0.1]),
    upper=np.array([0.1, 0.2, 0.1]),
    level_set=0.5,
    num_points=150,
):
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

    dx, dy, dz = (upper - lower) / (num_points - 1)

    # Meshgrid the points to get all query points.
    xv, yv, zv = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")

    # Load points into torch, flatten.
    test_points = (
        torch.stack([torch.from_numpy(arr) for arr in (xv, yv, zv)], axis=-1)
        .reshape(-1, num_points, 3)
        .to(next(nerf.parameters()).device, next(nerf.parameters()).dtype)
    )

    # Query the network for density values at each point and reshape.
    test_densities = torch.nn.ReLU()(
        nerf.density(test_points)
        .reshape(num_points, num_points, num_points)
        .cpu()
        .detach()
    ).numpy()

    verts, faces, normals, values = measure.marching_cubes(
        test_densities, level_set, spacing=(dx, dy, dz)
    )

    verts[:, 0] -= upper[0]
    verts[:, -1] -= upper[-1]

    return verts, faces, -normals, values


def poisson_mesh(mesh):
    """
    Performs Poisson reconstruction to generate a (hopefully) watertight
    version of a mesh. Note that it can fail sometimes, so make sure to check output.
    """
    trimesh.repair.fix_inversion(mesh)
    # faces, verts = pypoisson.poisson_reconstruction(
    #     np.array(mesh.triangles_center),
    #     np.array(mesh.face_normals),
    # )
    # return trimesh.Trimesh(verts, faces)
    return mesh


def iou(x, y):
    """
    Computes the IoU of two meshes.
    """
    intersection = x.intersection(y, engine="scad")
    union = x.union(y, engine="scad")

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

    success = int_poisson.is_watertight and union_poisson.is_watertight

    return int_poisson.volume / union_poisson.volume, success


def get_query_points(plane, urange, vrange, level):
    """
    Helper function that gets a set of query points in a plane.

    Args:
        plane: string specifying which plane to query, can be 'xy', 'yz', or 'zx'.
        urange: numpy array with entries to query in first plane axis.
        vrange: numpy array "" second plane axis.
        level: the coordinate of the plane along its normal (e.g., for plane='xy', and
            level=5., all coordinates will have z=5.
    """
    if plane == "xy":
        X, Y = np.meshgrid(urange, vrange)
        Z = level * np.ones_like(X)
    elif plane == "yz":
        Y, Z = np.meshgrid(urange, vrange)
        X = level * np.ones_like(Y)
    elif plane == "zx":
        Z, X = np.meshgrid(urange, vrange)
        Y = level * np.ones_like(Z)
    else:
        raise ValueError("plane must be 'xy', 'yz', or 'zx'.")

    return torch.from_numpy(np.stack([X, Y, Z], axis=-1)), (X, Y, Z)


def plot_density_contours(
    nerf,
    level=0.0,
    plane="xy",
    urange=np.linspace(-0.25, 0.25, num=500),
    vrange=np.linspace(-0.25, 0.25, num=500),
    ax=None,
    log_scale=False,
    colorbar=True,
):
    """
    Generates a contour plot in a desired plane of a NeRF's density channel
    over a desired range.

    Args:
        nerf: a nerf_shared.NeRF object whose density will be plotted.
        level: the z-coordinate of the plane to be plotted.
        urange: a numpy array defining the coordinates to be queried in the first axis.
        vrange: a numpy array "" the second axis.
        ax: (optional) argument allowing plotting on a predefined pyplot axis.
        log_scale: boolean flagging if the contours should be sampled in log scale.
        colorbar: boolean flag to add colorbar to the plot.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))

    device, dtype = next(nerf.parameters()).device, next(nerf.parameters()).dtype
    query_points, (X, Y, Z) = get_query_points(plane, urange, vrange, level)

    query_points = query_points.to(device, dtype)

    densities = nerf.density(query_points)

    if log_scale:
        levels = np.logspace(-1, 2.7, num=25)
    else:
        levels = np.linspace(0.1, 500, num=25)

    if plane == "xy":
        c = ax.contour(X, Y, densities.cpu().numpy(), levels=levels)
    elif plane == "yz":
        c = ax.contour(Y, Z, densities.cpu().numpy(), levels=levels)
    elif plane == "zx":
        c = ax.contour(Z, X, densities.cpu().numpy(), levels=levels)

    if colorbar:
        plt.colorbar(c)

    return ax


def pixel_plot(
    nerf,
    level=0.0,
    plane="xy",
    urange=np.linspace(-0.25, 0.25, num=500),
    vrange=np.linspace(-0.25, 0.25, num=500),
    ax=None,
    colorbar=True,
):
    """
    Generates a pixel plot in a desired plane of a NeRF's density channel
    over a desired range.

    Args:
        nerf: a nerf_shared.NeRF object whose density will be plotted.
        level: the z-coordinate of the plane to be plotted.
        urange: a numpy array defining the coordinates to be queried in the first axis.
        vrange: a numpy array "" the second axis.
        ax: (optional) argument allowing plotting on a predefined pyplot axis.
        colorbar: boolean flag to add colorbar to the plot.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))

    device, dtype = next(nerf.parameters()).device, next(nerf.parameters()).dtype
    query_points, _ = get_query_points(plane, urange, vrange, level)

    query_points = query_points.to(device, dtype)

    densities = nerf.density(query_points)

    c = ax.imshow(densities.cpu().numpy(), origin="lower")
    if colorbar:
        plt.colorbar(c)

    return ax


def gradient_norm_plot(
    nerf,
    level=0.0,
    plane="xy",
    urange=np.linspace(-0.25, 0.25, num=500),
    vrange=np.linspace(-0.25, 0.25, num=500),
    ax=None,
    colorbar=True,
):
    """
    Plots the gradient norm, sampled in a desired plane.

    Args:
        nerf: a nerf_shared.NeRF object whose density will be plotted.
        level: the z-coordinate of the plane to be plotted.
        urange: a numpy array defining the coordinates to be queried in the first axis.
        vrange: a numpy array "" the second axis.
        ax: (optional) argument allowing plotting on a predefined pyplot axis.
        colorbar: boolean flag to add colorbar to the plot.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))

    device, dtype = next(nerf.parameters()).device, next(nerf.parameters()).dtype
    query_points, (X, Y, Z) = get_query_points(plane, urange, vrange, level)

    query_points = query_points.to(device, dtype)

    query_points.requires_grad = True

    densities = nerf.density(query_points)

    density_grads = torch.autograd.grad(torch.sum(densities), query_points)[0]
    grad_norms = torch.log(torch.norm(density_grads, dim=-1))

    c = ax.imshow(grad_norms.cpu().numpy(), origin="lower")
    if colorbar:
        plt.colorbar(c)

    return ax


def gradient_plot(
    nerf,
    level=0.0,
    plane="xy",
    urange=np.linspace(-0.25, 0.25, num=500),
    vrange=np.linspace(-0.25, 0.25, num=500),
    ax=None,
    colorbar=True,
):
    """
    Generates a quiver plot of the NeRF density gradients, in a desired plane.

    Args:
        nerf: a nerf_shared.NeRF object whose density will be plotted.
        level: the z-coordinate of the plane to be plotted.
        urange: a numpy array defining the coordinates to be queried in the first axis.
        vrange: a numpy array "" the second axis.
        ax: (optional) argument allowing plotting on a predefined pyplot axis.
        colorbar: boolean flag to add colorbar to the plot.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))

    device, dtype = next(nerf.parameters()).device, next(nerf.parameters()).dtype
    query_points, (X, Y, Z) = get_query_points(plane, urange, vrange, level)

    query_points = query_points.to(device, dtype)

    query_points.requires_grad = True

    densities = nerf.density(query_points)

    density_grads = torch.autograd.grad(torch.sum(densities), query_points)[0]
    grad_norms = torch.norm(density_grads, dim=-1)
    grad_norms = torch.where(
        grad_norms < 1e-2, 1e-2 * torch.ones_like(grad_norms), grad_norms
    )
    density_dirs = (density_grads / (grad_norms[..., None])).cpu()

    if plane == "xy":
        ax.quiver(X, Y, density_dirs[..., 0], density_dirs[..., 1], grad_norms.cpu())

    return ax


def get_grasp_points(mesh, grasp_vars, residual_dirs=True):
    """Takes a batch of grasp origins/dirs and computes their intersections and normals."""
    # Unpack ray origins/dirs.
    rays_o, rays_d = grasp_vars[..., :3], grasp_vars[..., 3:]

    B, n_f, _ = rays_o.shape
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    # Transform ray directions if using residual directions.
    if residual_dirs:
        rays_d = grasp_utils.res_to_true_dirs(
            rays_o, rays_d, torch.from_numpy(mesh.centroid).to(rays_o)
        )

    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Cast to numpy, reshape.
    rays_o_np, rays_d_np = rays_o.cpu().numpy(), rays_d.cpu().numpy()

    grasp_points, grasp_normals = np.zeros_like(rays_o_np), np.zeros_like(rays_d_np)
    grasp_mask = np.zeros_like(rays_o_np[..., 0])

    # TODO: handle when rays miss.
    hit_points, ray_ids, face_ids = mesh.ray.intersects_location(
        rays_o_np, rays_d_np, multiple_hits=False
    )

    grasp_points[ray_ids, :] = hit_points
    grasp_normals[ray_ids, :] = -mesh.face_normals[face_ids]
    grasp_mask[ray_ids] = 1

    grasp_points = torch.from_numpy(grasp_points).reshape(B, n_f, 3).to(rays_o)
    grasp_normals = torch.from_numpy(grasp_normals).reshape(B, n_f, 3).to(rays_d)
    grasp_mask = torch.from_numpy(grasp_mask).reshape(B, n_f).to(rays_o).bool()

    return grasp_points, grasp_normals, grasp_mask


def correct_z_dists(mesh, rays_o, rays_d, mesh_config):

    if isinstance(rays_o, torch.Tensor):
        rays_o_np, rays_d_np = rays_o.cpu().numpy(), rays_d.cpu().numpy()
    else:
        rays_o_np, rays_d_np = rays_o, rays_d

    rays_o_np, rays_d_np = rays_o_np.reshape(-1, 3), rays_d_np.reshape(-1, 3)

    hit_points, ray_ids, face_ids = mesh.ray.intersects_location(
        rays_o_np, rays_d_np, multiple_hits=False
    )

    dists = np.linalg.norm(rays_o_np - hit_points, axis=-1)
    rays_o_corrected = (
        rays_o_np + (dists - mesh_config.des_z_dist).reshape(3, 1) * rays_d_np
    )

    if isinstance(rays_o, torch.Tensor):
        rays_o_corrected = torch.from_numpy(rays_o_corrected).to(rays_o)

    return rays_o_corrected
