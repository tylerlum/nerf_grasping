# %%
import numpy as np
from nerf_grasping.optimizer_utils import (
    AllegroGraspConfig,
)
import pathlib
import trimesh
from localscope import localscope
from functools import partial
import torch

# %%
# PARAMS
INIT_GRASP_CONFIG_DICT_PATH = pathlib.Path(
    "data/2023-11-23_rubikscuberepeat_labelnoise_2/evaled_grasp_config_dicts/ddg-gd_rubik_cube_poisson_004_00_0_1000.npy"
)
assert INIT_GRASP_CONFIG_DICT_PATH.exists()

MESHDATA_PATH = pathlib.Path("data/meshdata")
assert MESHDATA_PATH.exists()
THIS_MESH_PATH = (
    MESHDATA_PATH / "ddg-gd_rubik_cube_poisson_004" / "coacd" / "decomposed.obj"
)
assert THIS_MESH_PATH.exists()
OBJECT_SCALE = 0.1

# %%
# CONSTANTS
N_JOINTS = 16
N_FINGERS = 4
N_XYZ = 3
N_QUAT = 4
N_GRASPS = 10


# %%
def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


# %%
mesh = trimesh.load(THIS_MESH_PATH, force="mesh")
mesh.apply_scale(OBJECT_SCALE)

# %%
init_grasp_config_dict = np.load(INIT_GRASP_CONFIG_DICT_PATH, allow_pickle=True).item()

# %%
init_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(init_grasp_config_dict)[
    :N_GRASPS
]

# %%
assert_equals(init_grasp_configs.wrist_pose.shape, (N_GRASPS, N_XYZ + N_QUAT))
assert_equals(init_grasp_configs.joint_angles.shape, (N_GRASPS, N_JOINTS))
assert_equals(init_grasp_configs.grasp_dirs.shape, (N_GRASPS, N_FINGERS, N_XYZ))
assert_equals(
    init_grasp_configs.fingertip_transforms.shape, (N_GRASPS, N_FINGERS, N_XYZ + N_QUAT)
)

# %%
import plotly.graph_objects as go
from plotly.graph_objects import Figure


@localscope.mfc(allowed=["N_FINGERS", "N_XYZ"])
def plot_grasp_config(mesh, grasp_config: AllegroGraspConfig, idx: int = 0) -> Figure:
    fig = go.Figure()

    wrist_pos = grasp_config.wrist_pose[idx, :3].detach().cpu().numpy()
    fingertip_positions = (
        grasp_config.fingertip_transforms[idx, :, :3].detach().cpu().numpy()
    )
    grasp_dirs = grasp_config.grasp_dirs[idx, :, :3].detach().cpu().numpy()
    assert_equals(wrist_pos.shape, (N_XYZ,))
    assert_equals(fingertip_positions.shape, (N_FINGERS, N_XYZ))
    assert_equals(grasp_dirs.shape, (N_FINGERS, N_XYZ))

    fig.add_trace(
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            name="mesh",
            opacity=0.5,
            color="lightpink",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[wrist_pos[0]],
            y=[wrist_pos[1]],
            z=[wrist_pos[2]],
            mode="markers",
            marker=dict(size=5, color="red"),
            name="wrist",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=fingertip_positions[:, 0],
            y=fingertip_positions[:, 1],
            z=fingertip_positions[:, 2],
            mode="markers",
            marker=dict(size=5, color="blue"),
            name="fingertips",
        )
    )
    DELTA = 0.02
    target_fingertip_positions = fingertip_positions + grasp_dirs * DELTA
    for i in range(N_FINGERS):
        fig.add_trace(
            go.Scatter3d(
                x=[fingertip_positions[i, 0], target_fingertip_positions[i, 0]],
                y=[fingertip_positions[i, 1], target_fingertip_positions[i, 1]],
                z=[fingertip_positions[i, 2], target_fingertip_positions[i, 2]],
                mode="lines",
                line=dict(color="blue", width=5),
                name="fingertips dirs",
            )
        )
    return fig


@localscope.mfc(allowed=["N_GRASPS", "N_FINGERS", "N_XYZ"])
def plot_grasps(mesh, grasps: torch.Tensor, idx: int = 0) -> Figure:
    fig = go.Figure()

    assert_equals(grasps.shape[1:], (N_FINGERS, N_XYZ * 2))
    fingertip_positions = grasps[idx, :, :3]
    grasp_dirs = grasps[idx, :, 3:]
    assert_equals(fingertip_positions.shape, (N_FINGERS, N_XYZ))
    assert_equals(grasp_dirs.shape, (N_FINGERS, N_XYZ))

    fig.add_trace(
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            name="mesh",
            opacity=0.5,
            color="lightpink",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=fingertip_positions[:, 0],
            y=fingertip_positions[:, 1],
            z=fingertip_positions[:, 2],
            mode="markers",
            marker=dict(size=5, color="blue"),
            name="fingertips",
        )
    )
    DELTA = 0.02
    target_fingertip_positions = fingertip_positions + grasp_dirs * DELTA
    for i in range(N_FINGERS):
        fig.add_trace(
            go.Scatter3d(
                x=[fingertip_positions[i, 0], target_fingertip_positions[i, 0]],
                y=[fingertip_positions[i, 1], target_fingertip_positions[i, 1]],
                z=[fingertip_positions[i, 2], target_fingertip_positions[i, 2]],
                mode="lines",
                line=dict(color="blue", width=5),
                name="fingertips dirs",
            )
        )
    return fig


fig = plot_grasp_config(mesh, init_grasp_configs, idx=0)
fig.show()

# %%

mu_0 = torch.cat(
    [
        init_grasp_configs.fingertip_transforms.translation()[0],
        init_grasp_configs.grasp_dirs[0],
    ],
    dim=-1,
).flatten()
Sigma_0 = torch.eye(N_FINGERS * N_XYZ * 2) * 1e-3
assert_equals(mu_0.shape, (N_FINGERS * N_XYZ * 2,))
assert_equals(Sigma_0.shape, (N_FINGERS * N_XYZ * 2, N_FINGERS * N_XYZ * 2))


# %%
def box_projection(x, object_bounds):
    B, _ = x.shape
    x = x.reshape(B, 3, -1)

    lower = torch.tensor([oo[0] for oo in object_bounds]).to(x)
    upper = torch.tensor([oo[1] for oo in object_bounds]).to(x)

    x[..., :3] = x[..., :3].clamp(lower, upper)

    return x.reshape(B, -1)


def optimize_cem(
    cost,
    mu_0,
    Sigma_0,
    num_iters=25,
    num_samples=250,
    elite_frac=0.1,
    projection=None,
):
    """
    Implements the cross-entropy method to optimize a given cost function.
    Args:
        cost: a cost function mapping variables x to their cost J(x).
        mu_0: mean of the initial sample distribution.
        Sigma_0: covariance of initial sample distribution.
        num_iters: number of iterations of CEM to run.
        num_samples: number of samples to draw per iteration.
        elite_frac: fraction of samples to use to re-fit sample distribution, in (0, 1).
    """
    n = mu_0.shape[0]
    mu, Sigma = mu_0, Sigma_0
    num_elite = int(elite_frac * num_samples)
    device = mu_0.device
    cost_history = []
    best_point = None
    best_cost = torch.inf
    for ii in range(num_iters):
        # Sample points from current distribution.
        x = (
            mu.reshape(1, n, 1)
            + torch.linalg.cholesky(Sigma).reshape(1, n, n)
            @ torch.randn(num_samples, n, 1, device=device)
        ).reshape(num_samples, n)

        if projection:
            x = projection(x)

        # Evaluate costs of each point.
        with torch.no_grad():
            cost_vals = cost(x)
        cost_history.append(cost_vals)
        print(
            "minimum cost_val:",
            torch.min(cost_vals),
            "mean cost_val:",
            torch.mean(cost_vals),
        )
        # Get elite indices.
        _, inds = torch.sort(cost_vals)
        elite_inds = inds[:num_elite]

        if cost_vals[inds[0]] < best_cost:
            best_point = x[inds[0]]

        # Refit the sample distribution.
        mu = torch.mean(x[elite_inds, :], dim=0)
        residuals = x[elite_inds, :] - mu.reshape(1, n)
        Sigma = (1 / (num_elite - 1)) * torch.sum(
            torch.stack(
                [
                    residuals[ii, :][:, None] @ residuals[ii, :][None, :]
                    for ii in range(num_elite)
                ],
                dim=0,
            ),
            dim=0,
        ) + 1e-8 * torch.eye(n, device=device)

    return mu, Sigma, cost_history, best_point


def get_grasp_points(mesh, grasp_vars, residual_dirs=False):
    """Takes a batch of grasp origins/dirs and computes their intersections and normals."""
    # Unpack ray origins/dirs.
    rays_o, rays_d = grasp_vars[..., :3], grasp_vars[..., 3:]

    B, n_f, _ = rays_o.shape
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    # Transform ray directions if using residual directions.
    if residual_dirs:
        raise ValueError("Not implemented")
        # rays_d = grasp_utils.res_to_true_dirs(rays_o, rays_d, torch.from_numpy(mesh.centroid).to(rays_o))

    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Cast to numpy, reshape.
    rays_o_np, rays_d_np = rays_o.detach().cpu().numpy(), rays_d.detach().cpu().numpy()

    # Put ray origins into mesh frame.
    # rays_o_np = rays_o_np - mesh.centroid.reshape(1, 3)

    grasp_points, grasp_normals = np.zeros_like(rays_o_np), np.zeros_like(rays_d_np)
    grasp_mask = np.zeros_like(rays_o_np[..., 0])

    # TODO: handle when rays miss.
    hit_points, ray_ids, face_ids = mesh.ray.intersects_location(
        rays_o_np, rays_d_np, multiple_hits=False
    )

    grasp_points[ray_ids, :] = hit_points
    grasp_normals[ray_ids, :] = -mesh.face_normals[face_ids]
    grasp_mask[ray_ids] = 1

    # Put rays back into world frame.
    # grasp_points = grasp_points + mesh.centroid.reshape(1, 3)
    #
    grasp_points = torch.from_numpy(grasp_points).reshape(B, n_f, 3).to(rays_o)
    grasp_normals = torch.from_numpy(grasp_normals).reshape(B, n_f, 3).to(rays_d)
    grasp_mask = torch.from_numpy(grasp_mask).reshape(B, n_f).to(rays_o).bool()

    # new_grasp_mask = grasp_mask.any(-1)
    # new_grasps = torch.cat(
    #     [grasp_points[new_grasp_mask], grasp_normals[new_grasp_mask]], dim=-1
    # )
    # breakpoint()
    # fig = plot_grasps(mesh, new_grasps.detach(), idx=-1)
    # fig.show()

    return grasp_points, grasp_normals, grasp_mask


def rot_from_vec(n_z, start_vec=None):
    """
    Creates rotation matrix which maps the basis vector e_3 to a vector n_z.
    Gets poorly conditioned when n_z ≅ ±e_3.
    Args:
        n_z: Batch of normal dirs, shape [B, 3].
    """
    # Construct constants.
    n_z = n_z.reshape(-1, 3)
    Identity = (
        torch.eye(3, device=n_z.device).reshape(1, 3, 3).expand(n_z.shape[0], 3, 3)
    )
    if start_vec is None:
        start_vec = Identity[:, :, 2]

    # Compute cross product to find axis of rotation.
    v = torch.cross(start_vec, n_z, dim=-1)
    theta = torch.arccos(torch.sum(start_vec * n_z, dim=-1)).reshape(-1, 1, 1)
    K = skew(v)

    ans = Identity + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K

    return ans


def skew(v):
    """
    Returns the skew-symmetric form of a batch of 3D vectors v (shape [B, 3]).
    """
    v = v.reshape(-1, 3)

    K = torch.zeros(v.shape[0], 3, 3, device=v.device)

    K[:, 0, 1] = -v[:, 2]
    K[:, 0, 2] = v[:, 1]
    K[:, 1, 0] = v[:, 2]
    K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]
    K[:, 2, 1] = v[:, 0]

    return K


def grasp_matrix(grasp_points, normals):
    """
    Constructs a grasp matrix for the object represented by the NeRF density,
    evaluated at a set of grasp points.
    Args:
        nerf: NeRF model whose density represents the object.
        grasp_points: a list of grasp points (torch.Tensors in 3D) at which to
            construct the grasp matrix, shape [B, n_f, 3].
    Returns a grasp matrix (torch.Tensor) for the given grasp.
    """
    B, n_f, _ = grasp_points.shape

    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-5)

    R = rot_from_vec(normals.reshape(-1, 3)).reshape(B, n_f, 3, 3)
    p_cross = skew(grasp_points.reshape(-1, 3)).reshape(B, n_f, 3, 3)

    grasp_mats = torch.cat([R, p_cross @ R], dim=-2)
    return torch.cat([grasp_mats[:, ii, :, :] for ii in range(n_f)], dim=-1)


def psv(grasp_points, normals):
    """
    Evaluates the minimum-singular-value grasp metric proposed in Li and Sastry '88.
    Args:
        grasp_points: a list of grasp points (torch.Tensors in 3D) at which to evaluate
             the grasp metric.
    Returns the product singular value of the grasp matrix formed by these points.
    """
    G = grasp_matrix(grasp_points, normals)
    return torch.prod(torch.linalg.svdvals(G), dim=-1)


def l1_metric(grasp_points_t, normals_t, mu=1.0, num_edges=10, grasp_mask=None):
    """L1 Grasp quality metric using PyFastGrasp. Assumes object center of mass is at origin"""
    import fastgrasp as fg

    B, _, _ = grasp_points_t.shape

    if grasp_mask is not None:
        valid_inds = torch.argwhere(grasp_mask.reshape(-1)).cpu().numpy().reshape(-1)
    else:
        valid_inds = torch.arange(B)

    device = normals_t.device
    grasp_points = grasp_points_t.detach().cpu().numpy().reshape(-1, 9)[valid_inds, :]
    normals = normals_t.detach().cpu().numpy().reshape(-1, 9)[valid_inds, :]
    centroid = np.zeros((len(grasp_points), 3), dtype="float64")
    grasps = np.concatenate([grasp_points, normals, centroid], axis=1)
    result = np.zeros(len(grasps))
    _ = fg.getLowerBoundsPurgeQHull(grasps, mu, num_edges, result)

    result_full = np.zeros(B)
    result_full[valid_inds] = result
    return torch.tensor(result_full, device=device)


def cos_similarity(a, b):
    """
    Returns the cosine similarity of two batches of vectors, a and b.
    """
    return torch.sum(a * b, dim=-1) / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1))


def get_cost_function(cost_fn="psv", return_details=False):
    """Factory for grasp cost function; generates grasp cost for CEM using config/model."""

    centroid = torch.from_numpy(mesh.centroid.copy())

    def cost_function(grasp_vars):
        # Reshape grasp vars into something useful, get dims.
        n_f = 4
        gps = grasp_vars.reshape(-1, n_f, 6)
        B = gps.shape[0]

        # If mesh is a triangle mesh, extract points/normals from it.
        grasp_points, grad_ests, grasp_mask = get_grasp_points(
            mesh, gps, residual_dirs=False
        )
        grasp_mask = grasp_mask.all(-1, keepdim=True)

        risk_sensitivity = None
        num_grasp_samples = 1

        # check_grasp_point_collapse(grasp_points, gps)
        # Reshape grasp points and grads for cost evaluation.
        grasp_points = grasp_points.reshape(-1, n_f, 3)  # [B * num_grasps, n_f, 3]
        grad_ests = grad_ests.reshape(-1, n_f, 3)  # [B * num_grasps, n_f, 3]

        # Center grasp_points around centroid.
        grasp_points_centered = grasp_points.to(centroid) - centroid.reshape(1, 1, 3)

        # Switch-case for cost function.
        if cost_fn == "psv":
            grasp_metric = partial(psv)
        # elif exp_config.cost_function == config.CostType.MSV:
        #     grasp_metric = partial(msv)
        # elif exp_config.cost_function == config.CostType.POLY_AREA:
        #     grasp_metric = poly_area
        # elif exp_config.cost_function == config.CostType.FC:
        #     grasp_metric = ferrari_canny
        elif cost_fn == "l1":
            grasp_metric = partial(
                l1_metric, grasp_mask=grasp_mask.expand(B, num_grasp_samples)
            )
        else:
            raise ValueError(f"Unknown cost function {cost_fn}")

        raw_cost = grasp_metric(grasp_points_centered, grad_ests).reshape(
            B, num_grasp_samples
        )

        # Add cost term penalizing cosine distance between approach angle and normal.
        rays_o, rays_d = gps[..., :3], gps[..., 3:]

        RESIDUAL_DIRS = False
        if RESIDUAL_DIRS:
            raise ValueError("Not implemented")
            # rays_d = grasp_utils.res_to_true_dirs(
            #     rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), centroid
            # ).reshape(
            #     -1, n_f, 3
            # )  # [B, n_f, 3]

        rays_d = (
            rays_d.unsqueeze(1)
            .expand(-1, num_grasp_samples, -1, -1)
            .reshape(-1, n_f, 3)
        )
        raw_cost -= 5e-2 * (
            1 - torch.mean(cos_similarity(rays_d, grad_ests), dim=-1)
        ).reshape(B, num_grasp_samples)

        # Exponentiate cost if using risk sensitivity.
        if risk_sensitivity:
            g_cost = torch.exp(-risk_sensitivity * raw_cost)
        else:
            g_cost = -raw_cost

        # Take expectation along sample dim.
        g_cost = g_cost.mean(-1)  # shape (B,)

        # Set invalid grasp costs to an upper bound
        INVALID_GRASP_COST = 2.0
        g_cost = torch.where(
            torch.all(grasp_mask, dim=-1),
            g_cost,
            INVALID_GRASP_COST * torch.ones_like(g_cost),
        )

        if risk_sensitivity:
            g_cost = (1 / risk_sensitivity) * torch.log(g_cost)

        if return_details:
            return g_cost, raw_cost, grasp_points, grad_ests, grasp_mask
        return g_cost

    return cost_function


def correct_z_dists(mesh, rays_o, rays_d, des_z_dist):
    if isinstance(rays_o, torch.Tensor):
        rays_o_np, rays_d_np = (
            rays_o.detach().cpu().numpy(),
            rays_d.detach().cpu().numpy(),
        )
    else:
        rays_o_np, rays_d_np = rays_o, rays_d

    rays_o_np, rays_d_np = rays_o_np.reshape(-1, 3), rays_d_np.reshape(-1, 3)

    # Put rays into mesh frame.
    # rays_o_np = rays_o_np - mesh.ig_centroid.reshape(1, 3)

    hit_points, ray_ids, face_ids = mesh.ray.intersects_location(
        rays_o_np, rays_d_np, multiple_hits=False
    )

    print("mesh extents: ", mesh.extents)
    print("hit points: ", hit_points)

    dists = np.linalg.norm(rays_o_np - hit_points, axis=-1)
    rays_o_corrected = (
        rays_o_np + (dists - des_z_dist).reshape(N_FINGERS, 1) * rays_d_np
    )

    print("raw distances: ", dists)

    # Put back into ig frame.
    # rays_o_corrected = rays_o_corrected + mesh.ig_centroid.reshape(1, 3)

    rays_o_corrected[:, 1] = np.maximum(rays_o_corrected[:, 1], object_bounds[1][0])
    dists_corrected = np.linalg.norm(
        rays_o_corrected - hit_points, axis=-1, keepdims=True
    )

    print("correct_distances: ", dists_corrected)

    if isinstance(rays_o, torch.Tensor):
        rays_o_corrected = torch.from_numpy(rays_o_corrected).to(rays_o)
    return rays_o_corrected


def compute_sampled_grasps(model, grasp_points, centroid):
    """Converts grasp vars to ray origins/directions; attempts to clip
    grasp to lie above floor + be equidistant from the surface."""

    print("optimized vals: ", grasp_points)
    if isinstance(model, trimesh.Trimesh):
        rays_o, rays_d = grasp_points[:, :3], grasp_points[:, 3:]
        # rays_d = grasp_utils.res_to_true_dirs(rays_o, rays_d, centroid)
        rays_o = correct_z_dists(model, rays_o, rays_d, 0.025)
    else:
        raise ValueError("Not implemented")
    print("corrected vals:", rays_o, rays_d, centroid)
    # rays_o, rays_d = grasp_utils.nerf_to_ig(rays_o), grasp_utils.nerf_to_ig(rays_d)
    return rays_o, rays_d


cem_num_iters = 15
cem_num_samples = 500
cem_elite_frac = 0.1

# %%
sampled_grasps = torch.zeros((N_GRASPS, N_FINGERS, N_XYZ * 2))
object_bounds = torch.tensor(mesh.bounds).T * 1.5
assert_equals(object_bounds.shape, (N_XYZ, 2))
projection_fn = partial(box_projection, object_bounds=object_bounds)
cost_fn = get_cost_function()
centroid = torch.from_numpy(mesh.centroid.copy())
assert_equals(centroid.shape, (N_XYZ,))

for ii in range(N_GRASPS):
    mu_f, Sigma_f, cost_history, best_point = optimize_cem(
        cost_fn,
        mu_0,
        Sigma_0,
        num_iters=cem_num_iters,
        num_samples=cem_num_samples,
        elite_frac=cem_elite_frac,
        projection=projection_fn,
    )

    if best_point is None:
        raise ValueError("CEM failed to find a feasible grasp")
    grasp_points = best_point.reshape(N_FINGERS, N_XYZ * 2)
    rays_o, rays_d = compute_sampled_grasps(mesh, grasp_points, centroid)

    sampled_grasps[ii, :, :3] = rays_o
    sampled_grasps[ii, :, 3:] = rays_d

# %%
fig = plot_grasps(mesh, sampled_grasps.detach(), idx=0)
fig.show()


# %%
costs = cost_fn(sampled_grasps)
print(f"costs: {costs}")
print(f"sampled_grasps.shape {sampled_grasps.shape}")

# %%
detailed_cost_fn = get_cost_function(cost_fn="psv", return_details=True)

# %%
g_cost, raw_cost, grasp_points, grad_ests, grasp_mask = detailed_cost_fn(sampled_grasps)
idx = 0
print(f"g_cost[idx]: {g_cost[idx]}")
print(f"raw_cost[idx]: {raw_cost[idx]}")
print(f"grasp_points[idx]: {grasp_points[idx]}")
print(f"grad_ests[idx]: {grad_ests[idx]}")
print(f"grasp_mask[idx]: {grasp_mask[idx]}")


# %%
breakpoint()

# %%
grasp_points[0]

# %%
distances = trimesh.proximity.signed_distance(mesh, grasp_points[0])
distances

# %%
