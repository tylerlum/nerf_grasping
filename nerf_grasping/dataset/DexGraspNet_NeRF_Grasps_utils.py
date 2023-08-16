import os
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import trimesh
import pathlib
import plotly.graph_objects as go
import nerf_grasping
import nerfstudio
from nerfstudio.utils import eval_utils
from nerfstudio.cameras.rays import RayBundle, RaySamples
from matplotlib import pyplot as plt


def get_object_string(cfg_path: pathlib.Path) -> str:
    assert "_0_" in str(cfg_path), f"_0_ not in {cfg_path}"
    return [ss for ss in cfg_path.parts if "_0_" in ss][0]


def get_object_scale(cfg_path: pathlib.Path) -> float:
    # BRITTLE
    # Assumes "_0_" only shows up once at the end
    # eg. sem-VideoGameConsole-49ba4f628a955bf03742135a31826a22_0_06
    obj_str = get_object_string(cfg_path)
    idx = obj_str.index("_0_")
    return float(obj_str[idx + 1 :].replace("_", "."))


def get_object_code(cfg_path: pathlib.Path) -> str:
    # BRITTLE
    # Assumes "_0_" only shows up once at the end
    # eg. sem-VideoGameConsole-49ba4f628a955bf03742135a31826a22_0_06
    obj_str = get_object_string(cfg_path)
    idx = obj_str.index("_0_")
    object_code = obj_str[:idx]
    return object_code


def get_nerf_configs(nerf_checkpoints_path: str) -> List[str]:
    return list(pathlib.Path(nerf_checkpoints_path).rglob("config.yml"))


def get_contact_candidates_and_target_candidates(
    grasp_data: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    link_name_to_contact_candidates = grasp_data["link_name_to_contact_candidates"]
    link_name_to_target_contact_candidates = grasp_data[
        "link_name_to_target_contact_candidates"
    ]
    contact_candidates = np.concatenate(
        [
            contact_candidate
            for _, contact_candidate in link_name_to_contact_candidates.items()
        ],
        axis=0,
    )
    target_contact_candidates = np.concatenate(
        [
            target_contact_candidate
            for _, target_contact_candidate in link_name_to_target_contact_candidates.items()
        ],
        axis=0,
    )
    return contact_candidates, target_contact_candidates


def get_start_and_end_and_up_points(
    contact_candidates: np.ndarray,
    target_contact_candidates: np.ndarray,
    num_fingers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # BRITTLE: Assumes same number of contact points per finger
    # BRITTLE: Assumes UP_POINT_IDX is position of contact candidate up from center
    UP_POINT_IDX = 3
    contact_candidates_per_finger = contact_candidates.reshape(num_fingers, -1, 3)
    target_contact_candidates_per_finger = target_contact_candidates.reshape(
        num_fingers, -1, 3
    )
    start_points = contact_candidates_per_finger.mean(axis=1)
    end_points = target_contact_candidates_per_finger.mean(axis=1)
    up_points = contact_candidates_per_finger[:, UP_POINT_IDX, :]
    assert start_points.shape == end_points.shape == up_points.shape == (num_fingers, 3)
    return np.array(start_points), np.array(end_points), np.array(up_points)


def get_scene_dict() -> Dict[str, Any]:
    return dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(title="Z"),
        aspectmode="data",
    )


def plot_mesh(mesh: trimesh.Trimesh, color="lightpink") -> go.Figure:
    vertices = mesh.vertices
    faces = mesh.faces

    # Create the mesh3d trace
    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=0.5,
        name="Mesh",
    )

    # Create the layout
    layout = go.Layout(
        scene=get_scene_dict(),
        showlegend=True,
        title="Mesh",
    )

    # Create the figure
    fig = go.Figure(data=[mesh_plot], layout=layout)

    # Return the figure
    return fig


def load_nerf(cfg_path: pathlib.Path) -> nerfstudio.models.base_model.Model:
    _, pipeline, _, _ = eval_utils.eval_setup(cfg_path, test_mode="inference")
    return pipeline.model.field


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def get_transform(start: np.ndarray, end: np.ndarray, up: np.ndarray) -> np.ndarray:
    # BRITTLE: Assumes new_z and new_y are pretty much perpendicular
    # If not, tries to find closest possible
    new_z = normalize(end - start)
    # new_y should be perpendicular to new_z
    up_dir = normalize(up - start)
    new_y = normalize(up_dir - np.dot(up_dir, new_z) * new_z)
    new_x = np.cross(new_y, new_z)

    transform = np.eye(4)
    transform[:3, :3] = np.stack([new_x, new_y, new_z], axis=1)
    transform[:3, 3] = start
    return transform


def plot_mesh_and_transforms(
    mesh: trimesh.Trimesh, transforms: List[np.ndarray], num_fingers: int
) -> go.Figure:
    assert len(transforms) == num_fingers, f"{len(transforms)} != {num_fingers}"

    # Add the scatter plot to a figure and display it
    fig = plot_mesh(mesh)
    for finger_idx in range(num_fingers):
        transform = transforms[finger_idx]
        length = 0.02
        origin = np.array([0, 0, 0])
        x_axis = np.array([length, 0, 0])
        y_axis = np.array([0, length, 0])
        z_axis = np.array([0, 0, length])

        new_origin = transform @ np.concatenate([origin, [1]])
        new_x_axis = transform @ np.concatenate([x_axis, [1]])
        new_y_axis = transform @ np.concatenate([y_axis, [1]])
        new_z_axis = transform @ np.concatenate([z_axis, [1]])
        x_plot = go.Scatter3d(
            x=[new_origin[0], new_x_axis[0]],
            y=[new_origin[1], new_x_axis[1]],
            z=[new_origin[2], new_x_axis[2]],
            mode="lines",
            marker=dict(
                size=8,
                color="red",
                colorscale="viridis",
            ),
            name=f"Finger {finger_idx} X Axis",
        )
        y_plot = go.Scatter3d(
            x=[new_origin[0], new_y_axis[0]],
            y=[new_origin[1], new_y_axis[1]],
            z=[new_origin[2], new_y_axis[2]],
            mode="lines",
            marker=dict(
                size=8,
                color="green",
                colorscale="viridis",
            ),
            name=f"Finger {finger_idx} Y Axis",
        )
        z_plot = go.Scatter3d(
            x=[new_origin[0], new_z_axis[0]],
            y=[new_origin[1], new_z_axis[1]],
            z=[new_origin[2], new_z_axis[2]],
            mode="lines",
            marker=dict(
                size=8,
                color="blue",
                colorscale="viridis",
            ),
            name=f"Finger {finger_idx} Z Axis",
        )

        fig.add_trace(x_plot)
        fig.add_trace(y_plot)
        fig.add_trace(z_plot)

    fig.update_layout(legend_orientation="h")
    return fig


# Grid of points in grasp frame (x, y, z)
GRASP_DEPTH_MM = 20
FINGER_WIDTH_MM = 10
FINGER_HEIGHT_MM = 15

# Want points equally spread out in space
DIST_BTWN_PTS_MM = 0.5

# +1 to include both end points
NUM_PTS_X = int(FINGER_WIDTH_MM / DIST_BTWN_PTS_MM) + 1
NUM_PTS_Y = int(FINGER_HEIGHT_MM / DIST_BTWN_PTS_MM) + 1
NUM_PTS_Z = int(GRASP_DEPTH_MM / DIST_BTWN_PTS_MM) + 1

NUM_FINGERS = 4


def get_ray_origins_finger_frame_helper(
    num_pts_x: int,
    num_pts_y: int,
    grasp_depth_mm: float,
    finger_width_mm: float,
    finger_height_mm: float,
) -> np.ndarray:
    grasp_depth_m = grasp_depth_mm / 1000.0
    gripper_finger_width_m = finger_width_mm / 1000.0
    gripper_finger_height_m = finger_height_mm / 1000.0

    # Create grid of grasp origins in finger frame with shape (num_pts_x, num_pts_y, 3)
    # So that grid_of_points[2, 3] = [x, y, z], where x, y, z are the coordinates of the '
    # ray origin for the [2, 3] "pixel" in the finger frame.
    # Origin of transform is at center of xy at 1/4 of the way into the depth z
    # x is width, y is height, z is depth
    x_coords = np.linspace(
        -gripper_finger_width_m / 2, gripper_finger_width_m / 2, num_pts_x
    )
    y_coords = np.linspace(
        -gripper_finger_height_m / 2, gripper_finger_height_m / 2, num_pts_y
    )
    # z_coords = np.linspace(-grasp_depth_m / 4, 3 * grasp_depth_m / 4, num_pts_z)

    xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
    zz = -grasp_depth_m / 4 * np.ones_like(xx)

    assert xx.shape == yy.shape == zz.shape == (num_pts_x, num_pts_y)
    ray_origins = np.stack([xx, yy, zz], axis=-1)

    assert ray_origins.shape == (num_pts_x, num_pts_y, 3)

    return ray_origins


def get_ray_origins_finger_frame() -> np.ndarray:
    ray_origins_finger_frame = get_ray_origins_finger_frame_helper(
        num_pts_x=NUM_PTS_X,
        num_pts_y=NUM_PTS_Y,
        grasp_depth_mm=GRASP_DEPTH_MM,
        finger_width_mm=FINGER_WIDTH_MM,
        finger_height_mm=FINGER_HEIGHT_MM,
    )
    return ray_origins_finger_frame


def get_transformed_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    assert points.shape == (n_points, 3), f"{points.shape}"
    assert transform.shape == (4, 4), f"{transform.shape}"

    extra_ones = np.ones((n_points, 1))
    points_homogeneous = np.concatenate([points, extra_ones], axis=1)

    # First (4, 4) @ (4, N) = (4, N)
    # Then transpose to get (N, 4)
    transformed_points = np.matmul(transform, points_homogeneous.T).T

    transformed_points = transformed_points[:, :3]
    assert transformed_points.shape == (n_points, 3), f"{transformed_points.shape}"
    return transformed_points


def get_transformed_dirs(dirs: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transforms direction vectors (i.e., doesn't apply the translation of a homologous tranform).
    """
    n_dirs = dirs.shape[0]
    assert dirs.shape == (n_dirs, 3), f"{dirs.shape}"
    assert transform.shape == (4, 4), f"{transform.shape}"

    transformed_dirs = np.matmul(transform[:3, :3], dirs.T).T  # only rotate directions.

    return transformed_dirs


def get_ray_samples(
    ray_origins_finger_frame: np.ndarray,
    transform: np.ndarray,
    num_pts_z: int = NUM_PTS_Z,
    grasp_depth_mm: float = float(GRASP_DEPTH_MM),
) -> RaySamples:
    grasp_depth_m = grasp_depth_mm / 1000.0

    num_pts_x, num_pts_y = ray_origins_finger_frame.shape[:2]

    assert ray_origins_finger_frame.shape == (num_pts_x, num_pts_y, 3)

    # Collapse batch dimensions + apply transform.
    ray_origins_world_frame = get_transformed_points(
        ray_origins_finger_frame.reshape(-1, 3), transform
    )
    ray_origins_world_frame = torch.tensor(ray_origins_world_frame).float().contiguous()

    ray_dirs_finger_frame = np.array([0.0, 0.0, 1.0]).reshape(
        1, 3
    )  # Ray dirs are along +z axis.
    ray_dirs_world_frame = get_transformed_dirs(ray_dirs_finger_frame, transform)

    # Cast to Tensor + expand to match origins shape.
    ray_dirs_world_frame = (
        torch.tensor(ray_dirs_world_frame)
        .expand(ray_origins_world_frame.shape)
        .float()
        .contiguous()
    )

    # Create dummy pixel areas object.
    pixel_area = (
        torch.ones_like(ray_dirs_world_frame[..., 0]).unsqueeze(-1).float().contiguous()
    )

    ray_bundle = RayBundle(ray_origins_world_frame, ray_dirs_world_frame, pixel_area)

    # Work out sample lengths.
    sample_dists = torch.linspace(0.0, grasp_depth_m, steps=num_pts_z)

    sample_dists = sample_dists.reshape(1, num_pts_z, 1).expand(
        ray_origins_world_frame.shape[0], -1, -1
    )

    # Pull ray samples -- note these are degenerate, i.e., the deltas field is meaningless.
    return ray_bundle.get_ray_samples(sample_dists, sample_dists)


def plot_mesh_and_query_points(
    mesh: trimesh.Trimesh,
    query_points_list: List[np.ndarray],
    query_points_colors_list: List[np.ndarray],
    num_fingers: int,
) -> go.Figure:
    assert (
        len(query_points_list) == len(query_points_colors_list) == num_fingers
    ), f"{len(query_points_list)} != {num_fingers}"
    fig = plot_mesh(mesh)

    for finger_idx in range(num_fingers):
        query_points = query_points_list[finger_idx]
        query_points_colors = query_points_colors_list[finger_idx]
        query_point_plot = go.Scatter3d(
            x=query_points[:, 0],
            y=query_points[:, 1],
            z=query_points[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=query_points_colors,
                colorscale="viridis",
                colorbar=dict(title="Density Scale") if finger_idx == 0 else {},
            ),
            name=f"Query Point Densities Finger {finger_idx}",
        )
        fig.add_trace(query_point_plot)

    fig.update_layout(
        legend_orientation="h",
        scene=dict(
            xaxis=dict(nticks=4, range=[-0.1, 0.1]),
            yaxis=dict(nticks=4, range=[-0.1, 0.1]),
            zaxis=dict(nticks=4, range=[-0.1, 0.1]),
        ),
    )  # Avoid overlapping legend
    fig.update_layout(scene_aspectmode="cube")
    return fig


def get_bbox_ray_samples(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    num_points_x: int = 5 * NUM_PTS_X,
    num_points_y: int = 5 * NUM_PTS_Y,
    num_points_z: int = 5 * NUM_PTS_Z,
):
    """
    Creates a nerfstudio RaySamples object for querying the NeRF on a
    grid inside an axis-aligned bounding box.
    """

    x_points = np.linspace(bbox_min[0], bbox_max[0], num_points_x)
    y_points = np.linspace(bbox_min[1], bbox_max[1], num_points_y)

    xx, yy = np.meshgrid(x_points, y_points, indexing="ij")
    zz = bbox_min[-1] * np.ones_like(xx)

    ray_origins = np.stack([xx, yy, zz], axis=-1)

    transform = torch.eye(4)

    return get_ray_samples(
        ray_origins, transform, num_points_z, 1000 * (bbox_max[-1] - bbox_min[-1])
    )


def plot_nerf_densities(
    mesh,
    nerf_model,
    bbox_min=np.array([-0.1, -0.05, -0.05]),
    bbox_max=-np.array([-0.1, -0.05, -0.05]),
    nerf_thresh=5.0,
):
    ray_samples = get_bbox_ray_samples(bbox_min, bbox_max)
    densities = nerf_model.get_density(ray_samples.to("cuda"))[0].detach().cpu().numpy()
    query_points = ray_samples.frustums.get_positions().detach().cpu().numpy()

    valid_inds = np.where(densities.reshape(-1) > nerf_thresh)
    return plot_mesh_and_query_points(
        mesh,
        [query_points.reshape(-1, 3)[valid_inds]],
        [densities.reshape(-1)[valid_inds]],
        1,
    )
