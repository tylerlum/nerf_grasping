import os
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import trimesh
import pathlib
import pypose as pp
import plotly.graph_objects as go
import nerf_grasping
import nerfstudio
from matplotlib import pyplot as plt
from nerf_grasping.grasp_utils import (
    get_ray_samples_helper,
    get_ray_origins_finger_frame_helper,
)
from nerfstudio.cameras.rays import RayBundle, RaySamples


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

def parse_object_code_and_scale(object_code_and_scale_str: str) -> Tuple[str, float]:
    # Input: sem-Gun-4745991e7c0c7966a93f1ea6ebdeec6f_0_10
    # Output: sem-Gun-4745991e7c0c7966a93f1ea6ebdeec6f, 0.10
    keyword = "_0_"
    idx = object_code_and_scale_str.rfind(keyword)
    object_code = object_code_and_scale_str[:idx]

    idx_offset_for_scale = keyword.index("0")
    object_scale = float(
        object_code_and_scale_str[idx + idx_offset_for_scale :].replace("_", ".")
    )
    return object_code, object_scale



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


def plot_mesh_and_transforms(
    mesh: trimesh.Trimesh,
    transforms: List[np.ndarray],
    num_fingers: int,
    title: str = "Mesh and Transforms",
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

    fig.update_layout(legend_orientation="h", title_text=title)
    return fig


def plot_mesh_and_query_points(
    mesh: trimesh.Trimesh,
    query_points_list: List[np.ndarray],
    query_points_colors_list: List[np.ndarray],
    num_fingers: int,
    title: str = "Query Points",
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
                color=query_points_colors.squeeze(-1),
                colorscale="viridis",
                colorbar=dict(title="Density Scale") if finger_idx == 0 else {},
            ),
            name=f"Query Point Densities Finger {finger_idx}",
        )
        fig.add_trace(query_point_plot)

    fig.update_layout(
        legend_orientation="h",
        scene=dict(
            xaxis=dict(nticks=4, range=[-0.2, 0.2]),
            yaxis=dict(nticks=4, range=[-0.2, 0.2]),
            zaxis=dict(nticks=4, range=[-0.2, 0.2]),
        ),
        title_text=title,
    )  # Avoid overlapping legend
    fig.update_layout(scene_aspectmode="cube")
    return fig


def plot_mesh_and_high_density_points(
    mesh: trimesh.Trimesh,
    query_points: np.ndarray,
    query_points_colors: np.ndarray,
    density_threshold: float,
) -> go.Figure:
    n_pts = query_points.shape[0]
    assert query_points.shape == (n_pts, 3), f"{query_points.shape}"
    assert query_points_colors.shape == (n_pts,), f"{query_points_colors.shape}"

    fig = plot_mesh(mesh)

    # Filter
    query_points = query_points[query_points_colors > density_threshold]
    query_points_colors = query_points_colors[query_points_colors > density_threshold]

    query_point_plot = go.Scatter3d(
        x=query_points[:, 0],
        y=query_points[:, 1],
        z=query_points[:, 2],
        mode="markers",
        marker=dict(
            size=4,
            color=query_points_colors,
            colorscale="viridis",
            colorbar=dict(title="Density Scale"),
        ),
        name="Query Point Densities",
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


def get_ray_samples_in_mesh_region(
    mesh: trimesh.Trimesh, num_pts_x: int, num_pts_y: int, num_pts_z: int
) -> RaySamples:
    # Get bounds from mesh
    min_bounds, max_bounds = mesh.bounds
    x_min, y_min, z_min = min_bounds
    x_max, y_max, z_max = max_bounds
    finger_width_mm = (x_max - x_min) * 1000
    finger_height_mm = (y_max - y_min) * 1000
    grasp_depth_mm = (z_max - z_min) * 1000

    # Prepare ray samples
    ray_origins_object_frame = get_ray_origins_finger_frame_helper(
        num_pts_x=num_pts_x,
        num_pts_y=num_pts_y,
        finger_width_mm=finger_width_mm,
        finger_height_mm=finger_height_mm,
        z_offset_mm=z_min * 1000,
    )
    assert ray_origins_object_frame.shape == (num_pts_x, num_pts_y, 3)
    identity_transform = pp.identity_SE3(1).to(
        ray_origins_object_frame.device
    )
    ray_samples = get_ray_samples_helper(
        ray_origins_object_frame,
        identity_transform,
        num_pts_z=num_pts_z,
        grasp_depth_mm=grasp_depth_mm,
    )
    return ray_samples
