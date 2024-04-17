# %%
import tyro
from nerf_grasping.config.classifier_config import ClassifierConfig
import numpy as np
import trimesh
import json
import pathlib
import copy
import subprocess
from nerf_grasping.grasp_utils import load_nerf_field, get_nerf_configs, get_ray_samples, get_ray_origins_finger_frame
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import trimesh
from nerf_grasping.nerf_utils import (
    compute_centroid_from_nerf,
    get_ray_samples_in_region,
)
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    plot_mesh_and_query_points,
    plot_mesh_and_transforms,
)

from nerf_grasping.optimizer_utils import AllegroGraspConfig

from nerf_grasping.optimizer_utils import (
    GraspMetric,
    load_classifier,
)

NUM_FINGERS = 4

# %%
experiment_folder = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG")
object_name = "sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846"
assert experiment_folder.exists()

# %%
nerfdata_folder = experiment_folder / "nerfdata" / object_name
nerf_to_mesh_obj = experiment_folder / "nerf_to_mesh" / f"{object_name}.obj"
nerfcheckpoint_folder = experiment_folder / "nerfcheckpoints" / object_name
optimized_grasps_file = experiment_folder / "optimized_grasp_config_dicts" / f"{object_name}.npy"
assert nerfdata_folder.exists()
assert nerf_to_mesh_obj.exists()
assert nerfcheckpoint_folder.exists()
assert optimized_grasps_file.exists()

# %%
nerf_config = get_nerf_configs(str(nerfcheckpoint_folder))[-1]

# %%
nerf_field = load_nerf_field(nerf_config)

# %%
nerfdata_images = sorted(list((nerfdata_folder / "images").iterdir()))
assert len(nerfdata_images) > 0
fig, axs = plt.subplots(1, 2)
axs = axs.flatten()
axs[0].imshow(plt.imread(nerfdata_images[0]))
axs[1].imshow(plt.imread(nerfdata_images[-1]))

# %%
def transform_point(transform_matrix, point):
    point = np.append(point, 1)
    return np.dot(transform_matrix, point)[:3]


def add_transform_matrix_traces(fig, transform_matrix, length=0.1):
    origin = np.array([0, 0, 0])
    x_axis = np.array([length, 0, 0])
    y_axis = np.array([0, length, 0])
    z_axis = np.array([0, 0, length])

    origin_transformed = transform_point(transform_matrix, origin)
    x_axis_transformed = transform_point(transform_matrix, x_axis)
    y_axis_transformed = transform_point(transform_matrix, y_axis)
    z_axis_transformed = transform_point(transform_matrix, z_axis)

    for axis, color, name in zip([x_axis_transformed, y_axis_transformed, z_axis_transformed], ["red", "green", "blue"], ["x", "y", "z"]):
        fig.add_trace(
            go.Scatter3d(
                x=[origin_transformed[0], axis[0]],
                y=[origin_transformed[1], axis[1]],
                z=[origin_transformed[2], axis[2]],
                mode="lines",
                line=dict(color=color, width=5),
                name=name,
            )
        )

# %%
mesh = trimesh.load_mesh(nerf_to_mesh_obj)
mesh_centroid = mesh.centroid
print(f"Mesh centroid: {mesh_centroid}")

# %%
nerf_centroid = compute_centroid_from_nerf(
    nerf_field,
    lb=np.array([-0.25, -0.25, -0.25]),
    ub=np.array([0.25, 0.25, 0.25]),
    level=15,
    num_pts_x=100,
    num_pts_y=100,
    num_pts_z=100,
)
print(f"Nerf centroid: {nerf_centroid}")

# %%
fig = go.Figure()
fig.add_trace(
    go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color="lightblue",
        opacity=0.5,
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[mesh_centroid[0]],
        y=[mesh_centroid[1]],
        z=[mesh_centroid[2]],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="mesh_centroid",
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[nerf_centroid[0]],
        y=[nerf_centroid[1]],
        z=[nerf_centroid[2]],
        mode="markers",
        marker=dict(size=10, color="blue"),
        name="nerf_centroid",
    )
)
add_transform_matrix_traces(fig, np.eye(4))
fig.update_layout(title="N frame")
fig.show()


# %%
x_min, y_min, z_min = -0.25, -0.25, -0.25
x_max, y_max, z_max = 0.25, 0.25, 0.25
num_pts_x, num_pts_y, num_pts_z = 100, 100, 100
ray_samples = get_ray_samples_in_region(
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
    z_min=z_min,
    z_max=z_max,
    num_pts_x=num_pts_x,
    num_pts_y=num_pts_y,
    num_pts_z=num_pts_z,
)
query_points = np.copy(
    ray_samples.frustums.get_positions()
    .cpu()
    .numpy()
    .reshape(
        num_pts_x,
        num_pts_y,
        num_pts_z,
        3,
    )
)
nerf_densities = (
    nerf_field.get_density(ray_samples.to("cuda"))[0]
    .detach()
    .cpu()
    .numpy()
    .reshape(
        num_pts_x,
        num_pts_y,
        num_pts_z,
    )
)

# %%
query_points = query_points.reshape(-1, 3)
nerf_densities = nerf_densities.reshape(-1)

query_points_colors = np.copy(nerf_densities)

# %%
fig = go.Figure()
num_pts = query_points.shape[0]
assert query_points.shape == (num_pts, 3), f"{query_points.shape}"
assert query_points_colors.shape == (num_pts,), f"{query_points_colors.shape}"

# Filter
query_points = query_points[query_points_colors > 15]
query_points_colors = query_points_colors[query_points_colors > 15]

query_point_plot = go.Scatter3d(
    x=query_points[:, 0],
    y=query_points[:, 1],
    z=query_points[:, 2],
    mode="markers",
    marker=dict(
        size=1,
        color=query_points_colors,
        colorscale="viridis",
        colorbar=dict(title="Density Scale"),
    ),
    name="Query Point Densities",
)
fig.add_trace(query_point_plot)
fig.add_trace(
    go.Scatter3d(
        x=[nerf_centroid[0]],
        y=[nerf_centroid[1]],
        z=[nerf_centroid[2]],
        mode="markers",
        marker=dict(size=10, color="blue"),
        name="nerf_centroid",
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[mesh_centroid[0]],
        y=[mesh_centroid[1]],
        z=[mesh_centroid[2]],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="mesh_centroid",
    )
)

fig.update_layout(
    legend_orientation="h",
)  # Avoid overlapping legend
fig.update_layout(scene_aspectmode="data")
fig.update_layout(title="N frame")
add_transform_matrix_traces(fig, np.eye(4))
fig.show()

# %%
X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
X_N_O = trimesh.transformations.translation_matrix(nerf_centroid)  # TODO: Check this
X_N_Oy = X_N_O @ X_O_Oy
X_Oy_N = np.linalg.inv(X_N_Oy)

# %%
mesh_Oy = trimesh.load_mesh(nerf_to_mesh_obj)
mesh_Oy.apply_transform(X_Oy_N)

# %%
fig = go.Figure()
fig.add_trace(
    go.Mesh3d(
        x=mesh_Oy.vertices[:, 0],
        y=mesh_Oy.vertices[:, 1],
        z=mesh_Oy.vertices[:, 2],
        i=mesh_Oy.faces[:, 0],
        j=mesh_Oy.faces[:, 1],
        k=mesh_Oy.faces[:, 2],
        color="lightblue",
        opacity=0.5,
    )
)
add_transform_matrix_traces(fig, np.eye(4))
fig.update_layout(title="Oy frame")
fig.show()

# %%
evaled_grasp_config_dict = np.load(optimized_grasps_file, allow_pickle=True).item()
grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
    evaled_grasp_config_dict
)
grasp_frame_transforms = grasp_configs.grasp_frame_transforms
loss = evaled_grasp_config_dict["loss"]
print(f"Loss: {loss}")

# %%
grasp_frame_transforms.lshape

# %%
grasp_frame_transform_matrices = grasp_frame_transforms.matrix()

# %%
grasp_frame_transform_matrices.shape, grasp_frame_transform_matrices.device

# %%
GRASP_IDX = 0
grasp_frame_transform_matrices_np = grasp_frame_transform_matrices.detach().cpu().numpy()[GRASP_IDX]

# %%
grasp_frame_transform_matrices_np.shape


# %%
fig = plot_mesh_and_transforms(
    mesh=mesh_Oy,
    transforms=[grasp_frame_transform_matrices_np[0], grasp_frame_transform_matrices_np[1], grasp_frame_transform_matrices_np[2], grasp_frame_transform_matrices_np[3]],
    num_fingers=NUM_FINGERS,
    title="Mesh and Transforms",
    highlight_idx=0,
)
fig.update_layout(title="Oy frame")
add_transform_matrix_traces(fig, np.eye(4))
fig.show()

# %%
classifier_config_path = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/Train_DexGraspNet_NeRF_Grasp_Metric_workspaces/1700rotated_augmented_pose_HALTON_50_cnn-3d-xyz_l2_all_2024-04-10_10-14-17-971253/config.yaml")
classifier_config = tyro.extras.from_yaml(
    ClassifierConfig, classifier_config_path.open()
)
classifier_model = load_classifier(classifier_config=classifier_config)
grasp_metric = GraspMetric(
    nerf_field=nerf_field,
    classifier_model=classifier_model,
    fingertip_config=classifier_config.nerfdata_config.fingertip_config,
    X_N_Oy=X_N_Oy,
)

# %%
ray_samples = grasp_metric.compute_ray_samples(grasp_configs)
query_points = ray_samples.frustums.get_positions()
densities = grasp_metric.compute_nerf_densities(
    ray_samples,
)

# %%
query_points.shape, densities.shape

# %%
num_pts_x, num_pts_y, num_pts_z = densities.shape[-3:]
query_points_np = query_points[GRASP_IDX].reshape(NUM_FINGERS, -1, 3).detach().cpu().numpy()
densities_np = densities[GRASP_IDX].reshape(NUM_FINGERS, -1).detach().cpu().numpy()

# %%
query_points_np.shape, densities_np.shape

# %%
fig = plot_mesh_and_query_points(
    mesh=mesh,
    query_points_list=[query_points_np[0], query_points_np[1], query_points_np[2], query_points_np[3]],
    query_points_colors_list=[densities_np[0], densities_np[1], densities_np[2], densities_np[3]],
    num_fingers=NUM_FINGERS,
    title="Mesh and Query Points",
)
add_transform_matrix_traces(fig, np.eye(4))
fig.update_layout(title="N frame")
fig.show()

# %%
def transform_points(transform_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    assert points.shape[-1] == 3
    assert transform_matrix.shape == (4, 4)
    points_H = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    transformed_points_H = np.einsum("ij,nj->ni", transform_matrix, points_H)
    transformed_points = transformed_points_H[..., :3]
    return transformed_points

fig = plot_mesh_and_query_points(
    mesh=mesh_Oy,
    query_points_list=[transform_points( X_Oy_N, query_points_np[0]),
                          transform_points( X_Oy_N, query_points_np[1]),
                          transform_points( X_Oy_N, query_points_np[2]),
                          transform_points( X_Oy_N, query_points_np[3])],
    query_points_colors_list=[densities_np[0], densities_np[1], densities_np[2], densities_np[3]],
    num_fingers=NUM_FINGERS,
    title="Mesh and Query Points",
)
add_transform_matrix_traces(fig, np.eye(4))
fig.update_layout(title="Oy frame")
fig.show()

# %%

delta = (
    classifier_config.nerfdata_config.fingertip_config.grasp_depth_mm / 1000 / (classifier_config.nerfdata_config.fingertip_config.num_pts_z - 1)
)
nrows, ncols = NUM_FINGERS, 1
fig4, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

axes = axes.flatten()
for i in range(nrows):
    ax = axes[i]
    finger_alphas = 1 - np.exp(-delta * densities_np[i].reshape(
        classifier_config.nerfdata_config.fingertip_config.num_pts_x,
        classifier_config.nerfdata_config.fingertip_config.num_pts_y,
        classifier_config.nerfdata_config.fingertip_config.num_pts_z,
    ))
    finger_alphas_maxes = np.max(finger_alphas, axis=(0, 1))
    finger_alphas_means = np.mean(finger_alphas, axis=(0, 1))
    ax.plot(finger_alphas_maxes, label="max")
    ax.plot(finger_alphas_means, label="mean")
    ax.legend()
    ax.set_xlabel("z")
    ax.set_ylabel("alpha")
    ax.set_title(f"finger {i}")
    ax.set_ylim([0, 1])
fig4.tight_layout()
fig4.show()

#%%
num_images = 5
nrows, ncols = NUM_FINGERS, num_images
fig5, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
alphas_np = 1 - np.exp(-delta * densities_np)
alpha_min, alpha_max = alphas_np.min(), alphas_np.max()

for finger_i in range(NUM_FINGERS):
    for image_i in range(num_images):
        ax = axes[finger_i, image_i]

        image = alphas_np[finger_i].reshape(
            classifier_config.nerfdata_config.fingertip_config.num_pts_x,
            classifier_config.nerfdata_config.fingertip_config.num_pts_y,
            classifier_config.nerfdata_config.fingertip_config.num_pts_z,
        )[:, :, int(image_i * num_pts_z / num_images)]
        ax.imshow(
            image,
            vmin=alpha_min,
            vmax=alpha_max,
        )
        ax.set_title(f"finger {finger_i}, image {image_i}")
fig5.tight_layout()
fig5.show()


# %%
