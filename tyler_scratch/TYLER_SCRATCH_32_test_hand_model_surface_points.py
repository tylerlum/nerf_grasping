# %%
import torch
from nerf_grasping.optimizer_utils import (
    get_sorted_grasps_from_dict,
)
import numpy as np
import pathlib

import plotly.graph_objects as go

from nerf_grasping.grasp_utils import (
    load_nerf_field,
    get_nerf_configs,
    get_ray_samples,
    get_ray_origins_finger_frame,
)
from nerf_grasping.dexgraspnet_utils.hand_model import HandModel
from nerf_grasping.dexgraspnet_utils.hand_model_type import (
    HandModelType,
)
from nerf_grasping.dexgraspnet_utils.pose_conversion import (
    hand_config_to_pose,
)
import trimesh

from nerf_grasping.nerf_utils import (
    get_ray_samples_in_region,
    get_density,
)

# %%
optimized_grasp_config_dict_path = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG_17/optimized_grasp_config_dicts/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846.npy"
)
mesh_Oy_path = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG_17/nerf_to_mesh_Oy/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846.obj"
)
nerfcheckpoint_path = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG_17/nerfcheckpoints/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846"
)

experiment_folder = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG"
)
object_name = "sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846"
assert optimized_grasp_config_dict_path.exists()
assert mesh_Oy_path.exists()
assert nerfcheckpoint_path.exists()
GRASP_IDX = 3

# %%
optimized_grasp_config_dict = np.load(
    optimized_grasp_config_dict_path, allow_pickle=True
).item()
X_Oy_H_array, joint_angles_array, target_joint_angles_array = (
    get_sorted_grasps_from_dict(
        optimized_grasp_config_dict=optimized_grasp_config_dict,
        error_if_no_loss=True,
        check=False,
        print_best=False,
    )
)

# %%
mesh_Oy = trimesh.load_mesh(mesh_Oy_path)

# %%
nerf_config = get_nerf_configs(str(nerfcheckpoint_path))[-1]
nerf_field = load_nerf_field(nerf_config)

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
query_point_plot = go.Scatter3d(
    x=query_points[query_points_colors > 15, 0],
    y=query_points[query_points_colors > 15, 1],
    z=query_points[query_points_colors > 15, 2],
    mode="markers",
    marker=dict(
        size=1,
        color=query_points_colors[query_points_colors > 15],
        colorscale="viridis",
        colorbar=dict(title="Density Scale"),
    ),
    name="Query Point Densities",
)
fig.add_trace(query_point_plot)

fig.update_layout(
    legend_orientation="h",
)  # Avoid overlapping legend
fig.update_layout(scene_aspectmode="data")
fig.update_layout(title="N frame")
fig.show()


# %%
device = "cuda"
hand_model_type = HandModelType.ALLEGRO_HAND
hand_model = HandModel(hand_model_type=hand_model_type, device=device)

# %%
# Compute pregrasp and target hand poses
trans_array = X_Oy_H_array[:, :3, 3]
rot_array = X_Oy_H_array[:, :3, :3]

pregrasp_hand_pose = hand_config_to_pose(trans_array, rot_array, joint_angles_array).to(
    device
)
target_hand_pose = hand_config_to_pose(
    trans_array, rot_array, target_joint_angles_array
).to(device)

# %%
# Get plotly data
hand_model.set_parameters(pregrasp_hand_pose)
pregrasp_plot_data = hand_model.get_plotly_data(i=GRASP_IDX, opacity=1.0)

hand_model.set_parameters(target_hand_pose)
target_plot_data = hand_model.get_plotly_data(i=GRASP_IDX, opacity=0.5)

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
        color="lightpink",
        opacity=0.50,
    )
)
for x in pregrasp_plot_data:
    fig.add_trace(x)
for x in target_plot_data:
    fig.add_trace(x)
fig.show()


# %%
hand_model.get_surface_points()

# %%
hand_model_surface_pts = HandModel(
    hand_model_type=hand_model_type, device=device, n_surface_points=1000
)
hand_model_surface_pts.set_parameters(pregrasp_hand_pose)

pregrasp_plot_surface_ptsdata = hand_model.get_plotly_data(
    i=GRASP_IDX, opacity=1.0, with_surface_points=True
)

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
        color="lightpink",
        opacity=0.50,
    )
)
for x in pregrasp_plot_surface_ptsdata:
    fig.add_trace(x)
fig.show()

# %%
pts = hand_model_surface_pts.get_surface_points()

# %%
pts.shape

# %%
fig.add_trace(
    go.Scatter3d(
        x=pts[GRASP_IDX].cpu().numpy().reshape(-1, 3)[:, 0],
        y=pts[GRASP_IDX].cpu().numpy().reshape(-1, 3)[:, 1],
        z=pts[GRASP_IDX].cpu().numpy().reshape(-1, 3)[:, 2],
        mode="markers",
        marker=dict(size=2),
    )
)
fig.show()



# %%
TEMP = get_density(
    field=nerf_field,
    positions=torch.from_numpy(query_points).to("cuda"),
)[0]

# %%
TEMP.shape
# %%
nerf_densities_2 = get_density(
    field=nerf_field,
    positions=torch.from_numpy(query_points).to("cuda"),
)[0].detach().cpu().numpy().reshape(num_pts_x, num_pts_y, num_pts_z)
nerf_densities_2 = nerf_densities_2.reshape(-1)

# %%
assert np.allclose(nerf_densities, nerf_densities_2)


# %%
GRASP_IDX = 4

# %%
densities_at_surface_pts = get_density(
    field=nerf_field,
    positions=pts[GRASP_IDX].to("cuda"),
)[0].detach().cpu().numpy().reshape(-1)

# %%
sum_densities = densities_at_surface_pts.sum()
print(f"Sum of densities at surface points: {sum_densities}")

# %%
max_densities = densities_at_surface_pts.max()
print(f"Max density at surface points: {max_densities}")

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=pts[GRASP_IDX].cpu().numpy().reshape(-1, 3)[:, 0],
        y=pts[GRASP_IDX].cpu().numpy().reshape(-1, 3)[:, 1],
        z=pts[GRASP_IDX].cpu().numpy().reshape(-1, 3)[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=densities_at_surface_pts,
            colorscale="viridis",
            colorbar=dict(title="Density Scale"),
        ),
    )
)
fig.show()

# %%
fig = go.Figure()
# Histogram of densities
fig.add_trace(
    go.Histogram(
        x=densities_at_surface_pts,
        nbinsx=100,
        # histnorm="probability",
        name="Density at Surface Points",
    )
)
fig.show()

# %%
inside = mesh_Oy.contains(pts[GRASP_IDX].cpu().numpy().reshape(-1, 3))
print(f"Num points inside: {inside.sum()}")


# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=pts[GRASP_IDX].cpu().numpy().reshape(-1, 3)[:, 0],
        y=pts[GRASP_IDX].cpu().numpy().reshape(-1, 3)[:, 1],
        z=pts[GRASP_IDX].cpu().numpy().reshape(-1, 3)[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=inside * 100,
            colorscale="viridis",
            colorbar=dict(title="Density Scale"),
        ),
    )
)
fig.show()

# %%
fig = go.Figure()
# Histogram of densities
fig.add_trace(
    go.Histogram(
        x=inside,
        nbinsx=100,
        # histnorm="probability",
        name="Density at Surface Points",
    )
)
fig.show()

# %%
num_large_densities, max_densities, sum_densities, num_insides = [], [], [], []
for i in range(32):
    densities_at_surface_pts = get_density(
        field=nerf_field,
        positions=pts[i].to("cuda"),
    )[0].detach().cpu().numpy().reshape(-1)

    inside = mesh_Oy.contains(pts[i].cpu().numpy().reshape(-1, 3))
    num_large_densities.append((densities_at_surface_pts > 15).sum())
    max_densities.append(densities_at_surface_pts.max())
    sum_densities.append(densities_at_surface_pts.sum())
    num_insides.append(inside.sum())

num_large_densities = np.array(num_large_densities)
max_densities = np.array(max_densities)
sum_densities = np.array(sum_densities)
num_insides = np.array(num_insides)

# %%
no_penetration_idxs = [2, 4, 5, 6, 7, 10, 12, 13, 16, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29]
penetrations = np.array([0 if i in no_penetration_idxs else 1 for i in range(32)])

# %%
print(sum(penetrations))

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=penetrations + np.random.normal(0, 0.1, 32),
        y=np.log(max_densities),
        mode="markers",
        name="Max Density",
    )
)
fig.show()

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=penetrations + np.random.normal(0, 0.1, 32),
        y=np.log(sum_densities),
        mode="markers",
        name="Sum Density",
    )
)
fig.show()


# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=penetrations + np.random.normal(0, 0.1, 32),
        y=np.log(num_insides + 1),
        mode="markers",
        name="Num Inside",
    )
)
fig.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Example data
y = penetrations
x = num_insides

# Generate ROC curve data
fpr, tpr, thresholds = roc_curve(y, x)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Find the optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold}")


# %%
plt.figure()
for x, name in zip([max_densities, sum_densities, num_insides, num_large_densities], ["Max Density", "Sum Density", "Num Inside", "Num Large Densities"]):
    fpr, tpr, thresholds = roc_curve(penetrations, x)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (area = {roc_auc:.2f})')
    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold for {name}: {optimal_threshold}")

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %%
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# %%
X = np.array([max_densities, num_large_densities]).T
# X = np.array([max_densities]).T
y = penetrations
assert X.shape == (32, 2)
assert y.shape == (32,)


# %%
seed = 41
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# Initialize and train the Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=seed, max_depth=2)
tree_clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred_train = tree_clf.predict(X_train)
y_pred_test = tree_clf.predict(X_test)
print("TRAIN")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("Classification Report:\n", classification_report(y_train, y_pred_train))
print("TEST")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))

# Print the tree rules
# tree_rules = export_text(tree_clf, feature_names=['x1'])
tree_rules = export_text(tree_clf, feature_names=['x1', 'x2'])
print(tree_rules)
# %%
