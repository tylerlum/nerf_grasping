# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visualize DexGraspNet NeRF Grasps
#
# ## Summary (Jul 23, 2023)
#
# The purpose of this script is to load a NeRF object model and labeled grasps on this object, and then visualize it

# %%
import math
import nerf_grasping
from localscope import localscope
from nerf import utils
import torch
import os
import trimesh
import numpy as np
from plotly import graph_objects as go
from dataclasses import dataclass
import matplotlib.pyplot as plt

# %%
# PARAMS
dexgraspnet_data_root = "/juno/u/tylerlum/github_repos/DexGraspNet/data"
dexgraspnet_meshdata_root = os.path.join(dexgraspnet_data_root, "meshdata")
dexgraspnet_dataset_root = os.path.join(
    dexgraspnet_data_root,
    "2023-07-01_dataset_DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS_v2",
)
mesh_path = os.path.join(
    dexgraspnet_meshdata_root, "sem-VideoGameConsole-49ba4f628a955bf03742135a31826a22", "coacd", "decomposed.obj"
)
dataset_path = os.path.join(dexgraspnet_dataset_root, "sem-VideoGameConsole-49ba4f628a955bf03742135a31826a22.npy")
nerf_checkpoint_folder = "2023-07-25_nerf_checkpoints"
nerf_model_workspace = "sem-VideoGameConsole-49ba4f628a955bf03742135a31826a22_0_06"
nerf_size_scale = 0.06
nerf_bound = 2.0
nerf_scale = 1.0

# %%
grasp_dataset = np.load(dataset_path, allow_pickle=True)
grasp_dataset.shape

# %%
for data_dict in grasp_dataset:
    scale = data_dict["scale"]
    if not math.isclose(scale, nerf_size_scale, rel_tol=1e-3):
        continue
    link_name_to_contact_candidates = data_dict["link_name_to_contact_candidates"]
    link_name_to_target_contact_candidates = data_dict[
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
    print(f"contact_candidates.shape: {contact_candidates.shape}")
    print(f"target_contact_candidates.shape: {target_contact_candidates.shape}")
    break

# %%
mesh = trimesh.load(mesh_path, force="mesh")

# %%
mesh.centroid

# %%
mesh.apply_transform(trimesh.transformations.scale_matrix(nerf_size_scale))

# %%
mesh.centroid


# %%
@dataclass
class Bounds3D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def max_bounds(self, other):
        assert isinstance(other, Bounds3D)
        return Bounds3D(
            x_min=min(self.x_min, other.x_min),
            x_max=max(self.x_max, other.x_max),
            y_min=min(self.y_min, other.y_min),
            y_max=max(self.y_max, other.y_max),
            z_min=min(self.z_min, other.z_min),
            z_max=max(self.z_max, other.z_max),
        )

    def extend(self, scale: float):
        return Bounds3D(
            x_min=self.x_min * scale,
            x_max=self.x_max * scale,
            y_min=self.y_min * scale,
            y_max=self.y_max * scale,
            z_min=self.z_min * scale,
            z_max=self.z_max * scale,
        )

    @property
    def x_range(self):
        return self.x_max - self.x_min

    @property
    def y_range(self):
        return self.y_max - self.y_min

    @property
    def z_range(self):
        return self.z_max - self.z_min


# %%
@localscope.mfc
def get_bounds(mesh: trimesh.Trimesh) -> Bounds3D:
    min_bounds, max_bounds = mesh.bounds
    return Bounds3D(
        x_min=min_bounds[0],
        x_max=max_bounds[0],
        y_min=min_bounds[1],
        y_max=max_bounds[1],
        z_min=min_bounds[2],
        z_max=max_bounds[2],
    )


@localscope.mfc
def get_scene_dict(mesh: trimesh.Trimesh):
    # bounds = get_bounds(mesh)
    # return dict(
    #     xaxis=dict(title="X", range=[bounds.x_min, bounds.x_max]),
    #     yaxis=dict(title="Y", range=[bounds.y_min, bounds.y_max]),
    #     zaxis=dict(title="Z", range=[bounds.z_min, bounds.z_max]),
    #     aspectratio=dict(x=bounds.x_range, y=bounds.y_range, z=bounds.z_range),
    #     aspectmode="manual",
    # )
    return dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(title="Z"),
        aspectmode="data",
    )


# %%
@localscope.mfc
def plot_mesh(mesh, color="lightpink"):
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
        scene=get_scene_dict(mesh),
        showlegend=True,
        title="Mesh",
    )

    # Create the figure
    fig = go.Figure(data=[mesh_plot], layout=layout)

    # Return the figure
    return fig


# # %%
# # TODO REMOVE
# root_dir = nerf_grasping.get_repo_root()
# nerf_checkpoints = os.listdir(f"{root_dir}/{nerf_checkpoint_folder}")
# num_ok = 0
# for nerf_checkpoint in nerf_checkpoints:
#     path = f"{root_dir}/{nerf_checkpoint_folder}/{nerf_checkpoint}/checkpoints"
#     if not os.path.exists(path):
#         print(f"path {path} does not exist")
#         continue
#     num_checkpoints = len(os.listdir(path))
#     if num_checkpoints > 0:
#         # print(f"nerf_checkpoint: {nerf_checkpoint}, num_checkpoints: {num_checkpoints}")
#         print(nerf_checkpoint)
#         num_ok += 1
# 
# print(num_ok)
# # %%
# # TODO REMOVE
# workspaces = [
#     "core-pistol-ac88c6856c21ab422a79dd7a0c99f28d_0_10",
#     "core-cellphone-c65f71b54023ee9897d7ccf55973b548_0_06",
#     "core-jar-40f0d91382abe04b2396ca3dd50467ab_0_15",
#     "sem-USBStick-6484ba8442fc7c8829545ddc91df1dc1_0_06",
#     "sem-Book-c7f991b1a9bfcff0fe1e2f026632da15_0_06",
#     "core-camera-fdcd83539b8db2c8b5635bf39f10a28a_0_08",
#     "core-cellphone-52a81d42c352a903a0eb5a85db887292_0_12",
#     "mujoco-Schleich_Spinosaurus_Action_Figure_0_06",
#     "core-jar-d4b9e1085ebd4f226bc258c0f0234be0_0_08",
#     "sem-Camera-4b99c1df215aa8e0fb1dc300162ac931_0_12",
#     "sem-Thumbtack-42ece945238a9f7a8877c667ba5c2021_0_08",
#     "core-jar-763474ce228585bf687ad2cd85bde80a_0_15",
#     "sem-Book-c7f991b1a9bfcff0fe1e2f026632da15_0_15",
#     "sem-Bottle-3108a736282eec1bc58e834f0b160845_0_12",
#     "mujoco-Reebok_SH_COURT_MID_II_0_12",
#     "sem-FoodItem-9ffc98584d1c0ec218c8c60c1a0cb5ed_0_10",
#     "sem-FoodItem-6868aac7c700ebecb52e9c8db06cc58b_0_08",
#     "ddg-gd_box_poisson_001_0_06",
#     "ddg-gd_box_poisson_001_0_08",
#     "core-can-56dfb6a30f498643bbf0c65ae96423ae_0_12",
#     "core-pistol-8c944c84863d3d6254b976bcc599b162_0_15",
#     "core-knife-850cc847a23896206cde72e597358f67_0_15",
#     "core-pillow-f3833476297f19c664b3b9b23ddfcbc_0_12",
#     "mujoco-Horse_Dreams_Pencil_Case_0_12",
#     "mujoco-Womens_Cloud_Logo_Authentic_Original_Boat_Shoe_in_Black_Supersoft_8LigQYwf4gr_0_08",
#     "core-camera-fdcd83539b8db2c8b5635bf39f10a28a_0_06",
#     "sem-Piano-2d830fc20d8095cac2cc019b058015_0_15",
#     "mujoco-Womens_Bluefish_2Eye_Boat_Shoe_in_White_Tumbled_YG44xIePRHw_0_08",
#     "mujoco-Office_Depot_Dell_Series_1_Remanufactured_Ink_Cartridge_TriColor_0_12",
#     "core-jar-44e3fd4a1e8ba7dd433f5a7a254e9685_0_06",
#     "mujoco-ASICS_GELResolution_5_Flash_YellowBlackSilver_0_08",
#     "ddg-gd_dumpbell_poisson_000_0_15",
#     "mujoco-Perricoen_MD_No_Concealer_Concealer_0_12",
#     "core-jar-5bbc259497af2fa15db77ed1f5c8b93_0_08",
#     "core-pillow-b422f9f038fc1f4da3149acda85b1964_0_12",
#     "core-mug-ea127b5b9ba0696967699ff4ba91a25_0_15",
#     "sem-FoodItem-9ffc98584d1c0ec218c8c60c1a0cb5ed_0_08",
#     "sem-Radio-215ce10da9e958ae4c40f34de8f3bdb8_0_12",
#     "sem-Car-71ecab71f04e7cd235c52f8f88910645_0_06",
#     "ddg-gd_donut_poisson_000_0_15",
#     "mujoco-Tieks_Ballet_Flats_Diamond_White_Croc_0_08",
#     "sem-LightBulb-e19e45f9d13f05a4bfae4699de9cb91a_0_08",
#     "core-jar-44e3fd4a1e8ba7dd433f5a7a254e9685_0_10",
#     "core-bottle-ed55f39e04668bf9837048966ef3fcb9_0_08",
#     "sem-Car-f9c2bc7b4ef896e7146ff63b4c7525d9_0_08",
#     "sem-Book-d8d4004791c4f61b80fa98b5eeb7036c_0_08",
#     "mujoco-Perricone_MD_Photo_Plasma_0_10",
#     "core-pillow-f3833476297f19c664b3b9b23ddfcbc_0_10",
#     "sem-Bottle-738d7eb5c8800842f8060bac8721179_0_06",
#     "core-bottle-8a0320b2be22e234d0d131ea6d955bf0_0_10",
#     "sem-Piano-1b76644af9341db74a630b59d0e937b5_0_15",
#     "sem-VideoGameConsole-49ba4f628a955bf03742135a31826a22_0_06",
#     "core-pistol-aec662fe0a40e53df4b175375c861e62_0_08",
#     "core-mug-b6f30c63c946c286cf6897d8875cfd5e_0_12",
#     "core-pistol-aec662fe0a40e53df4b175375c861e62_0_06",
#     "core-bottle-f47cbefc9aa5b6a918431871c8e05789_0_15",
#     "ddg-gd_watering_can_poisson_003_0_12",
#     "sem-CellPhone-6c7fc79a5028bd769caad6e0fbf3962c_0_08",
#     "core-pistol-1e93ef2704131b52e111721a37269b0f_0_10",
#     "core-pistol-aec662fe0a40e53df4b175375c861e62_0_15",
#     "sem-USBStick-a2d20c909ed9c6d85d723f8969b531b_0_12",
#     "mujoco-Balderdash_Game_0_06",
#     "ddg-ycb_063-a_marbles_0_12",
#     "sem-Tank-79abfbd42cb5a78f0985368fed75674_0_10",
#     "sem-Camera-4b99c1df215aa8e0fb1dc300162ac931_0_10",
#     "core-mug-b6f30c63c946c286cf6897d8875cfd5e_0_06",
#     "core-cellphone-c8948cb8ec0f10ebc2287ee053005989_0_12",
#     "core-cellphone-e8345991892118949a3de19de0ca67aa_0_12",
#     "mujoco-Pokmon_X_Nintendo_3DS_Game_0_06",
# ]
# import subprocess
# # for workspace in workspaces:
# #     path = f"{root_dir}/{nerf_checkpoint_folder}/{workspace}"
# #     new_path = f"{root_dir}/{nerf_checkpoint_folder}_cleaned/{workspace}"
# #     subprocess.run(f"cp -r {path} {new_path}", shell=True, check=True)
# nerf_checkpoints = os.listdir(f"{root_dir}/{nerf_checkpoint_folder}")
# for folder in nerf_checkpoints:
#     if folder not in set(workspaces):
#         command = f"rm -rf {root_dir}/{nerf_checkpoint_folder}/{folder}"
#         print(command)
#         subprocess.run(command, shell=True, check=True)


# %%
fig = plot_mesh(mesh)
fig.show()

# %%


@localscope.mfc(allowed=["nerf_checkpoint_folder"])
def load_nerf(workspace: str, bound: float, scale: float):
    root_dir = nerf_grasping.get_repo_root()

    parser = utils.get_config_parser()
    opt = parser.parse_args(
        [
            "--workspace",
            f"{root_dir}/{nerf_checkpoint_folder}/{workspace}",
            "--fp16",
            "--test",
            "--bound",
            f"{bound}",
            "--scale",
            f"{scale}",
            "--mode",
            "blender",
            f"{root_dir}/torch-ngp",
        ]
    )
    # Use options to determine proper network structure.
    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    # Create uninitialized network.
    model = NeRFNetwork(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
    )

    # Create trainer with NeRF; use its constructor to load network weights from file.
    trainer = utils.Trainer(
        "ngp",
        vars(opt),
        model,
        workspace=opt.workspace,
        criterion=None,
        fp16=opt.fp16,
        metrics=[None],
        use_checkpoint="latest",
    )
    assert len(trainer.stats["checkpoints"]) != 0, "failed to load checkpoint"
    return trainer.model


# %%
nerf_model = load_nerf(
    workspace=nerf_model_workspace, bound=nerf_bound, scale=nerf_scale
)

# %%
nerf_model


# %%
@localscope.mfc
def get_nerf_densities(nerf_model, query_points: torch.Tensor):
    """
    Evaluates density of a batch of grasp points, shape [B, n_f, 3].
    query_points is torch.Tensor in nerf frame
    """
    B, n_f, _ = query_points.shape
    query_points = query_points.reshape(1, -1, 3)

    return nerf_model.density(query_points).reshape(B, n_f)


# %%
@localscope.mfc
def get_query_points_in_bounds(bounds: Bounds3D, n_pts_per_dim: int) -> np.ndarray:
    """
    Returns a batch of query points in the mesh region.
    """
    x = np.linspace(bounds.x_min, bounds.x_max, n_pts_per_dim)
    y = np.linspace(bounds.y_min, bounds.y_max, n_pts_per_dim)
    z = np.linspace(bounds.z_min, bounds.z_max, n_pts_per_dim)
    xv, yv, zv = np.meshgrid(x, y, z)
    return np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)


query_points_mesh_region_obj_frame = get_query_points_in_bounds(
    get_bounds(mesh).extend(1.5), n_pts_per_dim=100
)

# %%

query_points_mesh_region_obj_frame.shape

# %%
from nerf_grasping.grasp_utils import nerf_to_ig, ig_to_nerf

query_points_mesh_region_isaac_frame = np.copy(query_points_mesh_region_obj_frame)
query_points_mesh_region_nerf_frame = ig_to_nerf(
    query_points_mesh_region_isaac_frame, return_tensor=True
)

# %%
nerf_densities_torch = get_nerf_densities(
    nerf_model, query_points_mesh_region_nerf_frame.reshape(1, -1, 3).float().cuda()
).reshape(query_points_mesh_region_nerf_frame.shape[:-1])
nerf_densities = nerf_densities_torch.detach().cpu().numpy()

# %%
points = query_points_mesh_region_obj_frame.reshape(-1, 3)
densities = nerf_densities.reshape(-1)

# %%
USE_PLOTLY = False
if USE_PLOTLY:
    import plotly.express as px

    fig = px.histogram(
        x=densities,
        log_y=True,
        title="Densities",
        labels={"x": "Values", "y": "Frequency"},
    )

    fig.show()
else:
    plt.hist(densities, log=True)
    plt.title("Densities")
    plt.show()


# %%
@localscope.mfc
def get_colored_points_scatter(points, colors):
    assert len(points.shape) == 2 and points.shape[1] == 3

    points_to_plot = points

    # Use plotly to make scatter3d plot
    scatter = go.Scatter3d(
        x=points_to_plot[:, 0],
        y=points_to_plot[:, 1],
        z=points_to_plot[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=colors,
            colorscale="viridis",
            colorbar=dict(title="Density Scale"),
        ),
        name="Query Points Densities",
    )

    return scatter


# %%
threshold = 200.0
filtered_points = points[densities > threshold]
filtered_densities = densities[densities > threshold]
colored_points_scatter = get_colored_points_scatter(
    points=filtered_points, colors=filtered_densities
)

# Add the scatter plot to a figure and display it
fig = plot_mesh(mesh)
fig.add_trace(colored_points_scatter)
# Plot contact_candidates and target_contact_candidates
starts_plot = go.Scatter3d(
    x=contact_candidates[:, 0],
    y=contact_candidates[:, 1],
    z=contact_candidates[:, 2],
    mode="markers",
    marker=dict(
        size=5,
        color="red",
        colorscale="viridis",
    ),
    name="Contact Candidates",
)
ends_plot = go.Scatter3d(
    x=target_contact_candidates[:, 0],
    y=target_contact_candidates[:, 1],
    z=target_contact_candidates[:, 2],
    mode="markers",
    marker=dict(
        size=5,
        color="blue",
        colorscale="viridis",
    ),
    name="Target Contact Candidates",
)
fig.add_trace(starts_plot)
fig.add_trace(ends_plot)


fig.update_layout(legend_orientation="h")

fig.show()


# %%
import numpy as np
from sklearn.cluster import KMeans


def compress_vectors(original_vectors, N):
    # Step 1: Perform k-means clustering
    kmeans = KMeans(n_clusters=N, random_state=42)
    cluster_ids = kmeans.fit_predict(original_vectors)

    # Step 2: Compute the mean of each cluster
    compressed_vectors = np.zeros((N, original_vectors.shape[1]))
    cluster_counts = np.zeros(N)

    for i, cluster_id in enumerate(cluster_ids):
        compressed_vectors[cluster_id] += original_vectors[i]
        cluster_counts[cluster_id] += 1

    for i in range(N):
        if cluster_counts[i] > 0:
            compressed_vectors[i] /= cluster_counts[i]

    # Step 3: Normalize the mean vectors
    # compressed_vectors /= np.linalg.norm(compressed_vectors, axis=1)[:, np.newaxis]

    return compressed_vectors, cluster_ids


# Example usage:
original_vectors = target_contact_candidates - contact_candidates
N = 4
compressed, cluster_ids = compress_vectors(original_vectors, N)
print("Original vectors:")
print(original_vectors)
print("Compressed vectors:")
print(compressed)
print("Cluster IDs for each vector:")
print(cluster_ids)
# %%
# Use plotly to scatter 3d
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=compressed[:, 0],
            y=compressed[:, 1],
            z=compressed[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color="red",
                colorscale="viridis",
            ),
            name="Compressed Vectors",
        ),
        go.Scatter3d(
            x=original_vectors[:, 0],
            y=original_vectors[:, 1],
            z=original_vectors[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color="blue",
                colorscale="viridis",
            ),
            name="Original Vectors",
        ),
    ]
)
fig.show()

# %%
np.linalg.norm(compressed, axis=-1)

# %%
