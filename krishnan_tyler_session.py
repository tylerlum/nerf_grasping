# %%
import numpy as np
from nerf_grasping.optimizer_utils import (
    AllegroGraspConfig,
)
import pathlib
import trimesh

# %%
INIT_GRASP_CONFIG_DICT_PATH = pathlib.Path(
    "data/2023-11-23_rubikscuberepeat_labelnoise_2/evaled_grasp_config_dicts/ddg-gd_rubik_cube_poisson_004_00_0_1000.npy"
)
assert INIT_GRASP_CONFIG_DICT_PATH.exists()

MESHDATA_PATH = pathlib.Path(
    "data/meshdata"
)
assert MESHDATA_PATH.exists()
THIS_MESH_PATH = MESHDATA_PATH / "ddg-gd_rubik_cube_poisson_004" / "coacd" /  "decomposed.obj"
assert THIS_MESH_PATH.exists()

# %%
mesh = trimesh.load(THIS_MESH_PATH, force="mesh")
mesh.apply_scale(0.1)

# %%
init_grasp_config_dict = np.load(
    INIT_GRASP_CONFIG_DICT_PATH, allow_pickle=True
).item()

# %%
init_grasp_config_dict["joint_angles"].shape

# %%
init_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
    init_grasp_config_dict
)

# %%
init_grasp_configs


# %%
init_grasp_configs.wrist_pose.shape

# %%
init_grasp_configs.joint_angles.shape

# %%
init_grasp_configs.grasp_dirs.shape

# %%
init_grasp_configs.fingertip_transforms.shape

# %%
import plotly.graph_objects as go
from plotly.graph_objects import Figure
def plot_grasp_config(mesh, grasp_config: AllegroGraspConfig, idx: int = 0) -> Figure:
    fig = go.Figure()

    wrist_pose = grasp_config.wrist_pose
    fingertip_transforms = grasp_config.fingertip_transforms
    grasp_dirs = grasp_config.grasp_dirs

    wrist_pos = wrist_pose[idx, :3].detach().cpu().numpy()
    fingertip_positions = fingertip_transforms[idx, :, :3].detach().cpu().numpy()
    grasp_dirs = grasp_dirs[idx, :, :3].detach().cpu().numpy()

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
    N_FINGERS = 4
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
mu_0

