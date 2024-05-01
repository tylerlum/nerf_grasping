# %%
# (nerf_grasping_env) ➜  nerf_grasping git:(2024-04-11_NewModel) python nerf_grasping/optimizer.py \
# --init-grasp-config-dict-path /juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/aggregated_evaled_grasp_config_dicts_train/aggregated_evaled_grasp_config_dict_train.npy --output-path /juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/aggregated_evaled_grasp_config_dicts_train_optimized_v2 \
# --grasp-metric.nerf-checkpoint-path /home/tylerlum/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/nerfcheckpoints/core-bottle-1071fa4cddb2da2fc8724d5673a063a6_0_0632/nerfacto/2024-04-09_163057/config.yml \
# --grasp-metric.classifier-config-path /juno/u/tylerlum/github_repos/nerf_grasping/Train_DexGraspNet_NeRF_Grasp_Metric_workspaces/1700rotated_augmented_pose_HALTON_50_cnn-3d-xyz_l2_all_2024-04-10_10-14-17-971253/config.yaml \
# optimizer:sgd-optimizer-config \
# --optimizer.finger-lr 1e-4 \
# --optimizer.grasp-dir-lr 1e-4 \
# --optimizer.wrist-lr 1e-4 \
# --optimizer.num_steps 0 \
# --optimizer.num_grasps 32
# 
# 
#     optimized_grasp_config_dict = get_optimized_grasps(
#         OptimizationConfig(
#             use_rich=True,
#             init_grasp_config_dict_path=args.init_grasp_config_dict_path,
#             grasp_metric=GraspMetricConfig(
#                 nerf_checkpoint_path=nerf_config,
#                 classifier_config_path=args.classifier_config_path,
#                 X_N_Oy=X_N_Oy,
#             ),
#             optimizer=SGDOptimizerConfig(
#                 num_grasps=32,
#                 num_steps=0,
#                 finger_lr=1e-4,
#                 grasp_dir_lr=1e-4,
#                 wrist_lr=1e-4,
#             ),
#             output_path=pathlib.Path(experiment_folder / "optimized_grasp_config_dicts" / "optimized_grasp_config_dict.npy"),
#         )
#     )

# %%

from nerf_grasping.optimizer import get_optimized_grasps
from nerf_grasping.optimizer_utils import get_sorted_grasps_from_dict
from nerf_grasping.config.optimization_config import OptimizationConfig
from nerf_grasping.config.optimizer_config import SGDOptimizerConfig
from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
from nerf_grasping.nerfstudio_train import train_nerfs
from nerf_grasping.baselines.nerf_to_mesh import nerf_to_mesh
from nerf_grasping.nerf_utils import (
    compute_centroid_from_nerf,
)
import trimesh
import nerf_grasping
import pathlib
import tyro
import numpy as np
from dataclasses import dataclass
from nerf_grasping.grasp_utils import (
    get_nerf_configs,
    load_nerf_pipeline,
)

# %%
centroid = np.array([2.44945139e-04, 2.62987996e-02, 7.10774193e-05])
# nerf_centroid: [-0.00014259 -0.01008059 -0.00026239]
# centroid = np.array([-0.00014259, -0.01008059, -0.00026239])

X_N_O = trimesh.transformations.translation_matrix(centroid)  # TODO: Check this

X_O_Oy = trimesh.transformations.rotation_matrix(
    0, [1, 0, 0]
)  # TODO: Check this
X_N_Oy = X_N_O @ X_O_Oy

# %%
centroid

# %%
print(f"X_N_Oy: {X_N_Oy}")
print(f"X_N_O: {X_N_O}")
print(f"X_O_Oy: {X_O_Oy}")

X_N_Oy = np.eye(4)
# X_N_Oy[0, 3] = 0.01
# X_N_Oy[1, 3] = 0.01
# X_N_Oy[2, 3] = 0.01


# %%
optimized_grasp_config_dict = get_optimized_grasps(
    OptimizationConfig(
        use_rich=True,
        init_grasp_config_dict_path=pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/aggregated_evaled_grasp_config_dicts_train/aggregated_evaled_grasp_config_dict_train.npy"),
        output_path=pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/aggregated_evaled_grasp_config_dicts_train_optimized_v9/core-bottle-1071fa4cddb2da2fc8724d5673a063a6_0_0632.npy"),
        grasp_metric=GraspMetricConfig(
            nerf_checkpoint_path=pathlib.Path("/home/tylerlum/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/nerfcheckpoints/core-bottle-1071fa4cddb2da2fc8724d5673a063a6_0_0632/nerfacto/2024-04-09_163057/config.yml"),
            classifier_config_path=pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/Train_DexGraspNet_NeRF_Grasp_Metric_workspaces/1700rotated_augmented_pose_HALTON_50_cnn-3d-xyz_l2_all_2024-04-10_10-14-17-971253/config.yaml"),
            X_N_Oy=X_N_Oy,
            # X_N_Oy=None,
        ),
        optimizer=SGDOptimizerConfig(
            num_grasps=32,
            num_steps=0,
            finger_lr=1e-4,
            grasp_dir_lr=1e-4,
            wrist_lr=1e-4,
        ),
    )
)

# %%
x = np.load('/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/aggregated_evaled_grasp_config_dicts_train_optimized_v3/core-bottle-1071fa4cddb2da2fc8724d5673a063a6_0_0632.npy', allow_pickle=True).item()
# %%
x['trans'] == optimized_grasp_config_dict['trans']

# %%
X_N_Oy

# %%
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(
    go.Mesh3d(

    )
)

# %%
mesh_centroid, nerf_centroid
# %%
