# %%
import pathlib

from nerf_grasping.optimizer import get_optimized_grasps
from nerf_grasping.optimizer_utils import (
    get_sorted_grasps_from_dict,
    GraspMetric,
    DepthImageGraspMetric,
    load_classifier,
    load_depth_image_classifier,
)
from nerf_grasping.config.nerfdata_config import DepthImageNerfDataConfig
from nerf_grasping.config.optimization_config import OptimizationConfig
from nerf_grasping.config.optimizer_config import SGDOptimizerConfig
from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
from nerf_grasping.nerfstudio_train import train_nerfs_return_trainer
from nerf_grasping.baselines.nerf_to_mesh import nerf_to_mesh
from nerf_grasping.nerf_utils import (
    compute_centroid_from_nerf,
)
from nerf_grasping.config.classifier_config import ClassifierConfig
import trimesh
import pathlib
import tyro
import numpy as np
from dataclasses import dataclass
import plotly.graph_objects as go



# %%
DATA_DIR = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/2024-04-25_ALBERT_data/nerfdata/new_mug_0_9999")
assert DATA_DIR.exists(), f"{DATA_DIR} does not exist"
assert (DATA_DIR / "transforms.json").exists(), f"{DATA_DIR / 'transforms.json'} does not exist"
assert (DATA_DIR / "images").exists(), f"{DATA_DIR / 'images'} does not exist"

# %%
nerf_trainer = train_nerfs_return_trainer.train_nerf(
    args=train_nerfs_return_trainer.Args(
        nerfdata_folder=DATA_DIR,
        nerfcheckpoints_folder=pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/2024-04-25_ALBERT_data/nerfcheckpoints"),
    )
)
nerf_model = nerf_trainer.pipeline.model
nerf_field = nerf_model.field
nerf_config = nerf_trainer.config.get_base_dir() / "config.yml"
assert nerf_config.exists(), f"{nerf_config} does not exist"

#     print("\n" + "=" * 80)
#     print("Step 3: Convert NeRF to mesh")
#     print("=" * 80 + "\n")
#     nerf_to_mesh_folder = experiment_folder / "nerf_to_mesh"
#     nerf_to_mesh_folder.mkdir(parents=True, exist_ok=True)
#     mesh_N = nerf_to_mesh(
#         field=nerf_field,
#         level=args.density_levelset_threshold,
#         lb=lb_N,
#         ub=ub_N,
#         save_path=nerf_to_mesh_folder / f"{args.object_name}.obj",
#     )  # TODO: Maybe tune other default params, but prefer not to need to
# 
#     print("\n" + "=" * 80)
#     print(
#         "Step 4: Compute X_N_Oy (transformation of the object y-up frame wrt the nerf frame)"
#     )
#     print("=" * 80 + "\n")
#     USE_MESH = False
#     mesh_centroid_N = mesh_N.centroid
#     nerf_centroid_N = compute_centroid_from_nerf(
#         nerf_field,
#         lb=lb_N,
#         ub=ub_N,
#         level=args.density_levelset_threshold,
#         num_pts_x=100,
#         num_pts_y=100,
#         num_pts_z=100,
#     )
#     print(f"mesh_centroid_N: {mesh_centroid_N}")
#     print(f"nerf_centroid_N: {nerf_centroid_N}")
#     centroid_N = mesh_centroid_N if USE_MESH else nerf_centroid_N
#     print(f"USE_MESH: {USE_MESH}, centroid_N: {centroid_N}")
#     assert centroid_N.shape == (3,), f"centroid_N.shape is {centroid_N.shape}, not (3,)"
#     X_N_O = trimesh.transformations.translation_matrix(centroid_N)
# 
#     X_N_Oy = X_N_O @ X_O_Oy
#     X_Oy_N = np.linalg.inv(X_N_Oy)
#     assert X_N_Oy.shape == (4, 4), f"X_N_Oy.shape is {X_N_Oy.shape}, not (4, 4)"
# 
#     # For debugging
#     mesh_Oy = trimesh.Trimesh(vertices=mesh_N.vertices, faces=mesh_N.faces)
#     mesh_Oy.apply_transform(X_Oy_N)
#     nerf_to_mesh_Oy_folder = experiment_folder / "nerf_to_mesh_Oy"
#     nerf_to_mesh_Oy_folder.mkdir(parents=True, exist_ok=True)
#     mesh_Oy.export(nerf_to_mesh_Oy_folder / f"{args.object_name}.obj",)
#     mesh_centroid_Oy = transform_point(X_Oy_N, centroid_N)
#     nerf_centroid_Oy = transform_point(X_Oy_N, centroid_N)

# %%
mesh_N = self.nerf_to_mesh(
    nerf_field,
    self.level,
    min_len=200,  # connected components with fewer than min_len edges are destroyed
    npts=51,  # grid size for marching cubes
    lb=np.array([-0.1, -0.1, -0.1]),  # include some of the object below the table
    ub=np.array([0.1, 0.1, 0.3]),
    save_path="/tmp/mesh_viz_object.obj",
)  # only used for baseline

# step 2: compute the transformation of the object wrt the nerf frame
# [NOTE] could be that computing the centroid from the mesh is better, but is more
# sensitive to the boundaries you choose
lb_N = np.array([-0.2, -0.2, 0.0])
ub_N = np.array([0.2, 0.2, 0.3])
centroid_N = compute_centroid_from_nerf(
    nerf_field,
    lb=lb_N,
    ub=ub_N,
    level=self.level,
    num_pts_x=100,
    num_pts_y=100,
    num_pts_z=100,
)

X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])  # Z up
X_N_O = trimesh.transformations.translation_matrix(centroid_N)
X_N_Oy = X_N_O @ X_O_Oy
X_Oy_N = np.linalg.inv(X_N_Oy)

# step 3: load the grasp metric
# if self.grasp_metric is None:
classifier_config_path = Path("/home/user/dev_ws/src/nerf_custom/nerf_grasping_ros/stanford/checkpoints/config.yaml")
classifier_config = tyro.extras.from_yaml(
    ClassifierConfig, classifier_config_path.open()
)

# load the classifier itself
classifier_model = load_classifier(classifier_config=classifier_config)
self.grasp_metric = GraspMetric(
    nerf_field=nerf_field,
    classifier_model=classifier_model,
    fingertip_config=classifier_config.nerfdata_config.fingertip_config,
    X_N_Oy=X_N_Oy,
)

# step 4: compute optimized grasps as grasp_config_dicts
init_grasp_config_dict_path = Path("/home/user/dev_ws/src/nerf_custom/nerf_grasping_ros/stanford/aggregated/aggregated_evaled_grasp_config_dict_train.npy")
optimized_grasp_config_dict = get_optimized_grasps(
    cfg=OptimizationConfig(
        use_rich=True,
        init_grasp_config_dict_path=init_grasp_config_dict_path,
        grasp_metric=None,  # This is not used because we are passing in a grasp_metric
        optimizer=SGDOptimizerConfig(
            num_grasps=num_grasps,  # [TODO] choose a good value for this - try to minimize inference time
            num_steps=0,
            finger_lr=1e-4,
            grasp_dir_lr=1e-4,
            wrist_lr=1e-4,
        ),
        output_path=Path(
            "/home/user/dev_ws/src/nerf_custom/stanford_experiments"
            + "/optimized_grasp_config_dicts"
            + "/" + datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"/{self.object_name}.npy"
        ),
    ),
    grasp_metric=self.grasp_metric,
)

# step 5: convert the optimized grasps to joint angles
X_Oy_Hs, q_algr_pres, q_algr_posts = (
    get_sorted_grasps_from_dict(
        optimized_grasp_config_dict=optimized_grasp_config_dict,
        error_if_no_loss=True,
        check=False,
        print_best=False,
    )
)

# passing the computed grasps to the next part of the pipeline
grasps = []
for i in range(num_grasps):
    X_Oy_H = X_Oy_Hs[i]  # wrist pose wrt the object frame
    X_W_N = trimesh.transformations.translation_matrix([self.nerf_frame_offset, 0, 0])
    X_W_H = X_W_N @ X_N_Oy @ X_Oy_H