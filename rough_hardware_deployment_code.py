from nerf_grasping.optimizer import get_optimized_grasps as TYLER_get_optimized_grasps
from nerf_grasping.optimizer_utils import get_sorted_grasps_from_dict as TYLER_get_sorted_grasps_from_dict
from nerf_grasping.config.optimization_config import OptimizationConfig
from nerf_grasping.config.optimizer_config import SGDOptimizerConfig
from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
import pathlib

nerf_data = ALBERT_run_hardware_nerf_data_collection()
nerf_checkpoint_path = ALBERT_run_hardware_nerf_training(nerf_data)
object_transform_world_frame = get_object_transform_world_frame(nerf_data)

assert nerf_checkpoint_path.exists(), f"{nerf_checkpoint_path} does not exist"
assert nerf_checkpoint_path.suffix == ".yml", f"{nerf_checkpoint_path} is not a .yml file"
assert object_transform_world_frame.shape == (4, 4), f"object_transform_world_frame.shape is {object_transform_world_frame.shape}, not (4, 4)"

INIT_GRASP_CONFIG_DICT_PATH = pathlib.Path("/path/to/init.npy")
NERF_CHECKPOINT_PATH = pathlib.Path("/path/to/nerfcheckpoints/config.yml")
CLASSIFIER_CONFIG_PATH = pathlib.Path("/path/to/classifier/config.yml")

optimized_grasp_config_dict = TYLER_get_optimized_grasps(
    OptimizationConfig(
        use_rich=True,
        init_grasp_config_dict_path=INIT_GRASP_CONFIG_DICT_PATH,
        grasp_metric=GraspMetricConfig(
            nerf_checkpoint_path=nerf_checkpoint_path,
            classifier_config_path=CLASSIFIER_CONFIG_PATH,
        ),
        optimizer=SGDOptimizerConfig(),
        object_transform_world_frame=object_transform_world_frame,
    )
)
wrist_trans, wrist_rot, joint_angles, target_joint_angles = TYLER_get_sorted_grasps_from_dict(
    optimized_grasp_config_dict=optimized_grasp_config_dict,
    object_transform_world_frame=object_transform_world_frame,
    error_if_no_loss=False,
    check=True,
    print_best=True,
)

num_grasps = wrist_trans.shape[0]

for i in range(num_grasps):
    print(f"Trying grasp {i} / {num_grasps}")

    if not ALBERT_is_feasible(wrist_trans[i], wrist_rot[i], joint_angles[i]):
        print(f"Grasp {i} is infeasible")
        continue

    ALBERT_execute_grasp(wrist_trans[i], wrist_rot[i], joint_angles[i], target_joint_angles[i])