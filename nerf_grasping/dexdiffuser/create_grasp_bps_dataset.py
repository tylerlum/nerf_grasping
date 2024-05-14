# %%
from tqdm import tqdm
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from bps import bps
import pathlib

from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    parse_object_code_and_scale,
)

# %%
path_str = "/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-05-06_rotated_stable_grasps_0/pointclouds_250imgs_400iters_5k/"
path_bigger_str = "/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-05-06_rotated_stable_grasps_bigger_0/pointclouds_250imgs_400iters_5k/"
path_smaller_str = "/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-05-06_rotated_stable_grasps_smaller_0/pointclouds_250imgs_400iters_5k/"

paths = [pathlib.Path(path_str.replace("_0", f"_{i}")) for i in range(7)]
path_biggers = [pathlib.Path(path_bigger_str.replace("_0", f"_{i}")) for i in range(7)]
path_smallers = [
    pathlib.Path(path_smaller_str.replace("_0", f"_{i}")) for i in range(7)
]

all_pointcloud_paths = paths + path_biggers + path_smallers

all_pointcloud_paths += [
    pathlib.Path(str(path).replace("250imgs", "100imgs"))
    for path in all_pointcloud_paths
]

# %%
all_pointcloud_data_paths = []
for path in tqdm(all_pointcloud_paths, desc="Finding data paths"):
    if not path.exists():
        print(f"Path {path} does not exist")
        continue

    data_paths = sorted(list(path.rglob("*.ply")))
    all_pointcloud_data_paths.extend(data_paths)

all_pointcloud_data_paths = sorted(all_pointcloud_data_paths)
print(f"Found {len(all_pointcloud_data_paths)} data paths")

# %%
all_grasp_data_paths = list(
    pathlib.Path(
        "/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-05-09_rotated_stable_grasps_noisy_TUNED/SHAKE_raw_evaled_grasp_config_dicts/"
    ).glob("*.npy")
)
assert len(all_grasp_data_paths) > 0, "No grasp data paths found"
print(f"Found {len(all_grasp_data_paths)} grasp data paths")

# %%
all_grasp_data_dicts = [
    np.load(grasp_data_path, allow_pickle=True).item()
    for grasp_data_path in tqdm(all_grasp_data_paths, desc="Loading grasp data")
]

# %%
N_FINGERS = 4
GRASP_DIM = 3 + 6 + 16 + N_FINGERS * 3
N_BASIS_PTS = 4096
BASIS_RADIUS = 0.3


basis_points = bps.generate_random_basis(
    n_points=N_BASIS_PTS, radius=BASIS_RADIUS, random_seed=13
) + np.array(
    [0.0, BASIS_RADIUS / 2, 0.0]
)  # Shift up to get less under the table
assert basis_points.shape == (
    N_BASIS_PTS,
    3,
), f"Expected shape ({N_BASIS_PTS}, 3), got {basis_points.shape}"

# %%
# Per object
all_points = []
for i, data_path in tqdm(
    enumerate(all_pointcloud_data_paths),
    desc="Getting point clouds",
    total=len(all_pointcloud_data_paths),
):
    point_cloud = o3d.io.read_point_cloud(str(data_path))
    point_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    points = np.asarray(point_cloud.points)
    all_points.append(points)

min_n_pts = min([x.shape[0] for x in all_points])
all_points = np.stack([x[:min_n_pts] for x in all_points])
n_point_clouds, n_point_cloud_pts = all_points.shape[:2]

x_bps = bps.encode(
    all_points,
    bps_arrangement="custom",
    bps_cell_type="dists",
    custom_basis=basis_points,
    verbose=0,
)
assert x_bps.shape == (
    n_point_clouds,
    N_BASIS_PTS,
), f"Expected shape ({n_point_clouds}, {N_BASIS_PTS}), got {x_bps.shape}"

# %%
object_code_and_scale_strs = [x.parents[0].name for x in all_pointcloud_data_paths]
object_codes, object_scales = [], []
for object_code_and_scale_str in object_code_and_scale_strs:
    object_code, object_scale = parse_object_code_and_scale(object_code_and_scale_str)
    object_codes.append(object_code)
    object_scales.append(object_scale)

object_code_and_scale_str_to_idx = {
    object_code_and_scale_str: i
    for i, object_code_and_scale_str in enumerate(object_code_and_scale_strs)
}

# %%
# Per grasp
(
    all_grasps,
    all_grasp_bps_idxs,
    all_passed_evals,
    all_passed_simulations,
    all_passed_penetration_thresholds,
    all_object_states,
) = ([], [], [], [], [], [])
num_data_points = 0
for i, grasp_data_path in tqdm(
    enumerate(all_grasp_data_paths),
    desc="Getting grasps",
    total=len(all_grasp_data_paths),
):
    grasp_config_dict = np.load(grasp_data_path, allow_pickle=True).item()
    object_code_and_scale_str = grasp_data_path.stem
    if object_code_and_scale_str not in object_code_and_scale_str_to_idx:
        print(f"Object code and scale {object_code_and_scale_str} not found")
        continue
    bps_idx = object_code_and_scale_str_to_idx[object_code_and_scale_str]

    trans = grasp_config_dict["trans"]
    rot = grasp_config_dict["rot"]
    joint_angles = grasp_config_dict["joint_angles"]
    grasp_orientations = grasp_config_dict["grasp_orientations"]
    B = trans.shape[0]
    assert trans.shape == (B, 3), f"Expected shape ({B}, 3), got {trans.shape}"
    assert rot.shape == (B, 3, 3), f"Expected shape ({B}, 3, 3), got {rot.shape}"
    assert joint_angles.shape == (
        B,
        16,
    ), f"Expected shape ({B}, 16), got {joint_angles.shape}"
    assert grasp_orientations.shape == (
        B,
        N_FINGERS,
        3,
        3,
    ), f"Expected shape ({B}, 3, 3), got {grasp_orientations.shape}"
    grasp_dirs = grasp_orientations[..., 2]
    grasps = np.concatenate(
        [
            trans,
            rot[..., :2].reshape(B, -1),
            joint_angles,
            grasp_dirs.reshape(B, -1),
        ],
        axis=1,
    )

    passed_evals = grasp_config_dict["passed_eval"]
    passed_simulations = grasp_config_dict["passed_simulation"]
    passed_penetration_thresholds = grasp_config_dict["passed_new_penetration_test"]
    object_state = grasp_config_dict["object_states_before_grasp"]
    assert passed_evals.shape == (
        B,
    ), f"Expected shape ({B},), got {passed_evals.shape}"
    assert passed_simulations.shape == (
        B,
    ), f"Expected shape ({B},), got {passed_simulations.shape}"
    assert passed_penetration_thresholds.shape == (
        B,
    ), f"Expected shape ({B},), got {passed_penetration_thresholds.shape}"
    N_NOISY_GRASPS = 6
    assert object_state.shape == (
        B,
        N_NOISY_GRASPS,
        13,
    ), f"Expected shape ({B}, {N_NOISY_GRASPS}, 13), got {object_state.shape}"
    object_state = object_state[:, 0, :]

    all_grasps.append(grasps)
    all_grasp_bps_idxs.append(np.repeat(bps_idx, B))
    all_passed_evals.append(passed_evals)
    all_passed_simulations.append(passed_simulations)
    all_passed_penetration_thresholds.append(passed_penetration_thresholds)
    all_object_states.append(object_state)
    num_data_points += B

# %%
all_grasps = np.concatenate(all_grasps, axis=0)
all_grasp_bps_idxs = np.concatenate(all_grasp_bps_idxs, axis=0)
all_passed_evals = np.concatenate(all_passed_evals, axis=0)
all_passed_simulations = np.concatenate(all_passed_simulations, axis=0)
all_passed_penetration_thresholds = np.concatenate(
    all_passed_penetration_thresholds, axis=0
)
all_object_states = np.concatenate(all_object_states, axis=0)

NUM_GRASPS = all_grasps.shape[0]
assert (
    all_grasps.shape[0] == NUM_GRASPS
), f"Expected shape ({NUM_GRASPS},), got {all_grasps.shape[0]}"
assert (
    all_grasp_bps_idxs.shape[0] == NUM_GRASPS
), f"Expected shape ({NUM_GRASPS},), got {all_grasp_bps_idxs.shape[0]}"
assert (
    all_passed_evals.shape[0] == NUM_GRASPS
), f"Expected shape ({NUM_GRASPS},), got {all_passed_evals.shape[0]}"
assert (
    all_passed_simulations.shape[0] == NUM_GRASPS
), f"Expected shape ({NUM_GRASPS},), got {all_passed_simulations.shape[0]}"
assert (
    all_passed_penetration_thresholds.shape[0] == NUM_GRASPS
), f"Expected shape ({NUM_GRASPS},), got {all_passed_penetration_thresholds.shape[0]}"
assert (
    all_object_states.shape[0] == NUM_GRASPS
), f"Expected shape ({NUM_GRASPS},), got {all_object_states.shape[0]}"
assert all_passed_penetration_thresholds.shape == (
    NUM_GRASPS,
), f"Expected shape ({NUM_GRASPS},), got {all_passed_penetration_thresholds.shape}"

# %%
print(f"NUM_GRASPS = {NUM_GRASPS}")


# %%
import h5py

OUTPUT_FILEPATH = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-05-14_rotated_stable_grasps_bps/data.h5"
)
assert not OUTPUT_FILEPATH.exists(), f"{OUTPUT_FILEPATH} already exists"

OUTPUT_FILEPATH.parent.mkdir(exist_ok=True, parents=True)


MAX_NUM_POINT_CLOUDS = n_point_clouds
MAX_NUM_GRASPS = NUM_GRASPS

hdf5_file = h5py.File(OUTPUT_FILEPATH, "w")


# Just one
basis_points_dataset = hdf5_file.create_dataset(
    "/basis_points", shape=(N_BASIS_PTS, 3), dtype="f"
)

# Per object
bpss_dataset = hdf5_file.create_dataset(
    "/bpss",
    shape=(
        MAX_NUM_POINT_CLOUDS,
        N_BASIS_PTS,
    ),
    dtype="f",
)
point_cloud_filepath_dataset = hdf5_file.create_dataset(
    "/point_cloud_filepath",
    shape=(MAX_NUM_POINT_CLOUDS,),
    dtype=h5py.string_dtype(),
)
object_code_dataset = hdf5_file.create_dataset(
    "/object_code", shape=(MAX_NUM_POINT_CLOUDS,), dtype=h5py.string_dtype()
)
object_scale_dataset = hdf5_file.create_dataset(
    "/object_scale", shape=(MAX_NUM_POINT_CLOUDS,), dtype="f"
)

# Per grasp
grasps_dataset = hdf5_file.create_dataset(
    "/grasps", shape=(MAX_NUM_GRASPS, GRASP_DIM), dtype="f"
)
grasp_bps_idx_dataset = hdf5_file.create_dataset(
    "/grasp_bps_idx", shape=(MAX_NUM_GRASPS,), dtype="i"
)
passed_eval_dataset = hdf5_file.create_dataset(
    "/passed_eval",
    shape=(MAX_NUM_GRASPS,),
    dtype="f",
)
passed_simulation_dataset = hdf5_file.create_dataset(
    "/passed_simulation",
    shape=(MAX_NUM_GRASPS,),
    dtype="f",
)
passed_penetration_threshold_dataset = hdf5_file.create_dataset(
    "/passed_penetration_threshold",
    shape=(MAX_NUM_GRASPS,),
    dtype="f",
)
object_state_dataset = hdf5_file.create_dataset(
    "/object_state",
    shape=(
        MAX_NUM_GRASPS,
        13,
    ),
    dtype="f",
)
grasp_idx_dataset = hdf5_file.create_dataset(
    "/grasp_idx", shape=(MAX_NUM_GRASPS,), dtype="i"
)

# %%
# Just one
basis_points_dataset[:] = basis_points

# Per object
bpss_dataset[:n_point_clouds] = x_bps
point_cloud_filepath_dataset[:n_point_clouds] = [
    str(x) for x in all_pointcloud_data_paths
]
object_code_dataset[:n_point_clouds] = object_codes
object_scale_dataset[:n_point_clouds] = object_scales

# Per grasp
grasps_dataset[:NUM_GRASPS] = all_grasps
grasp_bps_idx_dataset[:NUM_GRASPS] = all_grasp_bps_idxs
passed_eval_dataset[:NUM_GRASPS] = all_passed_evals
passed_simulation_dataset[:NUM_GRASPS] = all_passed_simulations
passed_penetration_threshold_dataset[:NUM_GRASPS] = all_passed_penetration_thresholds
object_state_dataset[:NUM_GRASPS] = all_object_states
hdf5_file.attrs["num_grasps"] = NUM_GRASPS


# %%
hdf5_file.close()

# %%
