# %%
from nerf_grasping.fr3_algr_ik.ik import solve_ik
import numpy as np
import trimesh
from nerf_grasping.dexgraspnet_utils.hand_model import HandModel
from nerf_grasping.dexgraspnet_utils.hand_model_type import (
    HandModelType,
)
from nerf_grasping.dexgraspnet_utils.pose_conversion import (
    hand_config_to_pose,
)
import plotly.graph_objects as go
import math


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

    for axis, color, name in zip(
        [x_axis_transformed, y_axis_transformed, z_axis_transformed],
        ["red", "green", "blue"],
        ["x", "y", "z"],
    ):
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
# grasp_config_dict = np.load("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-16_rotated_grasps_aggregated_augmented_pose_HALTON_50/aggregated_evaled_grasp_config_dicts_train/aggregated_evaled_grasp_config_dict_train.npy", allow_pickle=True).item()
# grasp_config_dict = np.load("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-16_rotated_grasps_aggregated_augmented_pose_HALTON_50/aggregated_evaled_grasp_config_dicts_train_optimized/ddg-ycb_077_rubiks_cube_0_0545.npy", allow_pickle=True).item()
# grasp_config_dict = np.load("/juno/u/tylerlum/Downloads/ddg-ycb_077_rubiks_cube_0_0545.npy", allow_pickle=True).item()
grasp_config_dict = np.load(
    "/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-16_rotated_grasps_aggregated_augmented_pose_HALTON_50/aggregated_evaled_grasp_config_dicts_train_optimized/new_mug_0_9999.npy",
    allow_pickle=True,
).item()
# grasp_config_dict = np.load("/juno/u/tylerlum/Downloads/new_mug_0_9999.npy", allow_pickle=True).item()

mesh = trimesh.load_mesh(
    "/juno/u/tylerlum/github_repos/nerf_grasping/nerf_grasping/baselines/nerf_meshdata_rotated/new_mug/coacd/decomposed.obj"
)

# %%
GRASP_IDX = 0
trans = grasp_config_dict["trans"]
rot = grasp_config_dict["rot"]
joint_angles = grasp_config_dict["joint_angles"]

n_grasps = trans.shape[0]
assert GRASP_IDX < n_grasps
assert trans.shape == (n_grasps, 3)
assert rot.shape == (n_grasps, 3, 3)
assert joint_angles.shape == (n_grasps, 16)

# %%
X_Oy_H = np.eye(4)
X_Oy_H[:3, :3] = rot[GRASP_IDX]
X_Oy_H[:3, 3] = trans[GRASP_IDX]

# %%
theta = math.radians(90) + math.radians(20)
cos_theta = math.cos(theta)
x_dir = X_Oy_H[:3, 0]
y_dir = X_Oy_H[:3, 1]
z_dir = X_Oy_H[:3, 2]
print(f"X_Oy_H = {X_Oy_H}")
print(f"x_dir = {x_dir}")
print(f"y_dir = {y_dir}")
print(f"z_dir = {z_dir}")
if x_dir[0] < cos_theta:
    print(f"WARNING: x_dir[0] < cos_theta, x_dir[0] = {x_dir[0]}")
    print("Expect this to fail because now palm is pointing toward the robot")
if z_dir[0] < cos_theta:
    print(f"WARNING: z_dir[0] < cos_theta, z_dir[0] = {z_dir[0]}")
    print("Expect this to fail because now middle finger is pointing toward the robot")


# %%
device = "cuda"
hand_model_type = HandModelType.ALLEGRO_HAND
hand_model = HandModel(
    hand_model_type=hand_model_type, device=device, n_surface_points=1000
)

# %%
hand_pose = hand_config_to_pose(trans, rot, joint_angles).to(device)
hand_model.set_parameters(hand_pose)

pregrasp_plot_data = hand_model.get_plotly_data(i=GRASP_IDX, opacity=1.0)


# %%
fig = go.Figure(data=pregrasp_plot_data)
# yup
yup_camera = dict(
    up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25)
)

fig.update_layout(
    scene=dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(title="Z"),
        aspectmode="data",
    ),
    scene_camera=yup_camera,
)
add_transform_matrix_traces(fig, X_Oy_H, length=0.1)
add_transform_matrix_traces(fig, np.eye(4), length=0.1)
fig.add_trace(
    go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        opacity=0.5,
        color="blue",
    )
)
fig.show()

# %%
print(mesh.bounds)


# %%
X_W_N = trimesh.transformations.translation_matrix([0.65, 0, 0])

assert mesh.bounds.shape == (2, 3)
# table_y_Oy = -0.05522743
# X_N_O = trimesh.transformations.translation_matrix([0, 0, -table_y_Oy])
X_N_O = trimesh.transformations.translation_matrix([0.022149, -0.000491, 0.057019])

# Z-up
X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])

# %%
X_W_H = X_W_N @ X_N_O @ X_O_Oy @ X_Oy_H

# %%
print(f"X_W_H = {X_W_H}")
x_dir = X_W_H[:3, 0]
y_dir = X_W_H[:3, 1]
z_dir = X_W_H[:3, 2]
print(f"X_W_H = {X_W_H}")
print(f"x_dir = {x_dir}")
print(f"y_dir = {y_dir}")
print(f"z_dir = {z_dir}")
if x_dir[0] < cos_theta:
    print(f"WARNING: x_dir[0] < cos_theta, x_dir[0] = {x_dir[0]}")
    print("Expect this to fail because now palm is pointing toward the robot")
if z_dir[0] < cos_theta:
    print(f"WARNING: z_dir[0] < cos_theta, z_dir[0] = {z_dir[0]}")
    print("Expect this to fail because now middle finger is pointing toward the robot")

# %%
q_star = solve_ik(X_W_H, joint_angles[GRASP_IDX], visualize=True)

# %%
print(f"q_star = {q_star}")

# %%

# dummy_X_W_H = np.eye(4)
# dummy_X_W_H[:3, 3] = [0.7, 0, 0.05]
# dummy_X_W_H[:3, 0] = [0, 0, -1]
# dummy_X_W_H[:3, 1] = [0, 1, 0]
# dummy_X_W_H[:3, 2] = [1, 0, 0]
# print(f"np.linalg.det(dummy_X_W_H[:3, :3]) = {np.linalg.det(dummy_X_W_H[:3, :3])}")
# dummy_joint_angles = joint_angles[GRASP_IDX]
# dummy_q_star = solve_ik(dummy_X_W_H, dummy_joint_angles)
# %%
# X_W_H = np.array(
#     [
#         [-0.40069854, 0.06362686, 0.91399777, 0.66515265],
#         [-0.367964, 0.90242159, -0.22413731, 0.02321906],
#         [-0.83907259, -0.4261297, -0.33818674, 0.29229766],
#         [0.0, 0.0, 0.0, 1.0],
#     ]
# )
X_W_H = np.array(
    [
        [-0.40069854, 0.06362686, 0.91399777, 0.66515265],
        [-0.367964, 0.90242159, -0.22413731, 0.02321906],
        [-0.83907259, -0.4261297, -0.33818674, 0.29229766],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
q_algr_pre = np.array(
    [
        0.29094562,
        0.7371094,
        0.5108592,
        0.12263706,
        0.12012535,
        0.5845135,
        0.34382993,
        0.605035,
        -0.2684319,
        0.8784579,
        0.8497135,
        0.8972184,
        1.3328283,
        0.34778783,
        0.20921567,
        -0.00650969,
    ]
)
q_star = solve_ik(X_W_H, q_algr_pre, visualize=True)

# %%
from tqdm import tqdm

# %%
# Define the original transformation matrix X_W_H and the initial guess for the joint angles q_algr_pre
X_W_H = np.array(
    [
        [-0.40069854, 0.06362686, 0.91399777, 0.66515265],
        [-0.367964, 0.90242159, -0.22413731, 0.02321906],
        [-0.83907259, -0.4261297, -0.33818674, 0.29229766],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

q_algr_pre = np.array(
    [
        0.29094562,
        0.7371094,
        0.5108592,
        0.12263706,
        0.12012535,
        0.5845135,
        0.34382993,
        0.605035,
        -0.2684319,
        0.8784579,
        0.8497135,
        0.8972184,
        1.3328283,
        0.34778783,
        0.20921567,
        -0.00650969,
    ]
)

# Create a grid of points
grid_size = 10
LENGTH = 0.4
trans_range = np.linspace(-LENGTH, LENGTH, grid_size)
results = []

# Check IK feasibility for each point in the grid
for dx in tqdm(trans_range):
    for dy in trans_range:
        for dz in trans_range:
            # Copy the matrix and modify the translation part
            new_X_W_H = np.copy(X_W_H)
            new_X_W_H[:3, 3] += np.array([dx, dy, dz])

            try:
                # Here you would call the IK solver, such as `solve_ik(new_X_W_H, q_algr_pre)`
                q_star = solve_ik(
                    new_X_W_H, q_algr_pre
                )  # Assuming solve_ik is your IK function
                results.append((dx, dy, dz, True))
                print("SUCCESS")
            except RuntimeError as e:
                results.append((dx, dy, dz, False))
                print("FAIL")

# Print results or process them further
print(results)

# %%
import plotly.graph_objects as go

# Sample data generation (replace with your actual results)
# results = [(dx, dy, dz, success), ...] from your IK tests

# Splitting the results into coordinates and colors based on success
x_pass, y_pass, z_pass = [], [], []
x_fail, y_fail, z_fail = [], [], []

for dx, dy, dz, success in results:
    if success:
        x_pass.append(dx)
        y_pass.append(dy)
        z_pass.append(dz)
    else:
        x_fail.append(dx)
        y_fail.append(dy)
        z_fail.append(dz)

# Create the 3D scatter plot
fig = go.Figure()

# Add successful points in green
fig.add_trace(
    go.Scatter3d(
        x=np.array(x_pass) + X_W_H[0, 3],
        y=np.array(y_pass) + X_W_H[1, 3],
        z=np.array(z_pass) + X_W_H[2, 3],
        mode="markers",
        marker=dict(size=5, color="green", opacity=0.8),  # Success points are green
        name="Success",
    )
)

# Add failed points in red
fig.add_trace(
    go.Scatter3d(
        x=np.array(x_fail) + X_W_H[0, 3],
        y=np.array(y_fail) + X_W_H[1, 3],
        z=np.array(z_fail) + X_W_H[2, 3],
        mode="markers",
        marker=dict(size=5, color="red", opacity=0.8),  # Failed points are red
        name="Failure",
    )
)

# Update the layout to make it more readable
fig.update_layout(
    title="IK Feasibility Results",
    scene=dict(
        xaxis_title="X Translation",
        yaxis_title="Y Translation",
        zaxis_title="Z Translation",
    ),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

add_transform_matrix_traces(fig, np.eye(4), length=0.1)
add_transform_matrix_traces(fig, X_W_N, length=0.1)

# Show the figure
fig.show()
# %%
results = []
for i in tqdm(range(n_grasps)):
    X_Oy_H = np.eye(4)
    X_Oy_H[:3, :3] = rot[i]
    X_Oy_H[:3, 3] = trans[i]

    X_W_H = X_W_N @ X_N_O @ X_O_Oy @ X_Oy_H
    q = joint_angles[i]

    try:
        q_star = solve_ik(X_W_H, q)
        results.append((i, True))
        print("SUCCESS")
    except RuntimeError as e:
        results.append((i, False))
        print("FAIL")


# %%
total = len(results)
num_success = sum([r[-1] for r in results])
print(f"num_success = {num_success} / {total} = {num_success / total}")
pass_idxs = set([r[0] for r in results if r[-1]])

# %%
results2 = []
theta = math.radians(60)
cos_theta = math.cos(theta)

theta2 = math.radians(60)
cos_theta2 = math.cos(theta2)

for i in tqdm(range(n_grasps)):
    X_Oy_H = np.eye(4)
    X_Oy_H[:3, :3] = rot[i]
    X_Oy_H[:3, 3] = trans[i]

    X_W_H = X_W_N @ X_N_O @ X_O_Oy @ X_Oy_H
    q = joint_angles[i]
    x_dir = X_W_H[:3, 0]
    y_dir = X_W_H[:3, 1]
    z_dir = X_W_H[:3, 2]

    finger_point_forward = z_dir[0] > cos_theta
    palm_point_upward = x_dir[2] > cos_theta2

    if finger_point_forward and not palm_point_upward:
        results2.append((i, True))
    else:
        results2.append((i, False))

# %%
total2 = len(results2)
num_success2 = sum([r[-1] for r in results2])
print(f"num_success2 = {num_success2} / {total2} = {num_success2 / total2}")
pass_idxs2 = set([r[0] for r in results2 if r[-1]])

# %%
in_both = pass_idxs.intersection(pass_idxs2)
in_1_not_2 = pass_idxs.difference(pass_idxs2)
in_2_not_1 = pass_idxs2.difference(pass_idxs)
print(f"len(in_both) = {len(in_both)}")
print(f"len(in_1_not_2) = {len(in_1_not_2)}")
print(f"len(in_2_not_1) = {len(in_2_not_1)}")
print(f"in_both[:10] = {list(in_both)[:10]}")
print(f"in_1_not_2[:10] = {list(in_1_not_2)[:10]}")
print(f"in_2_not_1[:10] = {list(in_2_not_1)[:10]}")

# %%
GRASP_IDX = 0
device = "cuda"
hand_model_type = HandModelType.ALLEGRO_HAND
hand_model = HandModel(
    hand_model_type=hand_model_type, device=device, n_surface_points=1000
)

# %%
hand_pose = hand_config_to_pose(trans, rot, joint_angles).to(device)
hand_model.set_parameters(hand_pose)

pregrasp_plot_data = hand_model.get_plotly_data(i=GRASP_IDX, opacity=1.0)


# %%
fig = go.Figure(data=pregrasp_plot_data)
# yup
yup_camera = dict(
    up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25)
)

fig.update_layout(
    scene=dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(title="Z"),
        aspectmode="data",
    ),
    scene_camera=yup_camera,
)
X_Oy_H = np.eye(4)
X_Oy_H[:3, :3] = rot[GRASP_IDX]
X_Oy_H[:3, 3] = trans[GRASP_IDX]
add_transform_matrix_traces(fig, X_Oy_H, length=0.1)
print(f"X_Oy_H = {X_Oy_H}")
fig.show()


# %%
cos_theta

# %%
# Palm down, finger forward
# W is Z-up, X-forward
# H is Z-finger, X-palm
X_W_H = np.array(
    [
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.5],

        [0.0, 0.0, 0.0, 1.0],
    ]
)

q_algr_pre = np.array(
    [
        0.29094562,
        0.7371094,
        0.5108592,
        0.12263706,
        0.12012535,
        0.5845135,
        0.34382993,
        0.605035,
        -0.2684319,
        0.8784579,
        0.8497135,
        0.8972184,
        1.3328283,
        0.34778783,
        0.20921567,
        -0.00650969,
    ]
)
q_star = solve_ik(X_W_H, q_algr_pre, visualize=True)
# %%
q_star

# %%
