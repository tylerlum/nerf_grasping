# %%
import numpy as np
import trimesh
import json
import pathlib
import copy

# %%
X_O_Oy = trimesh.transformations.rotation_matrix(
    np.pi / 2, [1, 0, 0]
)  # TODO: Check this

json_path = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG/nerfdata/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846/transforms_original.json")
assert json_path.exists()
output_json_path = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG/nerfdata/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846/transforms.json")
assert not output_json_path.exists()

json_dict = json.loads(json_path.read_text())
# %%
json_dict.keys()

# %%
frames = json_dict["frames"]

# %%
len(frames)

# %%
frame = frames[0]

# %%
frame.keys()

# %%
transform_matrix = np.array(frame["transform_matrix"])

# %%
transform_matrix


# %%
def transform_point(transform_matrix, point):
    point = np.append(point, 1)
    return np.dot(transform_matrix, point)[:3]


def add_transform_matrix_traces(fig, transform_matrix):
    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    origin_transformed = transform_point(transform_matrix, origin)
    x_axis_transformed = transform_point(transform_matrix, x_axis)
    y_axis_transformed = transform_point(transform_matrix, y_axis)
    z_axis_transformed = transform_point(transform_matrix, z_axis)

    for axis, color in zip([x_axis_transformed, y_axis_transformed, z_axis_transformed], ["red", "green", "blue"]):
        fig.add_trace(
            go.Scatter3d(
                x=[origin_transformed[0], axis[0]],
                y=[origin_transformed[1], axis[1]],
                z=[origin_transformed[2], axis[2]],
                mode="lines",
                line=dict(color=color, width=5),
            )
        )

# %%
import plotly.graph_objects as go

fig = go.Figure()
add_transform_matrix_traces(fig, np.eye(4))
add_transform_matrix_traces(fig, transform_matrix)
add_transform_matrix_traces(fig, X_O_Oy @ transform_matrix)
fig.show()

# %%
new_json_dict = copy.deepcopy(json_dict)

# %%
OFFSET = np.array([0.04, 0.02, 0.07])
new_frames = []
for frame in frames:
    new_frame = copy.deepcopy(frame)
    new_transform_matrix = np.array(frame["transform_matrix"])
    new_transform_matrix = X_O_Oy @ new_transform_matrix
    new_transform_matrix[:3, 3] += OFFSET
    new_frame["transform_matrix"] = new_transform_matrix.tolist()
    new_frames.append(new_frame)

new_json_dict["frames"] = new_frames

# %%
with open(output_json_path, "w") as outfile:
    outfile.write(json.dumps(new_json_dict))

# %%
