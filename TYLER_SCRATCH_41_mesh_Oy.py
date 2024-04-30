# %%
import trimesh
import numpy as np
import pathlib

# %%
# X_W_N = trimesh.transformations.translation_matrix([0.7, 0, 0])


# X_N_O = trimesh.transformations.translation_matrix([-0.04092566, -0.05782086,  0.04981683])
X_N_O = trimesh.transformations.translation_matrix([-0.02423195, -0.00194203,  0.13271753])

# X_N_O = trimesh.transformations.translation_matrix([0, 0, 0.1])
# Z-up
X_O_Oy = trimesh.transformations.rotation_matrix(
    np.pi / 2, [1, 0, 0]
)

X_N_Oy = X_N_O @ X_O_Oy
X_Oy_N = np.linalg.inv(X_N_Oy)

# %%
mesh_N = trimesh.load_mesh("/juno/u/tylerlum/github_repos/nerf_grasping/nerf_grasping/baselines/nerf_meshdata/goblet_0_9999/coacd/decomposed.obj")
mesh_Oy = mesh_N.copy()

# %%
mesh_Oy.apply_transform(X_Oy_N)

# %%
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(
    go.Mesh3d(
        x=mesh_N.vertices[:, 0],
        y=mesh_N.vertices[:, 1],
        z=mesh_N.vertices[:, 2],
        i=mesh_N.faces[:, 0],
        j=mesh_N.faces[:, 1],
        k=mesh_N.faces[:, 2],
        opacity=0.5,
        color="blue",
    )
)
fig.show()

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
        opacity=0.5,
        color="red",
    )
)
fig.show()

# %%
path = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/nerf_grasping/baselines/nerf_meshdata_rotated/goblet_0_9999/coacd/decomposed.obj")
path.parent.mkdir(parents=True, exist_ok=True)
mesh_Oy.export(path)

# %%
