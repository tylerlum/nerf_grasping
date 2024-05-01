# %%
import plotly.graph_objects as go
import trimesh
import pathlib

# %%
ground_truth_mesh_path = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata/ddg-ycb_077_rubiks_cube/coacd/decomposed.obj")
nerf_mesh_200iters_10density = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/nerf_grasping/baselines/nerf_meshdata_200iters_10density/ddg-ycb_077_rubiks_cube/coacd/decomposed.obj")
assert ground_truth_mesh_path.exists(), ground_truth_mesh_path
assert nerf_mesh_200iters_10density.exists(), nerf_mesh_200iters_10density

# %%
ground_truth_mesh = trimesh.load_mesh(ground_truth_mesh_path)
nerf_mesh = trimesh.load_mesh(nerf_mesh_200iters_10density)

# %%
fig = go.Figure()
fig.add_trace(
    go.Mesh3d(
        x=ground_truth_mesh.vertices[:, 0],
        y=ground_truth_mesh.vertices[:, 1],
        z=ground_truth_mesh.vertices[:, 2],
        i=ground_truth_mesh.faces[:, 0],
        j=ground_truth_mesh.faces[:, 1],
        k=ground_truth_mesh.faces[:, 2],
        color="lightblue",
        opacity=0.5,
        name="Ground Truth",
    )
)
fig.add_trace(
    go.Mesh3d(
        x=nerf_mesh.vertices[:, 0],
        y=nerf_mesh.vertices[:, 1],
        z=nerf_mesh.vertices[:, 2],
        i=nerf_mesh.faces[:, 0],
        j=nerf_mesh.faces[:, 1],
        k=nerf_mesh.faces[:, 2],
        color="lightpink",
        opacity=0.5,
        name="NeRF",
    )
)
fig.show()

# %%
