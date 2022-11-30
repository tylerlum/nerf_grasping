import numpy as np
import torch
import os
from isaacgym import gymapi

from nerf_grasping import grasp_utils, nerf_utils, quaternions


def visualize_grasp_normals(
    gym, viewer, env, rays_o, rays_d, des_z_dist=0.1, colors=None
):
    """Visualizing surface normals at grasp points"""
    if isinstance(rays_o, torch.Tensor):
        ro = rays_o.detach().cpu().numpy()
    else:
        ro = rays_o
    if isinstance(rays_d, torch.Tensor):
        rd = rays_d.detach().cpu().numpy()
    else:
        rd = rays_d
    vertices = []
    if isinstance(des_z_dist, float):
        des_z_dist = [des_z_dist for i in range(3)]
    for i in range(3):
        vertices.append(ro[i])
        vertices.append(ro[i] + rd[i] * des_z_dist[i])
    vertices = np.stack(vertices, axis=0)
    if colors is None:
        colors = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype="float32")

    gym.add_lines(
        viewer,
        env,
        3,
        vertices,
        colors,
    )


def visualize_obj_com(gym, viewer, env, obj):
    vertices = []
    for ii in range(3):
        vertices.append(obj.position.cpu().numpy())
        obj_rot = quaternions.Quaternion.fromWLast(obj.orientation)
        endpoint = np.zeros(3)
        endpoint[ii] = 1
        endpoint = obj.position + obj_rot.rotate(endpoint)
        vertices.append(endpoint.cpu().numpy())

    vertices = np.stack(vertices, axis=0)
    colors = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype="float32")
    gym.add_lines(
        viewer,
        env,
        3,
        vertices,
        colors,
    )


def visualize_circle_markers(gym, env, sim, obj, n_markers=16):
    rad = 0.05
    theta = np.arange(n_markers) * 2 * np.pi / 16
    points = np.stack(
        [np.sin(theta) * rad, np.cos(theta) * rad, np.ones(16) * rad], axis=1
    )
    nerf_pos = grasp_utils.ig_to_nerf(points)
    densities = nerf_utils.nerf_densities(obj.model, nerf_pos.reshape(-1, 1, 3))
    points[:, 2] = 0.03
    densities = densities.cpu().detach().numpy() / 300
    densities = densities.flatten()
    colors = [[int(density > 0.5), int(density <= 0.5), 0] for density in densities]
    marker_handles = visualize_markers(gym, env, sim, points, colors)
    return marker_handles


def visualize_markers(
    gym, env, sim, positions, colors=[[0.0, 1.0, 0.0]] * 3, marker_handles=[]
):
    if marker_handles:
        return reset_marker_positions(gym, env, sim, positions, colors, marker_handles)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.angular_damping = 100.0
    asset_options.max_angular_velocity = 0.0
    asset_options.max_linear_velocity = 0.0
    asset_options.slices_per_cylinder = 40
    marker_handles = []
    for i, pos in enumerate(positions):
        color = colors[i]
        pose = gymapi.Transform()
        pose.p.x = pos[0]
        pose.p.y = pos[1]
        pose.p.z = pos[2]

        marker_asset = gym.create_sphere(sim, 0.005, asset_options)
        actor_handle = gym.create_actor(env, marker_asset, pose, f"marker_{i}", 10, 0)
        gym.set_rigid_body_color(
            env,
            actor_handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(*color),
        )
        marker_handles.append(actor_handle)
    return marker_handles


def reset_marker_positions(gym, env, sim, positions, colors, marker_handles=[]):
    for pos, color, handle in zip(positions, colors, marker_handles):
        state = gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_POS)
        state["pose"]["p"].fill(tuple(pos))
        assert gym.set_actor_rigid_body_states(
            env, handle, state, gymapi.STATE_POS
        ), "gym.set_actor_rigid_body_states failed"
        gym.set_rigid_body_color(
            env,
            handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(*color),
        )
    return marker_handles


def visualize_mesh_bbox(gym, viewer, env, obj, mesh, colors=[1.0, 0.647, 0.0]):
    # gym.clear_lines(viewer)
    position = obj.position
    obj_rot = quaternions.Quaternion.fromWLast(obj.orientation)

    w, h, d = mesh.extents  # X Z Y - Z is up
    vertices = []
    # Generate an array with 8 vertices
    for i in [-0.5, 0.5]:
        for j in [-0.5, 0.5]:
            for k in [-0.5, 0.5]:
                vertex = np.array([i * w, j * h, k * d])
                vertex = position + obj_rot.rotate(vertex)
                vertices.append(vertex.cpu().numpy())
    vertices = np.array(vertices)
    # For 4 vertices on the back face, create an edge to the opposite face
    edges = []
    for i in range(4):
        edges.append([vertices[i], vertices[i + 4]])
    # Add an edge between adjacent vertices
    for x in range(2):
        for i in [0, 3]:
            for j in [1, 2]:
                edges.append([vertices[4 * x + i], vertices[4 * x + j]])
    edges = np.array(edges)
    # Plot the edges using matplotlib
    colors = [[colors]] * 12
    gym.add_lines(
        viewer,
        env,
        12,
        edges,
        colors,
    )


def img_dir_to_vid(image_dir, name="test", cleanup=False):
    import glob
    import os

    import imageio

    writer = imageio.get_writer(f"{image_dir}/{name}.mp4", fps=20)
    img_files = sorted(
        glob.glob(os.path.join(image_dir, "img*.png")),
        key=lambda x: int(x.split("img")[1].split(".")[0]),
    )
    for file in img_files:
        im = imageio.imread(file)
        writer.append_data(im)
    writer.close()
    if cleanup:
        print("removing files")
        for file in img_files:
            os.remove(file)


def save_viewer_frame(gym, sim, viewer, save_dir, img_idx, save_freq=10):
    """Saves frame from viewer to"""
    gym.render_all_camera_sensors(sim)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if img_idx % save_freq == 0:
        gym.write_viewer_image_to_file(
            viewer, os.path.join(save_dir, f"img{img_idx}.png")
        )
