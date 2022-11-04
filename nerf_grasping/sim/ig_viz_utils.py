import matplotlib.pyplot as plt
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
    asset_options.angular_damping = 0.0
    asset_options.max_angular_velocity = 0.0
    asset_options.slices_per_cylinder = 40
    marker_handles = []
    for i, pos in enumerate(positions):
        color = colors[i]
        pose = gymapi.Transform()
        pose.p.x = pos[0]
        pose.p.y = pos[1]
        pose.p.z = pos[2]

        marker_asset = gym.create_sphere(sim, 0.005, asset_options)
        actor_handle = gym.create_actor(env, marker_asset, pose, f"marker_{i}", 1, 1)
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


def plot_grasp_distribution(
    mu_f, coarse_model, fine_model, renderer, residual_dirs=False
):
    # Test grasp sampling

    rays, weights, z_vals = nerf_utils.get_grasp_distribution(
        mu_f.reshape(1, 3, 6),
        coarse_model,
        fine_model,
        renderer,
        residual_dirs=residual_dirs,
    )
    plt.close("all")
    for ii in range(3):
        plt.plot(
            z_vals[0, ii, :].detach().cpu().numpy().T,
            weights[0, ii, :].cpu().detach().numpy().T,
            label="Finger " + str(ii + 1),
        )
    plt.ylim([0, 0.2])
    plt.title("Grasp Point Distribution")
    plt.xlabel("Distance to Surface [m]")
    plt.ylabel("Probability Mass")
    plt.legend()
    plt.show()


def plot_lines(start_points, lines, ax, color="C0"):
    assert len(lines) == len(
        start_points
    ), "# of points: {}, # of lines: {}. Must be equal!".format(
        len(start_points), len(lines)
    )
    for ii in range(len(start_points)):
        ax.quiver(
            start_points[ii, 0],
            start_points[ii, 1],
            start_points[ii, 2],
            lines[ii, 0],
            lines[ii, 1],
            lines[ii, 2],
            length=0.05,
            normalize=True,
            color=color,
        )
    return


def plot_grasps(points, lines=None, est_lines=None, obj_mesh=None, ax=None, c="C1"):
    """
    Plots points and lines (using ax.quiver) if given. Assumes all points and meshes
    are given in ig frame.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    if obj_mesh is not None:
        ax.scatter(*[obj_mesh[:, ii] for ii in range(3)], c="blue", alpha=0.025)
    # plots grasp points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c)
    if lines is not None:
        plot_lines(points, lines, ax)
    if est_lines is not None:
        plot_lines(points, est_lines, ax, color="orange")

    x_min, x_max = grasp_utils.OBJ_BOUNDS[0]
    y_min, y_max = grasp_utils.OBJ_BOUNDS[1]
    z_min, z_max = grasp_utils.OBJ_BOUNDS[2]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    return ax


def plot_cost_history(cost_hist, cost_fn):
    """Plots CEM cost history min and mean"""
    import torch

    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    ax[0].plot([x[~torch.isinf(x)].min().cpu().numpy().item() for x in cost_hist])
    ax[0].set_title("CEM cost_history (Min)")
    ax[1].plot([x[~torch.isinf(x)].mean().cpu().numpy().item() for x in cost_hist])
    ax[1].set_title("CEM cost_history (Mean)")
    fig.suptitle("Grasp cost fn: {}".format(cost_fn))


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
