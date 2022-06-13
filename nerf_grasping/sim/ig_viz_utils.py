import matplotlib.pyplot as plt
import numpy as np
import torch
from isaacgym import gymapi

from nerf_grasping import grasp_utils


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


def visualize_circle_markers(gym, env, sim, obj, n_markers=16):
    rad = 0.05
    theta = np.arange(n_markers) * 2 * np.pi / 16
    points = np.stack(
        [np.sin(theta) * rad, np.cos(theta) * rad, np.ones(16) * rad], axis=1
    )
    nerf_pos = grasp_utils.ig_to_nerf(points)
    densities = grasp_utils.nerf_densities(obj.model, nerf_pos.reshape(-1, 1, 3))
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
    asset_options.fix_base_link = False
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

    rays, weights, z_vals = grasp_utils.get_grasp_distribution(
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


def plot_grasps(points, lines=None, obj_mesh=None, ax=None, c="C1"):
    """
    Plots points and lines (using ax.quiver) if given. Assumes all points and meshes
    are given in ig frame.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    if obj_mesh is not None:
        ax.scatter(*[obj_mesh[:, ii] for ii in range(3)], c="blue", alpha=0.025)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c)
    if lines is not None:
        assert len(lines) == len(
            points
        ), "# of points: {}, # of lines: {}. Must be equal!".format(
            len(points), len(lines)
        )
        for ii in range(len(points)):
            ax.quiver(
                points[ii, 0],
                points[ii, 1],
                points[ii, 2],
                lines[ii, 0],
                lines[ii, 1],
                lines[ii, 2],
                length=0.05,
                normalize=True,
            )
    ax.set_zlim(0.0, 0.06)
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
