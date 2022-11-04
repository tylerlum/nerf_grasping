import matplotlib.pyplot as plt
from nerf_grasping import nerf_utils, grasp_utils


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


# https://stackoverflow.com/questions/59879666/plot-trimesh-object-like-with-axes3d-plot-trisurf


def plot_object_mesh(mesh, scatter=False):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    if scatter:
        ax.scatter(
            *[obj.gt_mesh.vertices[:, ii] for ii in range(3)], c="blue", alpha=0.025
        )
    else:
        ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            triangles=mesh.faces,
            Z=mesh.vertices[:, 2],
        )
