"""
Module implementing methods for grasp optimization with NeRFs.
"""
import torch

def grasp_matrix(nerf, grasp_points):
    """
    Constructs a grasp matrix for the object represented by the NeRF density,
    evaluated at a set of grasp points.

    Args:
        nerf: NeRF model whose density represents the object.
        grasp_points: a list of grasp points (torch.Tensors in 3D) at which to
            construct the grasp matrix, shape [B, n_f, 3].

    Returns a grasp matrix (torch.Tensor) for the given grasp.
    """
    B, n_f, _ = grasp_points.shape

    density_grads = nerf_grads(nerf, grasp_points.reshape(-1,3)).reshape(B, n_f, 3)

    density_grads = density_grads / (torch.norm(density_grads, dim=-1, keepdim=True) + 1e-5)

    R = rot_from_vec(density_grads.reshape(-1, 3)).reshape(B, n_f, 3, 3)
    p_cross = skew(grasp_points.reshape(-1,3)).reshape(B, n_f, 3, 3)

    grasp_mats = torch.cat([R, p_cross @ R], dim=-2)
    return torch.cat([grasp_mats[:, ii, :, :] for ii in range(n_f)], dim=-1)

def nerf_densities(nerf, grasp_points):
    """
    Evaluates density of a batch of grasp points, shape [B, n_f, 3].
    """
    query_points = grasp_points.reshape(1, -1, 3)

    return nerf.get_density(query_points)

def nerf_grads(nerf, grasp_points):
    """
    Evaluates NeRF densities at a batch of grasp points.

    Args:
        nerf: nerf_shared.NeRF object with density channel to diff.
        grasp_points: set of grasp points at which to take gradients.
    """
    densities = nerf_densities(nerf, grasp_points)
    density_grads = torch.autograd.grad(torch.sum(densities), grasp_points,
                                        create_graph=True)[0]

    return density_grads

def rot_from_vec(n_z):
    """
    Creates rotation matrix which maps the basis vector e_3 to a vector n_z.
    Gets poorly conditioned when n_z ≅ ±e_3.

    Args:
        n_z: Batch of normal dirs, shape [B, 3].
    """
    # Construct constants.
    n_z = n_z.reshape(-1, 3)
    I = torch.eye(3).reshape(1,3,3).expand(n_z.shape[0], 3, 3)
    e3 = I[:, :, 2]

    # Compute cross product to find axis of rotation.
    v = torch.cross(e3, n_z, dim=-1)
    theta = torch.arccos(torch.sum(e3 * n_z, dim=-1)).reshape(-1, 1, 1)
    K = skew(v)

    return I + torch.sin(theta)*K + (1-torch.cos(theta))*K@K

def skew(v):
    """
    Returns the skew-symmetric form of a batch of 3D vectors v (shape [B, 3]).
    """
    v = v.reshape(-1,3)

    K = torch.zeros(v.shape[0], 3, 3)

    K[:,0,1] = -v[:, 2]
    K[:,0,2] = v[:, 1]
    K[:,1,0] = v[:, 2]
    K[:,1,2] = -v[:, 0]
    K[:,2,0] = -v[:, 1]
    K[:,2,1] = v[:, 0]

    return K

def msv(nerf, grasp_points):
    """
    Evaluates the minimum-singular-value grasp metric proposed in Li and Sastry '88.

    Args:
        nerf: the NeRF model whose density field represents the object.
        grasp_points: a list of grasp points (torch.Tensors in 3D) at which to evaluate
             the grasp metric.

    Returns the minimum singular value of the grasp matrix formed by these points.
    """
    G = grasp_matrix(nerf, grasp_points)
    return torch.mean(torch.prod(torch.linalg.svdvals(G), dim=-1))
