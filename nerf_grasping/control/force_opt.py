import numpy as np
import torch
import cvxpy as cp


def skew_matrix(vectors):
    skew = np.zeros(vectors.shape[:-1] + (3, 3))

    skew[..., 0, 1] = -vectors[..., 2]
    skew[..., 1, 2] = -vectors[..., 0]
    skew[..., 2, 0] = -vectors[..., 1]
    skew[..., 1, 0] = vectors[..., 2]
    skew[..., 2, 1] = vectors[..., 0]
    skew[..., 0, 2] = vectors[..., 1]

    return skew


def example_rotation_transform(normals):
    # hopefully no one will try grabing directly under or above
    global_z_axis = np.array([0, 0, 1])

    #  n,3, 1      3, 3                       n, 3, 1
    local_x = skew_matrix(global_z_axis) @ normals[..., None]

    #  n,3,1         n,3,3              n,3,1
    local_y = skew_matrix(normals) @ local_x

    local_x /= np.linalg.norm(local_x, keepdims=True, axis=-2)
    local_y /= np.linalg.norm(local_y, keepdims=True, axis=-2)

    rotations = np.stack([local_x, local_y, normals[..., None]], axis=-1)[..., 0, :]
    return rotations


def calculate_grip_forces(
    positions, normals, target_force, target_torque, target_normal=0.4, mu=0.5
):
    """positions are relative to object CG if we want unbalanced torques"""

    torch_input = type(positions) == torch.Tensor
    if torch_input:
        assert type(normals) == torch.Tensor, "numpy vs torch needs to be consistant"
        assert (
            type(target_force) == torch.Tensor
        ), "numpy vs torch needs to be consistant"
        assert (
            type(target_torque) == torch.Tensor
        ), "numpy vs torch needs to be consistant"
        positions = positions.numpy()
        normals = normals.numpy()
        target_force = target_force.numpy()
        target_torque = target_torque.numpy()

    n, _ = positions.shape
    assert normals.shape == (n, 3)
    assert target_force.shape == (3,)

    # print('object frame positions: ', positions)

    F = cp.Variable((n, 3))
    constraints = []

    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    total_force = np.zeros((3))
    total_torque = np.zeros((3))

    Q = []
    for pos, norm, f in zip(positions, normals, F):
        q = example_rotation_transform(norm)
        Q.append(q)

        total_force = total_force + q @ f
        total_torque = total_torque + skew_matrix(pos) @ q @ f

    constraints.append(total_force == target_force)
    constraints.append(total_torque == target_torque)

    friction_cone = cp.norm(F[:, :2], axis=1) <= mu * F[:, 2]
    constraints.append(friction_cone)

    force_magnitudes = cp.norm(F - np.array([[0.0, 0.0, target_normal]]), axis=1)
    # friction_magnitudes = cp.norm(F[:,2], axis=1)
    prob = cp.Problem(cp.Minimize(cp.max(force_magnitudes)), constraints)
    prob.solve(solver=cp.SCS)

    if F.value is None:
        print("Failed to solve!")
        return torch.zeros((3, 3), dtype=torch.float32), False

    global_forces = np.zeros_like(F.value)
    for i in range(n):
        global_forces[i, :] = Q[i] @ F.value[i, :]

    if torch_input:
        global_forces = torch.tensor(global_forces).float()

    return global_forces, True


def check_force_closure(positions, normals, mu=0.5):
    torch_input = type(positions) == torch.Tensor
    if torch_input:
        assert type(normals) == torch.Tensor, "numpy vs torch needs to be consistant"
        positions = positions.numpy()
        normals = normals.numpy()

    n, _ = positions.shape
    assert normals.shape == (n, 3)

    F = cp.Variable((n, 3))
    constraints = []

    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    target_wrench = cp.Parameter(6)
    total_force = np.zeros((3))
    total_torque = np.zeros((3))

    Q = []
    for pos, norm, f in zip(positions, normals, F):
        q = example_rotation_transform(norm)
        Q.append(q)

        total_force = total_force + q @ f
        total_torque = total_torque + skew_matrix(pos) @ q @ f

    constraints.append(total_force == target_wrench[:3])
    constraints.append(total_torque == target_wrench[3:])

    friction_cone = cp.norm(F[:, :2], axis=1) <= mu * F[:, 2]
    constraints.append(friction_cone)

    force_magnitudes = cp.norm(F, axis=1)
    # friction_magnitudes = cp.norm(F[:,2], axis=1)
    prob = cp.Problem(cp.Minimize(cp.max(force_magnitudes)), constraints)
    for ii in range(6):
        for sign in [-1, 1]:
            wrench_val = np.zeros((6))
            wrench_val[ii] = sign
            target_wrench.value = wrench_val

            prob.solve(solver=cp.SCS)

            if F.value is None:
                print("Not in force closure!")
                print("dim: ", ii, ", sign: ", sign)
                return
