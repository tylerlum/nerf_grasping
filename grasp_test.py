from dataclasses import dataclass
from typing import Tuple, Any, Optional, Union
from isaacgym import gymapi, gymtorch, torch_utils
from nerf_grasping import config
from nerf_grasping.quaternions import Quaternion
from nerf_grasping.sim import ig_utils
from nerf_grasping.sim.sim_fingertip import FingertipEnv

import dcargs
import time
import torch
import numpy as np

import os

root_dir = os.path.abspath("./")
asset_dir = f"{root_dir}/assets"


def refresh_tensors(gym, sim):
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)


def step_gym(gym, sim, viewer=None):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    if viewer is not None:
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
    refresh_tensors(gym, sim)


def reset_actors(robot, obj, grasp_vars, viewer):
    robot.reset_actor(grasp_vars)
    obj.reset_actor()
    # step_gym calls gym.simulate, then refreshes tensors
    for i in range(1):
        step_gym(robot.gym, robot.sim, viewer)


def double_reset(robot, obj, grasp_vars, viewer):
    # print(f"robot position before reset: {robot.position}")
    # reset_actor sets actor rigid body states
    reset_actors(robot, obj, grasp_vars, viewer)
    start_pos = robot.get_ftip_start_pos(grasp_vars)
    for pos, handle in zip(start_pos, robot.actors):
        state = robot.gym.get_actor_rigid_body_states(
            robot.env, handle, gymapi.STATE_POS
        )
        print("after reset", state["pose"]["p"])

    print(f"Desired - Actual: {grasp_vars[0] - start_pos}")


def full_reset(robot, obj, root_state_tensor, viewer, grasp_vars):
    # reset_actor sets actor rigid body states
    robot.reset_actor(grasp_vars)
    obj.reset_actor()
    for i in range(4):
        step_gym(robot.gym, robot.sim, viewer)
    gym, env = robot.gym, robot.env
    object_start_pose = gym.get_actor_rigid_body_states(
        env, obj.actor, gymapi.STATE_ALL
    )["pose"]

    robot_start_poses = [
        gym.get_actor_rigid_body_states(robot.env, actor, gymapi.STATE_ALL)["pose"]
        for actor in robot.actors
    ]

    object_init_state = torch.tensor(
        [
            object_start_pose["p"]["x"][0],
            object_start_pose["p"]["y"][0],
            object_start_pose["p"]["z"][0],
            object_start_pose["r"]["x"][0],
            object_start_pose["r"]["y"][0],
            object_start_pose["r"]["z"][0],
            object_start_pose["r"]["w"][0],
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    robot_init_state = torch.stack(
        [
            torch.tensor(
                [
                    start_pose["p"]["x"][0],
                    start_pose["p"]["y"][0],
                    start_pose["p"]["z"][0],
                    start_pose["r"]["x"][0],
                    start_pose["r"]["y"][0],
                    start_pose["r"]["z"][0],
                    start_pose["r"]["w"][0],
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            for start_pose in robot_start_poses
        ]
    )
    gym, env = robot.gym, robot.env
    # reset object state tensor
    object_idx = gym.get_actor_index(env, obj.actor, gymapi.DOMAIN_SIM)
    root_state_tensor[object_idx] = object_init_state.clone()
    actor_indices = []

    for i, actor in enumerate(robot.actors):
        actor_idx = gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)
        actor_indices.append(actor_idx)
    root_state_tensor[actor_indices] = robot_init_state.clone()
    actor_indices = torch_utils.to_torch(
        actor_indices + [object_idx], dtype=torch.long, device="cpu"
    ).to(torch.int32)
    assert gym.set_actor_root_state_tensor_indexed(
        robot.sim,
        gymtorch.unwrap_tensor(root_state_tensor),
        gymtorch.unwrap_tensor(actor_indices),
        len(actor_indices),
    ), "resetting actor_root_state_tensor failed"
    # step_gym calls gym.simulate, then refreshes tensors
    robot.reset_actor(grasp_vars)
    obj.reset_actor()

    for i in range(4):
        step_gym(robot.gym, robot.sim, viewer)


def setup_gym():
    gym = gymapi.acquire_gym()

    sim = ig_utils.setup_sim(gym)
    env = ig_utils.setup_env(gym, sim)
    return gym, sim, env


def random_forces(timestep):
    fx = -np.sin(timestep * np.pi / 10) * 0.025 + 0.001
    fy = -np.sin(timestep * np.pi / 5) * 0.025 + 0.001
    f = np.array([[fx, fy, 0.0]] * 3)
    return f


def closest_point(a, b, p):
    ap = p - a
    ab = b - a
    res = []
    for i in range(3):
        result = a[i] + torch.dot(ap[i], ab[i]) / torch.dot(ab[i], ab[i]) * ab[i]
        res.append(result)
    return res


def setup_viewer(gym, sim, env):
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    # position outside stage
    cam_pos = gymapi.Vec3(0.7, 0.175, 0.6)
    # position above banana
    cam_pos = gymapi.Vec3(0.1, 0.02, 0.4)
    cam_target = gymapi.Vec3(0, 0, 0.2)
    gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)
    return viewer


def compute_potential(points, magnitude=0.01):
    potentials = []
    for i in range(len(points)):
        p = points[i]
        mask = np.arange(len(points)) != i
        other_pts = points[mask]
        p_dists = p - other_pts
        closest_pt = other_pts[np.argmin(np.linalg.norm(p_dists, axis=1).squeeze())]
        potentials.append((closest_pt - p))  # direction away from closest_pt
    potentials = np.stack(potentials)
    potentials = potentials / np.linalg.norm(potentials, axis=1) * magnitude
    return potentials


def get_mode(timestep):
    if timestep % 1000 < 100:
        return "reach"
    if timestep % 1000 < 200:
        return "grasp"
    if timestep % 1000 < 1000:
        return "lift"


def lifting_trajectory(env, grasp_vars):
    """Evaluates a lifting trajectory for a sampled grasp"""
    # double_reset(robot, obj, grasp_vars, viewer)
    # full_reset(robot, obj, root_state_tensor, viewer, grasp_vars)
    env.reset_actors(grasp_vars)

    print(
        "contact dist: ",
        ig_utils.get_mesh_contacts(
            env.obj.gt_mesh,
            env.robot.get_ftip_start_pos(grasp_vars),
            pos_offset=env.obj.position,
            rot_offset=env.obj.orientation,
        )[2],
    )

    grasp_points, grasp_normals = grasp_vars

    start_succ = 0
    fail_count = 0

    for timestep in range(500):
        height_err = (
            env.robot.target_height
            - env.obj.position.cpu().numpy()[-1]
            + env.obj.translation[-1]
        )

        # finds the closest contact points to the original grasp normal + grasp_point ray
        mode = get_mode(timestep)
        # compute potential to closest points
        potential = compute_potential(grasp_points)
        state = env.control(mode, grasp_vars)
        if timestep >= 100 and timestep % 50 == 0:
            print("MODE:", state["mode"])
            print("TIMESTEP:", timestep)
            print("POSITION ERR:", state["pos_err"])
            print("POTENTIAL:", potential)
            print("VELOCITY:", env.robot.velocity)
            print("FORCE MAG:", state["force_mag"])
            # print("Z Force:", f[:, 2])
            # print("OBJECT FORCES:", gym.get_rigid_contact_forces(sim)[obj.actor])
            if state["mode"] == "lift":
                print("HEIGHT_ERR:", height_err)
            # print(f"NET CONTACT FORCE:", net_cf[obj.index,:])
        if (env.robot.position[:, -1] <= 0.01).any():
            print("Finger too low!")
            return False
        if (env.robot.position[:, -1] >= 0.5).any():
            print("Finger too high!")
            return False
        if fail_count > 50:
            print("Too many cvx failures!")
            return False
        # if number of timesteps of grasp success exceeds 3 seconds
        succ_timesteps = 180
        err_bound = 0.03
        if timestep - start_succ >= succ_timesteps:
            return True

        # increment start_succ if height error outside err_bound
        if height_err > err_bound:
            start_succ = timestep
    return height_err <= err_bound


def main():
    exp_config = dcargs.cli(config.EvalExperiment)

    # Loads grasp data
    if exp_config.grasp_data is None:
        grasp_data_path = config.grasp_file(exp_config)
    else:
        assert os.path.exists(
            exp_config.grasp_data
        ), f"{exp_config.grasp_data} does not exist"
        grasp_data_path = exp_config.grasp_data
    grasps = np.load(f"{grasp_data_path}.npy")
    grasp_idx = exp_config.grasp_idx if exp_config.grasp_idx else 0
    # if grasp_idx are start, end indices
    if isinstance(grasp_idx, tuple):
        grasp_idx = grasp_idx[0]

    grasp_vars = (grasps[grasp_idx, :, :3], grasps[grasp_idx, :, 3:])
    env = FingertipEnv(exp_config, grasp_vars)

    # Evaluates sampled grasps
    successes = 0
    if exp_config.grasp_idx is None:
        grasp_ids = range(len(grasps))
    elif isinstance(exp_config.grasp_idx, tuple):
        grasp_ids = np.arange(exp_config.grasp_idx[0], exp_config.grasp_idx[1])
    else:
        grasp_ids = [exp_config.grasp_idx]

    for grasp_idx in grasp_ids:
        grasp_points = torch.tensor(grasps[grasp_idx, :, :3], dtype=torch.float32)
        grasp_normals = torch.tensor(grasps[grasp_idx, :, 3:], dtype=torch.float32)
        grasp_vars = (grasp_points, grasp_normals)

        print(f"EVALUATING GRASP from {grasp_data_path} {grasp_idx}: {grasp_points}")
        success = lifting_trajectory(env, grasp_vars)
        successes += success
        if success:
            print(f"SUCCESS! grasp {grasp_idx}")

    print(
        f"Percent successes: {successes / len(grasp_ids) * 100}% out of {len(grasp_ids)}"
    )


# def main():
#     from nerf_grasping.sim.sim_fingertip import FingertipEnv
#
#     exp_config = dcargs.cli(config.EvalExperiment)
#     env = FingertipEnv(exp_config)
#     print(f"env.object_init_state: {env.object_init_state}")
#     print(f"env.robot_init_state: {env.robot_init_state}")
#     env.reset()
#     print(f"{env.robot.position}")
#     print(f"{env.robot.position}")


if __name__ == "__main__":
    main()
