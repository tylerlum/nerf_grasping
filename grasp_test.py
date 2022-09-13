from dataclasses import dataclass
from typing import Tuple, Any, Optional, Union
from isaacgym import gymapi, gymtorch, torch_utils
from nerf_grasping import config
from nerf_grasping.quaternions import Quaternion
from nerf_grasping.sim import ig_utils
from nerf_grasping.sim.sim_fingertip import FingertipEnv

import dcargs
import dataclasses
import time
import torch
import numpy as np
import pdb
import os
import wandb

root_dir = os.path.abspath("./")
asset_dir = f"{root_dir}/assets"


def random_forces(timestep):
    fx = -np.sin(timestep * np.pi / 10) * 0.025 + 0.001
    fy = -np.sin(timestep * np.pi / 5) * 0.025 + 0.001
    f = np.array([[fx, fy, 0.0]] * 3)
    return f


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
    if timestep % 1000 < 50:
        return "reach"
    if timestep % 1000 < 150:
        return "grasp"
    if timestep % 1000 < 1000:
        return "lift"


def lifting_trajectory(env, grasp_vars, step):
    """Evaluates a lifting trajectory for a sampled grasp"""
    # double_reset(env, grasp_vars)
    # full_reset(robot, obj, root_state_tensor, viewer, grasp_vars)
    env.reset_actors(grasp_vars)

    # pdb.set_trace()

    print(
        "contact dist: ",
        ig_utils.get_mesh_contacts(
            env.obj.gt_mesh,
            env.robot.get_ftip_start_pos(grasp_vars),
            pos_offset=env.obj.position,
            rot_offset=env.obj.orientation,
        )[2],
    )

    grasp_points, grasp_normals = grasp_vars  # IG frame.
    if isinstance(grasp_normals, torch.Tensor):
        grasp_points = grasp_points.detach().cpu()
        grasp_normals = grasp_normals.detach().cpu()

    # records timestep where first success occured
    start_succ = 0
    # counts consecutive failures from running cvx opt
    fail_count = 0

    for timestep in range(500):
        height_err = np.abs(
            env.robot.target_height - env.obj.position.cpu().numpy()[-1]
        )

        # finds the closest contact points to the original grasp normal + grasp_point ray
        mode = get_mode(timestep)
        # compute potential to closest points
        potential = compute_potential(grasp_points).sum()
        state = env.run_control(mode, grasp_vars)
        if state["grasp_opt_success"]:
            fail_count = 0
        else:
            fail_count += 1

        if timestep >= 100 and (timestep + 1) % 50 == 0:
            print("MODE:", state["mode"])
            print("TIMESTEP:", timestep)
            # print("POSITION ERR:", state["ftip_pos_err"])
            print("POTENTIAL:", potential)
            print("VELOCITY:", env.robot.velocity)
            print("FORCE MAG:", state["force_mag"])
            # print("Z Force:", f[:, 2])
            # print("OBJECT FORCES:", gym.get_rigid_contact_forces(sim)[obj.actor])
            if state["mode"] == "lift":
                print("HEIGHT ERR:", height_err)
                print("OBJECT VEL-Z:", state["obj_vel-z"])
                print("OBJ POSITION ERR-Z:", state["obj_pos_err-z"])
                print("MAX GRASP POS ERR [~slip]:", state["max_contact_pos_err"])
            # print(f"NET CONTACT FORCE:", net_cf[obj.index,:])

        log = {}
        for k, v in state.items():
            if isinstance(v, float):
                log[k] = v
            elif isinstance(v, (torch.Tensor, np.ndarray)) and np.prod(v.shape) == 1:
                log[k] = float(v)

        log.update(
            dict(
                max_ftip_pos_err=state["max_ftip_pos_err"],
                height_err=height_err,
                force_mag=state["force_mag"],
                final_timestep=timestep,
                final_potentential=potential,
                min_ftip_height=env.robot.position[:, -1].min(),
                max_ftip_height=env.robot.position[:, -1].max(),
                fail_count=fail_count,
            )
        )

        if (env.robot.position[:, -1] <= 0.01).any():
            print("Finger too low!")
            if wandb.run is not None:
                wandb.log(log, step=step)
            return False
        elif (env.robot.position[:, -1] >= 0.5).any():
            print(f"Finger too high!: {env.robot.position[:,-1]}")
            if wandb.run is not None:
                wandb.log(log, step=step)
            return False
        elif torch.isnan(env.robot.position).any():
            print("Finger position is nan!")
            if wandb.run is not None:
                wandb.log(log, step=step)
            return False
        if fail_count > 50:
            print("TIMESTEP:", timestep)
            print("Too many cvx failures!")
            if wandb.run is not None:
                wandb.log(log, step=step)
            return False
        # if number of timesteps of grasp success exceeds 3 seconds
        err_bound = 0.03
        if timestep - start_succ >= 180:
            if wandb.run is not None:
                wandb.log(log, step=step)
            return True

        # increment start_succ if height error outside err_bound
        if height_err > err_bound:
            start_succ = timestep
    if wandb.run is not None:
        wandb.log(log, step=step)
    return height_err <= err_bound


def main():
    exp_config = dcargs.cli(config.EvalExperiment)
    if exp_config.wandb:
        # update config dict to include path to grasp_data
        wandb.init(project="nerf_grasping", config=dataclasses.asdict(exp_config))
    # Loads grasp data
    if exp_config.grasp_data is None:
        grasp_data_path = f"{config.grasp_file(exp_config)}.npy"
    else:
        assert os.path.exists(
            exp_config.grasp_data
        ), f"{exp_config.grasp_data} does not exist"
        grasp_data_path = exp_config.grasp_data
    grasps = np.load(f"{grasp_data_path}")

    grasp_idx = exp_config.grasp_idx if exp_config.grasp_idx else 0
    # if grasp_idx are start, end indices
    if isinstance(grasp_idx, tuple):
        grasp_idx = grasp_idx[0]
    elif isinstance(exp_config.grasp_idx, str):
        grasp_idx = tuple([int(x) for x in exp_config.grasp_idx.split(" ")])[0]

    grasp_vars = (grasps[grasp_idx, :, :3], grasps[grasp_idx, :, 3:])
    env = FingertipEnv(exp_config, grasp_vars)
    print("OBJECT MASS:", env.obj.mass)

    # Evaluates sampled grasps
    successes = 0
    if exp_config.grasp_idx is None:
        grasp_ids = range(len(grasps))
    elif isinstance(exp_config.grasp_idx, tuple):
        grasp_ids = np.arange(exp_config.grasp_idx[0], exp_config.grasp_idx[1])
    elif isinstance(exp_config.grasp_idx, str):
        idx_range = [int(x) for x in exp_config.grasp_idx.split(" ")]
        assert len(idx_range) == 2, "grasp-idx can be at most 2 numbers"
        grasp_ids = np.arange(idx_range[0], idx_range[1])
    else:
        grasp_ids = [exp_config.grasp_idx]

    n_grasps = len(grasp_ids)
    for i, grasp_idx in enumerate(grasp_ids):
        grasp_points = torch.tensor(grasps[grasp_idx, :, :3], dtype=torch.float32)
        grasp_normals = torch.tensor(grasps[grasp_idx, :, 3:], dtype=torch.float32)
        grasp_vars = (grasp_points, grasp_normals)

        print(f"EVALUATING GRASP from {grasp_data_path} {grasp_idx}: {grasp_points}")
        try:
            success = lifting_trajectory(env, grasp_vars, grasp_idx)
        except KeyboardInterrupt:
            success = False
            pdb.set_trace()
        successes += success
        if success:
            print(f"SUCCESS! grasp {grasp_idx}")
        if torch.isnan(env.robot.position).any():
            print("exiting early, robot position is nan, resets do not work")
            # set n_grasps to be number of grasps evaluated so far
            n_grasps = i + 1
            break

    success_pct = successes / n_grasps * 100
    if exp_config.wandb:
        wandb.run.summary["success_pct"] = success_pct
        wandb.finish()
    print(f"Percent successes: {success_pct}% out of {n_grasps}")


if __name__ == "__main__":
    main()
