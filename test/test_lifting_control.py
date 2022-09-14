import dcargs
import dataclasses
import wandb
import numpy as np
import os
import pdb
from nerf_grasping.sim.sim_fingertip import FingertipEnv
from nerf_grasping.sim import ig_utils
from nerf_grasping import config
import torch


def get_mode(timestep):
    if timestep % 1000 < 50:
        return "reach"
    if timestep % 1000 < 150:
        return "grasp"
    if timestep % 1000 < 1000:
        return "lift"


def lifting_trajectory(env, grasp_vars):
    """Evaluates a lifting trajectory for a sampled grasp"""
    # double_reset(env, grasp_vars)
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
        # potential = compute_potential(grasp_points)
        state = env.run_control(mode, grasp_vars)
        if state["grasp_opt_success"]:
            fail_count = 0
        else:
            fail_count += 1

        if timestep >= 100 and (timestep + 1) % 50 == 0:
            print("MODE:", state["mode"])
            print("TIMESTEP:", timestep)
            print("POSITION ERR:", state["ftip_pos_err"])
            # print("POTENTIAL:", potential)
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
        # max_vel = env.robot.velocity.max(dim=1)
        # scalar values
        state["height_err"] = height_err
        state["min_ftip_height"] = env.robot.position[:, -1].min()
        state["max_ftip_height"] = env.robot.position[:, -1].max()
        state["fail_count"] = fail_count

        log = {}
        for k, v in state.items():
            if isinstance(v, float):
                log[k] = v
            elif isinstance(v, (torch.Tensor, np.ndarray)) and np.prod(v.shape) == 1:
                log[k] = float(v)

        if wandb.run is not None:
            wandb.log(state, step=timestep)

        if (env.robot.position[:, -1] <= 0.01).any():
            print("Finger too low!")
            return False
        elif (env.robot.position[:, -1] >= 0.5).any():
            print(f"Finger too high!: {env.robot.position[:,-1]}")
            return False
        elif torch.isnan(env.robot.position).any():
            print("Finger position is nan!")
            return False
        if fail_count > 50:
            print("TIMESTEP:", timestep)
            print("Too many cvx failures!")
            return False
        # if number of timesteps of grasp success exceeds 3 seconds
        err_bound = 0.03
        if timestep - start_succ >= 180:
            return True

        # increment start_succ if height error outside err_bound
        if height_err > err_bound:
            start_succ = timestep
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

    assert exp_config.grasp_idx is not None
    grasp_idx = exp_config.grasp_idx

    grasp_vars = (grasps[grasp_idx, :, :3], grasps[grasp_idx, :, 3:])
    env = FingertipEnv(exp_config, grasp_vars)
    print("OBJECT MASS:", env.obj.mass)

    # Evaluates sampled grasps
    grasp_points = torch.tensor(grasps[grasp_idx, :, :3], dtype=torch.float32)
    grasp_normals = torch.tensor(grasps[grasp_idx, :, 3:], dtype=torch.float32)
    grasp_vars = (grasp_points, grasp_normals)

    print(f"EVALUATING GRASP from {grasp_data_path} {grasp_idx}: {grasp_points}")
    try:
        success = lifting_trajectory(env, grasp_vars)
    except KeyboardInterrupt:
        success = False
        pdb.set_trace()
    if success:
        print(f"SUCCESS! grasp {grasp_idx}")
    if torch.isnan(env.robot.position).any():
        print("exiting early, robot position is nan, resets do not work")

    if exp_config.wandb:
        wandb.run.summary["success_pct"] = float(success)
        wandb.finish()

    if exp_config.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
