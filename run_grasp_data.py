from typing import Tuple, Any
from isaacgym import gymapi, gymtorch
from nerf_grasping.grasp_opt import grasp_matrix, rot_from_vec
from nerf_grasping.control import force_opt
from nerf_grasping import grasp_utils
from nerf_grasping.quaternions import Quaternion
from nerf_grasping.sim import ig_utils
from nerf_grasping.sim import ig_objects
from nerf_grasping.sim import ig_viz_utils
from nerf_grasping.sim.ig_robot import FingertipRobot

import time
import torch
import numpy as np
import trimesh

import os

root_dir = os.path.abspath("./")
asset_dir = f"{root_dir}/assets"


def get_mesh_contacts(
    gt_mesh, grasp_points, pos_offset=None, rot_offset=None, return_dist=False
):
    if pos_offset is not None:
        rot_offset = Quaternion.fromWLast(rot_offset)
        # project grasp_points into object frame
        grasp_points -= pos_offset.numpy()
        grasp_points = np.stack([rot_offset.rotate(gp) for gp in grasp_points])
    points, distance, index = trimesh.proximity.closest_point(gt_mesh, grasp_points)
    # grasp normals follow convention that points into surface,
    # trimesh computes normals pointing out of surface
    grasp_normals = -gt_mesh.face_normals[index]
    if pos_offset is not None:
        # project back into world frame
        points += pos_offset.numpy()
        grasp_normals = np.stack([rot_offset.T.rotate(x) for x in grasp_normals])
    retval = (
        (points, grasp_normals)
        if not return_dist
        else (points, grasp_normals, distance)
    )
    return retval


def random_forces(timestep):
    fx = -np.sin(timestep * np.pi / 10) * 0.025 + 0.001
    fy = -np.sin(timestep * np.pi / 5) * 0.025 + 0.001
    f = np.array([[fx, fy, 0.0]] * 3)
    return f


def double_reset(gym, sim, viewer, robot, obj, grasp_vars):
    robot.reset_actor(grasp_vars)
    obj.reset_actor()
    for i in range(4):
        ig_utils.step_gym(gym, sim, viewer)
    robot.reset_actor(grasp_vars)
    obj.reset_actor()
    for i in range(50):
        ig_utils.step_gym(gym, sim, viewer)


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


def object_pos_control(
    robot,
    obj,
    in_normal,
    target_position=None,
    target_normal=0.5,
    kp=10.0,
    kd=0.1,
    kp_angle=0.04,
    kd_angle=0.001,
    return_wrench=False,
):
    """Object position control for lifting trajectory"""
    success = True
    if target_position is None:
        target_position = np.array([0.0, 0.0, robot.target_height])
    tip_position = robot.position
    vel = obj.velocity
    angular_vel = obj.angular_velocity
    quat = Quaternion.fromWLast(obj.orientation)
    target_quat = Quaternion.Identity()
    cg_pos = obj.get_CG()  # thoughts it eliminates the pendulum effect? possibly?

    pos_error = cg_pos - target_position
    object_weight_comp = obj.mass * 9.8 * torch.tensor([0, 0, 1])
    # target_force = object_weight_comp - 0.9 * pos_error - 0.4 * vel
    # banana tuning
    target_force = object_weight_comp - kp * pos_error - kd * vel
    # target_force = -kp * pos_error - kd * vel
    target_torque = (
        -kp_angle * (quat @ target_quat.T).to_tangent_space() - kd_angle * angular_vel
    )
    if return_wrench:
        return torch.cat([target_force, target_torque])
    # grasp points in object frame
    # TODO: compute tip radius here?
    grasp_points = tip_position - cg_pos
    in_normal = torch.stack([quat.rotate(x) for x in in_normal], axis=0)
    global_forces, success = force_opt.calculate_grip_forces(
        grasp_points,
        in_normal,
        target_force,
        target_torque,
        target_normal,
        obj.mu,
    )
    if not success:
        # logging.warning("solve failed, maintaining previous forces")
        global_forces = (
            robot.previous_global_forces
        )  # will fail if we failed solve on first iteration
        assert global_forces is not None
    else:
        robot.previous_global_forces = global_forces

    # print(global_forces.shape)

    return global_forces, target_force, target_torque, success


def lifting_trajectory(
    gym,
    sim,
    viewer,
    robot,
    obj,
    grasp_vars,
    loaded_mesh=None,
    pos_control_kwargs={},
    obj_pos_control_kwargs={},
):
    if loaded_mesh is None:
        if obj.gt_mesh is None:
            obj.load_trimesh()
        mesh = obj.gt_mesh
    else:
        mesh = loaded_mesh
    double_reset(gym, sim, viewer, robot, obj, grasp_vars)
    grasp_points, grasp_normals = grasp_vars

    f_lift = None
    start_timestep = 0
    ge = None
    fail_count = 0

    for timestep in range(500):
        height_err = 0.04 - obj.position[-1].cpu().numpy().item()
        # time.sleep(0.01)
        ig_utils.step_gym(gym, sim, viewer)
        # finds the closest contact points to the original grasp normal + grasp_point ray
        closest_points = ig_utils.closest_point(
            grasp_points, grasp_points + grasp_normals, robot.position
        )

        # compute potential to closest points
        potential = compute_potential(grasp_points)
        if timestep < 50:
            mode = "reach"
            f = robot.position_control(grasp_points)
            pos_err = robot.position - grasp_points
        elif timestep < 150:
            mode = "grasp"
            pos_err = closest_points - robot.position
            pos_control = pos_err * 5
            vel_control = -10.0 * robot.velocity
            f = torch.tensor(grasp_normals * 0.0001) + pos_control + vel_control
        else:
            mode = "lift"
            closest_points[:, 2] = obj.position[2] + 0.005
            pos_err = closest_points - robot.position
            # f, target_force, target_torque, success = object_pos_control(
            #     obj, grasp_normals, target_normal=0.15
            if mesh is None:
                if ge is None or timestep < 130:
                    ge = robot.get_grad_ests(obj, robot.position)
            else:
                gp, ge = get_mesh_contacts(
                    mesh,
                    robot.position,
                    pos_offset=obj.position,
                    rot_offset=obj.orientation,
                )
                ge = torch.tensor(ge, dtype=torch.float32)
                if loaded_mesh:
                    ge = -ge
            f_lift, target_force, target_torque, success = object_pos_control(
                robot,
                obj,
                ge,
                target_normal=3.0,
                kp=1.5,
                kd=1.0,
                kp_angle=0.3,
                kd_angle=1e-2,
            )
            f = f_lift

            if not success:
                fail_count += 1

        # if f.norm() > 3:
        #     break
        gym.refresh_net_contact_force_tensor(sim)
        robot.apply_fingertip_forces(f)
        if timestep >= 100 and timestep % 50 == 0:
            print("MODE:", mode)
            print("TIMESTEP:", timestep)
            print("POSITION ERR:", pos_err)
            print("POTENTIAL:", potential)
            print("VELOCITY:", robot.velocity)
            print("FORCE MAG:", f.norm())
            # print("Z Force:", f[:, 2])
            # print("OBJECT FORCES:", gym.get_rigid_contact_forces(sim)[obj.actor])
            if mode == "lift":
                print("HEIGHT_ERR:", height_err)
            # print(f"NET CONTACT FORCE:", net_cf[obj.index,:])
        if (robot.position[:, -1] <= 0.01).any():
            print("Finger too low!")
            return False
        if (robot.position[:, -1] >= 0.5).any():
            print("Finger too high!")
            return False
        if fail_count > 10:
            print("Too many cvx failures!")
            return False
        # if number of timesteps of grasp success exceeds 3 seconds
        succ_timesteps = 180
        err_bound = 0.003
        if timestep - start_timestep >= succ_timesteps:
            return True

        # increment start_timestep if height error outside err_bound
        if height_err > err_bound:
            start_timestep = timestep
    return height_err <= err_bound


def main(
    visualization=True,
    obj_name="teddy_bear",
    grasp_data="grasp_data/teddy_bear_10.npy",
    grasp_idx=None,
    use_gt_mesh=False,
):
    gym = gymapi.acquire_gym()

    sim = ig_utils.setup_sim(gym)
    env = ig_utils.setup_env(gym, sim)
    ig_utils.setup_stage(gym, sim, env)
    viewer = ig_utils.setup_viewer(gym, sim, env) if visualization else None
    if obj_name == "teddy_bear":
        Obj = ig_objects.TeddyBear
    elif obj_name == "banana":
        Obj = ig_objects.Banana
    elif obj_name == "powerdrill":
        Obj = ig_objects.PowerDrill
    else:
        print("invalid obj_name:", obj_name)

    grasp_points, grasp_normals = Obj.grasp_points, Obj.grasp_normals

    grasp_normals = grasp_normals / grasp_normals.norm(dim=1, keepdim=True)
    grasp_vars = (grasp_points, grasp_normals)

    # Creates the robot, fop objective, and object
    robot = FingertipRobot(
        gym, sim, env, grasp_vars=grasp_vars, use_grad_est=True, norm_start_offset=0.05
    )
    obj = Obj(gym, sim, env)

    robot.setup_tensors()
    obj.setup_tensors()
    obj.load_nerf_model()
    obj.load_trimesh()
    for i in range(4):
        ig_utils.step_gym(gym, sim, viewer)

    # grasp_data = "grasp_data/banana_nerf10psv-rs10.npy"
    nerf = "nerf" in grasp_data
    mesh = None
    if not nerf:
        mesh_name = grasp_data.split("/")[1].rstrip(".npy")
        if mesh_name != "teddy_bear":
            mesh_name = mesh_name.split("_")[0]
        if "_" in mesh_name and not use_gt_mesh:
            mesh = trimesh.load(f"grasp_data/meshes/{mesh_name}.obj", force="mesh")
    grasps = np.load(grasp_data)
    successes = 0

    if grasp_idx is not None:
        grasps = grasps[grasp_idx][None]
    for grasp_id in range(len(grasps)):
        if grasp_idx is None or grasp_idx == grasp_id - 1:
            grasp_idx = grasp_id
        grasp_points = torch.tensor(grasps[grasp_idx, :, :3], dtype=torch.float32)
        grasp_normals = torch.tensor(grasps[grasp_idx, :, 3:], dtype=torch.float32)
        grasp_vars = (grasp_points, grasp_normals)

        print(f"EVALUATING GRASP from {grasp_data} {grasp_idx}: {grasp_points}")
        print(grasp_points, i)
        success = lifting_trajectory(
            gym,
            sim,
            viewer,
            robot,
            obj,
            grasp_vars,
            loaded_mesh=mesh,
        )
        successes += success
        if success:
            print(f"SUCCESS! grasp {grasp_idx}")

    print(f"Percent successes: {successes / len(grasps) * 100}% out of {len(grasps)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_name", help="object name", default="banana")
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--use_gt_mesh", action="store_true")
    parser.add_argument(
        "--grasp_data", help="path to generated grasp data file", type=str
    )
    parser.add_argument(
        "--grasp_idx",
        help="index of grasp from grasp_data to isolate a single grasp",
        type=int,
    )

    args = parser.parse_args()
    main(
        args.visualization,
        args.obj_name,
        args.grasp_data,
        args.grasp_idx,
        args.use_gt_mesh,
    )
