from typing import Tuple, Any
from isaacgym import gymapi, gymtorch
from nerf_grasping import config
from nerf_grasping.grasp_opt import grasp_matrix, rot_from_vec
from nerf_grasping.control import force_opt
from nerf_grasping import grasp_utils
from nerf_grasping.quaternions import Quaternion
from nerf_grasping.sim import ig_utils
from nerf_grasping.sim import ig_objects
from nerf_grasping.sim import ig_viz_utils
from nerf_grasping.sim.ig_robot import FingertipRobot

import dcargs
import time
import torch
import numpy as np
import trimesh

import os

root_dir = os.path.abspath("./")
asset_dir = f"{root_dir}/assets"


def refresh_tensors(gym, sim):
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)


def step_gym(gym, sim, viewer=None):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    if viewer is not None:
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
    refresh_tensors(gym, sim)


def double_reset(robot, obj, grasp_vars, viewer=None):
    print(f"robot position before reset: {robot.position}")
    robot.reset_actor(grasp_vars)
    obj.reset_actor()
    for i in range(4):
        step_gym(robot.gym, robot.sim, viewer)
    robot.reset_actor(grasp_vars)
    obj.reset_actor()
    for i in range(50):
        step_gym(robot.gym, robot.sim, viewer)
    print(f"robot position after reset: {robot.position}")


def setup_env(gym, sim):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
    gym.add_ground(sim, plane_params)

    spacing = 1.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, env_lower, env_upper, 0)
    return env


def setup_sim(gym):
    args = ig_utils.parse_arguments(description="Trifinger test")
    # only tested with this one
    assert args.physics_engine == gymapi.SIM_PHYSX

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0

    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    # sim_params.physx.use_gpu = True

    # sim_params.use_gpu_pipeline = True
    sim_params.use_gpu_pipeline = False
    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        args.physics_engine,
        sim_params,
    )
    assert sim is not None

    # intensity = 0.01 # for nerf generation
    # ambient = 0.21 / intensity
    intensity = 0.5
    ambient = 0.10 / intensity
    intensity = gymapi.Vec3(intensity, intensity, intensity)
    ambient = gymapi.Vec3(ambient, ambient, ambient)

    gym.set_light_parameters(sim, 0, intensity, ambient, gymapi.Vec3(0.5, 1, 1))
    gym.set_light_parameters(sim, 1, intensity, ambient, gymapi.Vec3(1, 0, 1))
    gym.set_light_parameters(sim, 2, intensity, ambient, gymapi.Vec3(0.5, -1, 1))
    gym.set_light_parameters(sim, 3, intensity, ambient, gymapi.Vec3(0, 0, 1))
    return sim


def setup_stage(gym, sim, env):
    # this one is convex decomposed
    stage_urdf_file = "trifinger/robot_properties_fingers/urdf/high_table_boundary.urdf"
    # stage_urdf_file = "trifinger/robot_properties_fingers/urdf/trifinger_stage.urdf"
    # stage_urdf_file = "trifinger/robot_properties_fingers/urdf/stage.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.disable_gravity = True
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.use_mesh_materials = True
    asset_options.thickness = 0.001

    stage_asset = gym.load_asset(sim, asset_dir, stage_urdf_file, asset_options)
    gym.create_actor(
        env, stage_asset, gymapi.Transform(), "Stage", 0, 0, segmentationId=1
    )


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


def object_pos_control(
    robot,
    obj,
    in_normal,
    target_position=None,
    return_wrench=False,
):
    """Object position control for lifting trajectory"""

    target_normal = robot.controller_params.target_normal
    kp = robot.controller_params.kp
    kd = robot.controller_params.kd
    kp_angle = robot.controller_params.kp_angle
    kd_angle = robot.controller_params.kd_angle

    if target_position is None:
        target_position = np.array([0.0, 0.0, robot.target_height])
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
    grasp_points_of = robot.get_contact_points(in_normal) - cg_pos
    in_normal = torch.stack([quat.rotate(x) for x in in_normal], axis=0)
    global_forces, success = force_opt.calculate_grip_forces(
        grasp_points_of,
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


def lifting_trajectory(robot, obj, grasp_vars, mesh=None, viewer=None):
    double_reset(robot, obj, grasp_vars)
    grasp_points, grasp_normals = grasp_vars

    f_lift = None
    start_timestep = 0
    ge = None
    fail_count = 0
    gym, sim = robot.gym, robot.sim

    for timestep in range(1000):
        height_err = (
            robot.target_height
            - obj.position[-1].cpu().numpy().item()
            + obj.translation[-1]
        )
        # time.sleep(0.01)
        step_gym(gym, sim, viewer)
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
            vel_control = -1.0 * robot.velocity
            contact_pts = robot.get_contact_points(grasp_normals)
            f = torch.tensor(grasp_normals * 0.05) + pos_control + vel_control
        else:
            mode = "lift"
            closest_points[:, 2] = obj.position[2] + 0.005
            pos_err = closest_points - robot.position
            if mesh is None:
                if ge is None or timestep < 130:
                    ge = robot.get_grad_ests(obj, contact_pts)
            else:
                gp, ge = get_mesh_contacts(
                    mesh,
                    contact_pts,
                    pos_offset=obj.position,
                    rot_offset=obj.orientation,
                )
                ge = torch.tensor(ge, dtype=torch.float32)
            f_lift, target_force, target_torque, success = object_pos_control(
                robot,
                obj,
                ge,
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
        if (robot.position[:, -1] <= 0.005).any():
            print("Finger too low!")
            import pdb

            pdb.set_trace()
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


def main():
    exp_config = dcargs.cli(config.Experiment)
    gym = gymapi.acquire_gym()

    sim = setup_sim(gym)
    env = setup_env(gym, sim)
    setup_stage(gym, sim, env)
    viewer = setup_viewer(gym, sim, env) if exp_config.visualize else None

    # Loads grasp data
    grasp_data_path = config.grasp_file(exp_config)
    grasps = np.load(f"{grasp_data_path}.npy")

    # Creates the robot
    robot = FingertipRobot(exp_config.robot_config)
    robot.setup_gym(gym, sim, env, grasps[0, :, :3], grasps[0, :, 3:])

    # Creates object and loads nerf and object mesh
    obj = ig_objects.load_object(exp_config)
    obj.setup_gym(gym, sim, env)
    obj.load_trimesh()

    # setup tensors
    robot.setup_tensors()
    obj.setup_tensors()

    # Loads nerf or mesh
    if isinstance(exp_config.model_config, config.Nerf):
        obj.load_nerf_model()
        mesh = None
    else:
        mesh_path = config.mesh_file(exp_config)
        mesh = trimesh.load(mesh_path)

    # Evaluates sampled grasps
    successes = 0

    for grasp_idx in range(len(grasps)):
        grasp_points = torch.tensor(grasps[grasp_idx, :, :3], dtype=torch.float32)
        grasp_normals = torch.tensor(grasps[grasp_idx, :, 3:], dtype=torch.float32)
        grasp_vars = (grasp_points, grasp_normals)

        print(f"EVALUATING GRASP from {grasp_data_path} {grasp_idx}: {grasp_points}")
        print(grasp_points, grasp_idx)
        success = lifting_trajectory(robot, obj, grasp_vars, mesh=mesh, viewer=viewer)
        successes += success
        if success:
            print(f"SUCCESS! grasp {grasp_idx}")

    print(f"Percent successes: {successes / len(grasps) * 100}% out of {len(grasps)}")


if __name__ == "__main__":
    main()
