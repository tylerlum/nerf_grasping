import argparse
import os.path as osp

from pathlib import Path
from isaacgym import gymapi
from isaacgym.gymutil import parse_device_str

import numpy as np
import torch
import trimesh
from nerf_grasping import grasp_utils
from nerf_grasping.quaternions import Quaternion


def gymutil_parser(
    description="Isaac Gym Example",
    headless=False,
    no_graphics=False,
    custom_parameters=[],
):
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument(
            "--headless",
            action="store_true",
            help="Run headless without creating a viewer window",
        )
    if no_graphics:
        parser.add_argument(
            "--nographics",
            action="store_true",
            help="Disable graphics context creation, no viewer window is created, and no headless rendering is available",
        )
    parser.add_argument(
        "--sim_device",
        type=str,
        default="cuda:0",
        help="Physics Device in PyTorch-like syntax",
    )
    parser.add_argument(
        "--pipeline", type=str, default="gpu", help="Tensor API pipeline (cpu/gpu)"
    )
    parser.add_argument(
        "--graphics_device_id", type=int, default=0, help="Graphics Device ID"
    )

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument(
        "--flex", action="store_true", help="Use FleX for physics"
    )
    physics_group.add_argument(
        "--physx", action="store_true", help="Use PhysX for physics"
    )

    parser.add_argument(
        "--num_threads", type=int, default=0, help="Number of cores used by PhysX"
    )
    parser.add_argument(
        "--subscenes",
        type=int,
        default=0,
        help="Number of PhysX subscenes to simulate in parallel",
    )
    parser.add_argument(
        "--slices", type=int, help="Number of client threads that process env slices"
    )

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(
                        argument["name"],
                        type=argument["type"],
                        default=argument["default"],
                        help=help_str,
                    )
                else:
                    parser.add_argument(
                        argument["name"], type=argument["type"], help=help_str
                    )
            elif "action" in argument:
                parser.add_argument(
                    argument["name"], action=argument["action"], help=help_str
                )

        else:
            print()
            print(
                "ERROR: command line argument name, type/action must be defined, argument not added to parser"
            )
            print("supported keys: name, type, default, action, help")
            print()

    return parser


def parse_arguments(
    *args,
    description="Isaac Gym Example",
    headless=False,
    no_graphics=False,
    custom_parameters=[],
):
    parser = gymutil_parser(description, headless, no_graphics, custom_parameters)
    args = parser.parse_args(args=args)

    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    assert pipeline == "cpu" or pipeline in (
        "gpu",
        "cuda",
    ), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = pipeline in ("gpu", "cuda")

    if args.sim_device_type != "cuda" and args.flex:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = "cuda:0"
        args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)

    if args.sim_device_type != "cuda" and pipeline == "gpu":
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = "CPU"
        args.use_gpu_pipeline = False

    # Default to PhysX
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = args.sim_device_type == "cuda"

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    # Using --nographics implies --headless
    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes
    return args


def setup_viewer(gym, sim, env):
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    # position outside stage
    cam_pos = gymapi.Vec3(0.7, 0.175, 0.6)
    # position above banana
    cam_pos = gymapi.Vec3(0.1, 0.02, 0.4)
    cam_target = gymapi.Vec3(0, 0, 0.2)
    gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)
    return viewer


def step_gym(gym, sim, viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    if viewer is not None:
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
    refresh_tensors(gym, sim)


def setup_env(gym, sim):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
    gym.add_ground(sim, plane_params)

    spacing = 1.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, env_lower, env_upper, 0)
    return env


def refresh_tensors(gym, sim):
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)


def setup_sim(gym):
    args = parse_arguments(description="Trifinger test")
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
    assert sim is not None, f"{__file__}.setup_sim() failed"

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
    asset_dir = osp.join(Path(__file__).parents[2], "assets")
    # this one is convex decomposed
    stage_urdf_file = "trifinger/robot_properties_fingers/urdf/high_table_boundary.urdf"

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


def closest_point(a, b, p):
    ap = p - a
    ab = b - a
    res = []
    for i in range(3):
        result = a[i] + torch.dot(ap[i], ab[i]) / torch.dot(ab[i], ab[i]) * ab[i]
        res.append(result)
    return torch.stack(res)


def get_mesh_contacts(gt_mesh, grasp_points, pos_offset=None, rot_offset=None):

    # Transform query points into object frame (Z-up, centroid origin).
    rot_offset = Quaternion.fromWLast(rot_offset)
    if pos_offset is not None:
        # project grasp_points into object frame
        grasp_points -= pos_offset.cpu().numpy().reshape(1, 3)
        grasp_points = np.stack([rot_offset.T.rotate(gp) for gp in grasp_points])

    # Query trimesh for closest point.
    points, distance, index = trimesh.proximity.closest_point(gt_mesh, grasp_points)
    # grasp normals follow convention that points into surface,
    # trimesh computes normals pointing out of surface
    grasp_normals = -gt_mesh.face_normals[index]

    #     print('grasp normals, NeRF frame:', grasp_normals)
    #     print('grasp points, NeRF frame:', points)
    #     print('surface distances, NeRF frame:', distance)

    if pos_offset is not None:
        # project back into world frame
        points = np.stack([rot_offset.rotate(gp) for gp in points])
        points += pos_offset.cpu().numpy().reshape(1, 3)
        grasp_normals = np.stack([rot_offset.rotate(x) for x in grasp_normals])

    return points, grasp_normals, distance
