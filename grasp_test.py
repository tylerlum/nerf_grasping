from typing import Tuple, Any
from isaacgym import gymapi, gymtorch
from nerf_grasping.grasp_opt import grasp_matrix, rot_from_vec
from nerf_grasping.control import pos_control, force_opt
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
from pathlib import Path

root_dir = os.path.abspath("./")
asset_dir = f"{root_dir}/assets"

def setup_viewer():
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    # position outside stage
    cam_pos = gymapi.Vec3(0.7, 0.175, 0.6)
    # position above banana
    cam_pos = gymapi.Vec3(0.1, 0.02, 0.4)
    cam_target = gymapi.Vec3(0, 0, 0.2)
    gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)
    return viewer


def step_gym():
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    if viewer is not None:
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
    refresh_tensors()


def setup_env():
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
    gym.add_ground(sim, plane_params)

    spacing = 1.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, env_lower, env_upper, 0)
    return env


def refresh_tensors():
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)


def setup_sim():
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
    gym.set_light_parameters(sim, 2, intensity, ambient,
                             gymapi.Vec3(0.5, -1, 1))
    gym.set_light_parameters(sim, 3, intensity, ambient, gymapi.Vec3(0, 0, 1))
    return sim


def setup_stage(env):
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
    gym.create_actor(env,
                     stage_asset,
                     gymapi.Transform(),
                     "Stage",
                     0,
                     0,
                     segmentationId=1)


def get_mesh_contacts(gt_mesh,
                      grasp_points,
                      pos_offset=None,
                      rot_offset=None,
                      return_dist=False):
    if pos_offset is not None:
        # project grasp_points into object frame
        grasp_points -= pos_offset
        grasp_points = np.stack([rot_offset.rotate(gp) for gp in grasp_points])
    points, distance, index = trimesh.proximity.closest_point(
        gt_mesh, grasp_points)
    # grasp normals follow convention that points into surface,
    # trimesh computes normals pointing out of surface
    grasp_normals = -gt_mesh.face_normals[index]
    if pos_offset is not None:
        # project back into world frame
        points += pos_offset
        grasp_normals = np.stack(
            [rot_offset.T.rotate(x) for x in grasp_normals])
    retval = ((points, grasp_normals) if not return_dist else
              (points, grasp_normals, distance))
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
        result = a[i] + torch.dot(ap[i], ab[i]) / torch.dot(ab[i],
                                                            ab[i]) * ab[i]
        res.append(result)
    return res

def setup_viewer():
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    # position outside stage
    cam_pos = gymapi.Vec3(0.7, 0.175, 0.6)
    # position above banana
    cam_pos = gymapi.Vec3(0.1, 0.02, 0.4)
    cam_target = gymapi.Vec3(0, 0, 0.2)
    gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)
    return viewer


def step_gym():
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    if viewer is not None:
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
    refresh_tensors()


def setup_env():
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
    gym.add_ground(sim, plane_params)

    spacing = 1.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, env_lower, env_upper, 0)
    return env


def refresh_tensors():
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)


def setup_sim():
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
    gym.set_light_parameters(sim, 2, intensity, ambient,
                             gymapi.Vec3(0.5, -1, 1))
    gym.set_light_parameters(sim, 3, intensity, ambient, gymapi.Vec3(0, 0, 1))
    return sim


def setup_stage(env):
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
    gym.create_actor(env,
                     stage_asset,
                     gymapi.Transform(),
                     "Stage",
                     0,
                     0,
                     segmentationId=1)


def get_mesh_contacts(gt_mesh,
                      grasp_points,
                      pos_offset=None,
                      rot_offset=None,
                      return_dist=False):
    if pos_offset is not None:
        # project grasp_points into object frame
        grasp_points -= pos_offset
        grasp_points = np.stack([rot_offset.rotate(gp) for gp in grasp_points])
    points, distance, index = trimesh.proximity.closest_point(
        gt_mesh, grasp_points)
    # grasp normals follow convention that points into surface,
    # trimesh computes normals pointing out of surface
    grasp_normals = -gt_mesh.face_normals[index]
    if pos_offset is not None:
        # project back into world frame
        points += pos_offset
        grasp_normals = np.stack(
            [rot_offset.T.rotate(x) for x in grasp_normals])
    retval = ((points, grasp_normals) if not return_dist else
              (points, grasp_normals, distance))
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
        result = a[i] + torch.dot(ap[i], ab[i]) / torch.dot(ab[i],
                                                            ab[i]) * ab[i]
        res.append(result)
    return res

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

    rotations = np.stack([local_x, local_y, normals[..., None]], axis=-1)[...,
                                                                          0, :]
    return rotations


def calculate_grip_forces(positions,
                          normals,
                          target_force,
                          target_torque,
                          target_normal=0.4,
                          mu=0.1):
    """positions are relative to object CG if we want unbalanced torques"""

    torch_input = type(positions) == torch.Tensor
    if torch_input:
        assert type(
            normals) == torch.Tensor, "numpy vs torch needs to be consistant"
        assert type(target_force
                   ) == torch.Tensor, "numpy vs torch needs to be consistant"
        assert (type(target_torque) == torch.Tensor
               ), "numpy vs torch needs to be consistant"
        positions = positions.numpy()
        normals = normals.numpy()
        target_force = target_force.numpy()
        target_torque = target_torque.numpy()

    n, _ = positions.shape
    assert normals.shape == (n, 3)
    assert target_force.shape == (3,)

    F = cp.Variable((n, 3))
    constraints = []

    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    total_force = np.zeros((3))
    total_torque = np.zeros((3))

    Q = []
    for pos, norm, f in zip(positions, normals, F):
        q = example_rotation_transform(norm)
        Q.append(q)

        total_force += q @ f
        total_torque += skew_matrix(pos) @ q @ f

    constraints.append(total_force == target_force)
    constraints.append(total_torque == target_torque)

    friction_cone = cp.norm(F[:, :2], axis=1) <= mu * F[:, 2]
    constraints.append(friction_cone)

    force_magnitudes = cp.norm(F - np.array([[0.0, 0.0, target_normal]]),
                               axis=1)
    # friction_magnitudes = cp.norm(F[:,2], axis=1)
    prob = cp.Problem(cp.Minimize(cp.max(force_magnitudes)), constraints)
    prob.solve()

    if F.value is None:
        print("Failed to solve!")
        return torch.zeros(3,3)

    global_forces = np.zeros_like(F.value)
    for i in range(n):
        global_forces[i, :] = Q[i] @ F.value[i, :]

    if torch_input:
        global_forces = torch.tensor(global_forces).float()

    return global_forces

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from diffcp import SolverError


class ForceOptProblem:

    def __init__(
        self,
        obj_mu=1.0,
        mass=0.0166,
        target_n=1.0,
        cone_approx=False,
        object_frame=False,
    ):
        self.obj_mu = obj_mu
        self.mass = mass
        self.target_n = target_n  # 1.0
        self.cone_approx = cone_approx
        self.object_frame = object_frame
        self.setup_cvxpy_layer(target_n, obj_mu, mass)

    def setup_cvxpy_layer(self, target_n=1.0, obj_mu=1.0, mass=None):
        # Try solving optimization problem
        # contact force decision variable
        target_n_t = torch.as_tensor(np.array([0, 0, target_n] * 3),
                                     dtype=torch.float32)
        target_n_cp = cp.Parameter((9,),
                                   name="target_n",
                                   value=target_n_t.data.numpy())
        L = cp.Variable(9, name="l")
        W = cp.Parameter((6,), name="w_des")
        G = cp.Parameter((6, 9), name="grasp_m")
        cm = np.vstack((np.eye(3), np.zeros((3, 3)))) * mass

        inputs = [G, W, target_n_cp]
        outputs = [L]
        # self.Cm = cp.Parameter((6, 3), value=cm*self.mass, name='com')

        f_g = np.array([0, 0, -9.81])
        if self.object_frame:
            R_w_2_o = cp.Parameter((6, 6), name="r_w_2_o")
            w_ext = W + R_w_2_o @ cm @ f_g
            inputs.append(R_w_2_o)
        else:
            w_ext = W + cm @ f_g

        f = G @ L - w_ext  # generated contact forces must balance wrench

        # Objective function - minimize force magnitudes
        contact_force = L - target_n_cp
        cost = cp.sum_squares(contact_force)

        # Friction cone constraints; >= 0
        constraints = []
        cone_constraints = []
        if self.cone_approx:
            cone_constraints += [cp.abs(L[1::3]) <= self.obj_mu * L[::3]]
            cone_constraints += [cp.abs(L[2::3]) <= self.obj_mu * L[::3]]
        else:
            cone_constraints.append(
                cp.SOC(self.obj_mu * L[::3], (L[2::3] + L[1::3])[None]))
        constraints.append(f == np.zeros(f.shape))

        self.prob = cp.Problem(cp.Minimize(cost),
                               cone_constraints + constraints)
        self.policy = CvxpyLayer(self.prob, inputs, outputs)

    def balance_force_test(self, des_wrench, balance_force, grasp_points,
                           normals, obj_orientation):
        if self.object_frame:
            R_w_2_o = self.get_w2o_rot(obj_orientation)
            weight = (R_w_2_o @ np.vstack(
                [np.eye(3) * self.mass, np.zeros(
                    (3, 3))]) @ np.array([0, 0, -self.gravity]))
        else:
            weight = np.vstack([np.eye(3), np.zeros(
                (3, 3))]) @ np.array([0, 0, -self.gravity * self.mass])
        G = grasp_matrix(grasp_points, normals)
        w_ext = des_wrench + weight
        f = G @ balance_force - w_ext
        return f

    def run_fop(self, des_wrench, grasp_points, normals, obj_orientation=None):
        G_t = grasp_matrix(
            grasp_points.unsqueeze(0).cpu(),
            normals.unsqueeze(0).cpu())
        des_wrench_t = torch.as_tensor(des_wrench, dtype=torch.float32)
        target_n_t = torch.as_tensor(np.array([0, 0, self.target_n] * 3),
                                     dtype=torch.float32)
        inputs = [G_t, des_wrench_t, target_n_t]
        if self.object_frame:
            assert (obj_orientation is not None
                   ), "fop requires obj_orientation arg if using object frame"
            R_w_2_o = self.get_w2o_rot(obj_orientation)
            R_w_2_o_t = torch.as_tensor(R_w_2_o, dtype=torch.float32)
            inputs.append(R_w_2_o_t)
        try:
            (balance_force,) = self.policy(*inputs)
            return balance_force
        except SolverError:
            return torch.zeros((3, 3), dtype=torch.float32)

    def __call__(self, des_wrench, grasp_points, normals, obj_orientation=None):
        return self.run_fop(des_wrench, grasp_points, normals, obj_orientation)

    @staticmethod
    def get_w2o_rot(obj_orientation):
        R_w_2_o = Rotation.from_quat(obj_orientation).as_matrix().T
        R_w_2_o = block_diag(R_w_2_o, R_w_2_o)
        return R_w_2_o

def object_pos_control(
    obj,
    in_normal,
    target_position=None,
    target_normal=2.5,
    kp=0.1,
    kd=0.1,
    kp_angle=0.04,
    kd_angle=0.001,
    return_wrench=False,
):
    """Object position control for lifting trajectory"""
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
    target_torque = (-kp_angle * (quat @ target_quat.T).to_tangent_space() -
                     kd_angle * angular_vel)
    if return_wrench:
        return torch.cat([target_force, target_torque])
    # grasp points in object frame
    # TODO: compute tip radius here?
    grasp_points = tip_position - cg_pos
    in_normal = torch.stack([quat.rotate(x) for x in in_normal], axis=0)
    try:
        global_forces = calculate_grip_forces(
            grasp_points,
            in_normal,
            target_force,
            target_torque,
            target_normal,
            obj.mu,
        )
    except AssertionError:
        logging.warning("solve failed, maintaining previous forces")
        global_forces = (robot.previous_global_forces
                        )  # will fail if we failed solve on first iteration
        assert global_forces is not None
    else:
        robot.previous_global_forces = global_forces

    print(global_forces.shape)

    return global_forces, target_force, target_torque

gym = gymapi.acquire_gym()

sim = setup_sim()
env = setup_env()
setup_stage(env)
viewer = setup_viewer()

Obj = ig_objects.Banana
grasp_points, grasp_normals = Obj.grasp_points, Obj.grasp_normals

grasp_normals = grasp_normals / grasp_normals.norm(dim=1, keepdim=True)
grasp_vars = (grasp_points, grasp_normals)

# Creates the robot, fop objective, and object
robot = FingertipRobot(gym, sim, env, grasp_vars=grasp_vars)
obj = Obj(gym, sim, env)
fop_obj = ForceOptProblem(mass=obj.mass, target_n=0.3)

robot.setup_tensors()
obj.setup_tensors()
obj.load_nerf_model()
obj.load_trimesh()
for i in range(4):
    step_gym()

grasps = np.load("grasp_data/banana_50.npy")
grasp_idx = np.random.choice(np.arange(grasps.shape[0]))

grasp_points = torch.tensor(grasps[grasp_idx, :, :3], dtype=torch.float32)
grasp_normals = torch.tensor(grasps[grasp_idx, :, 3:], dtype=torch.float32)
grasp_vars = (grasp_points, grasp_normals)

print(grasp_points)

robot.reset_actor(grasp_vars)
obj.reset_actor()
for i in range(4):
    step_gym()
robot.reset_actor(grasp_vars)
obj.reset_actor()
for i in range(50):
    step_gym()

states = []
f_lift = None
_net_cf = gym.acquire_net_contact_force_tensor(sim)
net_cf = gymtorch.wrap_tensor(_net_cf)

fop_obj.target_n = 0.1

for timestep in range(750):
    # time.sleep(0.01)
    step_gym()
    # finds the closest contact points to the original grasp normal + grasp_point ray
    closest_points = ig_utils.closest_point(grasp_points,
                                            grasp_points + grasp_normals,
                                            robot.position)
    #closest_points[:, 2] = obj.position[2] #+ 0.005
    if timestep < 50:
        mode = "reach"
        f = robot.position_control(grasp_points)
        pos_err = robot.position - grasp_points
    elif timestep < 150:
        mode = "grasp"
        pos_err = closest_points - robot.position
        pos_control = pos_err * 5
        vel_control = - 0.25 * robot.velocity
        f = torch.tensor(grasp_normals * 0.1) + pos_control + vel_control
    else:
        mode = "lift"
        pos_err = closest_points - robot.position
        height_err = robot.target_height - obj.position[-1]

        # f, target_force, target_torque = object_pos_control(
        #     obj, grasp_normals, target_normal=0.15
        gp, ge = get_mesh_contacts(obj.gt_mesh, closest_points)
        ge = torch.tensor(ge, dtype=torch.float32)
        f_lift, target_force, target_torque = object_pos_control(
            obj, ge, target_normal=1., kp=1.5, kd=1., kp_angle=0.1, kd_angle=1e-2)
        # des_wrench = torch.cat(object_pos_control(obj, ge)[1:])
        # des_wrench = torch.tensor([0, 0, 0.1, 0, 0, 0])
        # f_lift = fop_obj(des_wrench, closest_points, ge)
        f = f_lift  # + pos_err * 3

    gym.refresh_net_contact_force_tensor(sim)
    robot.apply_fingertip_forces(f)
    if timestep >= 100 and timestep % 50 == 0:
        print("MODE:", mode)
        print("TIMESTEP:", timestep)
        print("POSITION ERR:", pos_err)
        print("VELOCITY:", robot.velocity)
        print("FORCE MAG:", f.norm())
        print("Z Force:", f[:, 2])
        print("OBJECT FORCES:", gym.get_rigid_contact_forces(sim)[obj.actor])
        # print(f"NET CONTACT FORCE:", net_cf[obj.index,:])
    if (robot.position[:, -1] >= 0.5).any():
        break
    state = dict(pos_err=pos_err,
                 velocity=robot.velocity,
                 force_mag=f.norm(dim=1))