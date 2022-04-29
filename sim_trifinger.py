# isort:skip_file
import json
import logging
import shutil
from pathlib import Path
from unittest.mock import Mock

import cvxpy as cp
from isaacgym import gymapi, gymtorch
import lietorch
import numpy as np
import torch

# import mathutils
from PIL import Image

import grasp_opt
import grasp_utils
import gymutils
from nerf import utils
from quaternions import Quaternion

# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

import os.path as osp

root_dir = osp.dirname(osp.abspath(__file__))
asset_dir = f"{root_dir}/assets"


def get_fixed_camera_transfrom(gym, sim, env, camera):
    # currently x+ is pointing down camera view axis - other degree of freedom is messed up
    # output will have x+ be optical axis, y+ pointing left (looking down camera) and z+ pointing up
    t = gym.get_camera_transform(sim, env, camera)
    pos = torch.tensor([t.p.x, t.p.y, t.p.z])
    quat = Quaternion.fromWLast([t.r.x, t.r.y, t.r.z, t.r.w])

    x_axis = torch.tensor([1.0, 0, 0])
    y_axis = torch.tensor([0, 1.0, 0])
    z_axis = torch.tensor([0, 0, 1.0])

    optical_axis = quat.rotate(x_axis)
    side_left_axis = z_axis.cross(optical_axis)
    up_axis = optical_axis.cross(side_left_axis)

    optical_axis /= torch.norm(optical_axis)
    side_left_axis /= torch.norm(side_left_axis)
    up_axis /= torch.norm(up_axis)

    rot_matrix = torch.stack([optical_axis, side_left_axis, up_axis], dim=-1)
    fixed_quat = Quaternion.fromMatrix(rot_matrix)

    return pos, fixed_quat


# TODO move those two to a seperate file?


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


# TODO investiate why solves sometimes fail
def calculate_grip_forces(positions, normals, target_force, target_torque):
    """positions are relative to object CG if we want unbalanced torques"""
    mu = 0.5

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

    force_magnitudes = cp.norm(F, axis=1)
    # friction_magnitudes = cp.norm(F[:,2], axis=1)
    prob = cp.Problem(cp.Minimize(cp.max(force_magnitudes)), constraints)
    prob.solve()

    if F.value is None:
        logging.debug("Failed to solve!")
        logging.debug("F.value: %s", F.value)
        logging.debug("positions: %s", positions)
        logging.debug("normals: %s", normals)
        logging.debug("target_force: %s", target_force)
        logging.debug("target_torque: %s", target_torque)
        assert False

    global_forces = np.zeros_like(F.value)
    for i in range(n):
        global_forces[i, :] = Q[i] @ F.value[i, :]

    if torch_input:
        global_forces = torch.tensor(global_forces).float()

    return global_forces


class RigidObject:

    scale = 1.0
    bound = 3.0
    centroid = np.zeros((3, 1))
    use_centroid = False

    def __init__(self, gym, sim, env):
        self.gym = gym
        self.sim = sim
        self.env = env

        self.asset = self.create_asset()
        self.actor = self.configure_actor(gym, env)
        self.nerfs_loaded = False

    def get_CG(self):
        pos = self.rb_states[0, :3]
        quat = Quaternion.fromWLast(self.rb_states[0, 3:7])
        return quat.rotate(self.CG) + pos

    def setup_tensors(self):
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # (num_rigid_bodies, 13)
        rb_count = self.gym.get_actor_rigid_body_count(self.env, self.actor)
        rb_start_index = self.gym.get_actor_rigid_body_index(
            self.env, self.actor, 0, gymapi.DOMAIN_SIM
        )

        # TODO TEMP
        self.index = rb_start_index

        # NOTE: simple indexing will return a view of the data but advanced indexing will return a copy breaking the updateing
        self.rb_states = gymtorch.wrap_tensor(_rb_states)[
            rb_start_index : rb_start_index + rb_count, :
        ]

    def load_nerf_models(self):
        if self.nerfs_loaded:
            return
        parser = utils.get_config_parser()
        # TODO: hardcoded, replace with reference to dir corresponding to object

        args = parser.parse_args(
            [
                "--workspace",
                f"{root_dir}/torch-ngp/{self.workspace}",
                "--test",
                "--cuda_ray",
                "--bound",
                f"{self.bound}",
                "--scale",
                f"{self.scale}",
                "--mode",
                "blender",
                f"{root_dir}/torch-ngp",
            ]
        )
        self.model = grasp_utils.load_nerf(args)
        self.nerfs_loaded = True


class Box(RigidObject):
    name = "box"
    grasp_points = torch.tensor(
        [
            [
                0.0,
                0.05,
                0.05,
            ],
            [
                0.03,
                -0.05,
                0.05,
            ],
            [
                -0.03,
                -0.05,
                0.05,
            ],
        ]
    )

    grasp_normals = torch.tensor(
        [
            [
                0.0,
                -1.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
            ],
        ]
    )

    def create_asset(self):
        asset_options = gymapi.AssetOptions()

        asset_options.vhacd_enabled = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.density = 100

        asset_options.vhacd_params.mode = 0
        asset_options.vhacd_params.resolution = 300000
        asset_options.vhacd_params.max_convex_hulls = 10
        asset_options.vhacd_params.max_num_vertices_per_ch = 16

        asset = self.gym.create_box(self.sim, 0.1, 0.1, 0.1, asset_options)

        rs_props = self.gym.get_asset_rigid_shape_properties(asset)
        for p in rs_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.1
        self.gym.set_asset_rigid_shape_properties(asset, rs_props)

        return asset

    def configure_actor(self, gym, env):
        actor = self.gym.create_actor(
            env,
            self.asset,
            gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.101)),
            "box",
            0,
            0,
            segmentationId=2,
        )
        self.gym.set_rigid_body_color(
            self.env, actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.2, 0.3)
        )

        rigid_body_props = self.gym.get_actor_rigid_body_properties(self.env, actor)
        self.mass = sum(x.mass for x in rigid_body_props)
        com = rigid_body_props[0].com
        self.CG = torch.tensor([com.x, com.y, com.z])

        return actor


class TeddyBear(RigidObject):
    asset_file = "objects/urdf/teddy_bear.urdf"
    name = "teddy_bear"

    data_dir = f"{root_dir}/nerf_shared/data/isaac_teddy"
    config_path = f"{root_dir}/nerf_shared/configs/isaac_teddy.txt"
    centroid = np.array([-0.0001444, 0.00412231, 0.08663063])
    use_centroid = False

    grasp_points = torch.tensor(
        [[0.0350, 0.0580, 0.1010], [0.0000, -0.0480, 0.0830], [-0.0390, 0.0580, 0.1010]]
    )

    grasp_normals = torch.tensor(
        [
            [-0.0350, -0.0580, 0.0000],
            [0.0000, 1.0000, 0.0000],
            [0.0390, -0.0580, 0.0000],
        ]
    )

    def create_asset(self):
        asset_options = gymapi.AssetOptions()

        asset_options.vhacd_enabled = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.density = 100
        asset_options.override_inertia = True
        asset_options.override_com = True

        asset_options.vhacd_params.mode = 0
        asset_options.vhacd_params.resolution = 300000
        asset_options.vhacd_params.max_convex_hulls = 10
        asset_options.vhacd_params.max_num_vertices_per_ch = 16

        asset = self.gym.load_asset(self.sim, asset_dir, self.asset_file, asset_options)

        rs_props = self.gym.get_asset_rigid_shape_properties(asset)
        for p in rs_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.1
        self.gym.set_asset_rigid_shape_properties(asset, rs_props)

        return asset

    def configure_actor(self, gym, env):
        actor = self.gym.create_actor(
            env,
            self.asset,
            gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.1)),
            self.name,
            0,
            0,
            segmentationId=2,
        )

        rigid_body_props = self.gym.get_actor_rigid_body_properties(self.env, actor)
        self.mass = sum(x.mass for x in rigid_body_props)
        com = rigid_body_props[0].com
        self.CG = torch.tensor([com.x, com.y, com.z])
        return actor

    def reset_actor(self):
        self.rb_states[0] *= 0.0
        self.rb_states[0][6] = 1.0

        (rb_states,) = self.gym.get_actor_rigid_body_states(
            self.env, self.actor, gymapi.STATE_ALL
        )
        rb_states["pose"]["p"].fill((0, 0, 0))
        rb_states["pose"]["r"].fill((0, 0, 0, 1))
        rb_states["vel"]["linear"].fill((0, 0, 0))
        rb_states["vel"]["angular"].fill((0, 0, 0))
        self.gym.set_actor_rigid_body_states(
            self.env,
            self.actor,
            rb_states,
            gymapi.STATE_ALL,
        )


class PowerDrill(TeddyBear):

    workspace = "powerdrill"
    centroid = np.zeros(3)
    grasp_points = torch.tensor(
        [
            [-0.038539, 0.115021, 0.023878],
            [0.030017, -0.002467, 0.027816],
            [-0.029284, -0.099212, 0.027294],
        ]
    )

    grasp_normals = torch.tensor([[0, -1.0, 0.0], [-1, 0.0, 0.0], [1.0, 1.0, 0.0]])

    asset_file = "objects/urdf/power_drill.urdf"
    name = "power_drill"


class Banana(TeddyBear):
    workspace = "banana"
    grasp_points = torch.tensor(
        [
            [-0.00693, 0.085422, 0.013867],
            [0.018317, -0.001611, 0.013867],
            [-0.058538, -0.051027, 0.013867],
        ]
    )

    grasp_normals = torch.tensor([[1, -1.5, 0.0], [-2, 1.0, 0.0], [1, 0.0, 0.0]])
    asset_file = "objects/urdf/banana.urdf"
    name = "banana"


class Spatula(TeddyBear):
    asset_file = "objects/urdf/spatula.urdf"
    name = "spatula"


class Mug(TeddyBear):
    asset_file = "objects/urdf/mug.urdf"
    data_dir = "./nerf_shared/data/mug"
    config_path = "./nerf_shared/configs/mug.txt"
    name = "mug"


class BleachCleanser(TeddyBear):
    asset_file = "objects/urdf/bleach_cleanser.urdf"
    data_dir = "./nerf_shared/data/bleach_cleanser"
    config_path = "./nerf_shared/configs/bleach_cleanser.txt"
    name = "bleach_cleanser"


class Robot:
    # TODO this is where to robot contoler will live (need to just move it)

    dof_min = None
    dof_max = None
    dof_default = None

    def __init__(
        self,
        gym,
        sim,
        env,
        use_nerf_grasping=False,
        use_residual_dirs=False,
        metric="l1",
        cem_iters=10,
        cem_samples=500,
        target_height=0.07,
        use_grad_est=False,
    ):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.viewer = None

        self.added_lines = False
        self.asset = self.create_asset()
        self.actor = self.configure_actor()
        self.use_nerf_grasping = use_nerf_grasping
        self.use_residual_dirs = use_residual_dirs
        self.grasp_points = self.grasp_normals = self.grad_ests = None
        self.metric = metric
        self.cem_iters = cem_iters
        self.cem_samples = cem_samples
        self.target_height = target_height
        self.use_grad_est = use_grad_est

    def create_asset(self):
        robot_urdf_file = (
            "trifinger/robot_properties_fingers/urdf/pro/trifingerpro.urdf"
        )
        # robot_urdf_file = "trifinger/robot_properties_fingers/urdf/trifinger_with_stage.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True
        asset_options.disable_gravity = (
            True  # to make things easier - will eventually compensate ourselves
        )

        robot_asset = self.gym.load_asset(
            self.sim, asset_dir, robot_urdf_file, asset_options
        )

        trifinger_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        for p in trifinger_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.1
        self.gym.set_asset_rigid_shape_properties(robot_asset, trifinger_props)

        fingertips_frames = [
            "finger_tip_link_0",
            "finger_tip_link_120",
            "finger_tip_link_240",
        ]
        self.fingertips_frames = {}

        for frame_name in fingertips_frames:
            frame_handle = self.gym.find_asset_rigid_body_index(robot_asset, frame_name)
            assert frame_handle != gymapi.INVALID_HANDLE
            self.fingertips_frames[frame_name] = frame_handle

        dof_names = []
        for finger_pos in ["0", "120", "240"]:
            dof_names += [
                f"finger_base_to_upper_joint_{finger_pos}",
                f"finger_upper_to_middle_joint_{finger_pos}",
                f"finger_middle_to_lower_joint_{finger_pos}",
            ]

        self.dofs = {}  # TODO fix asset vs actor index differnce
        for dof_name in dof_names:
            dof_handle = self.gym.find_asset_dof_index(robot_asset, dof_name)
            assert dof_handle != gymapi.INVALID_HANDLE
            self.dofs[dof_name] = dof_handle

        return robot_asset

    def configure_actor(self):
        self.actor = self.gym.create_actor(
            self.env,
            self.asset,
            gymapi.Transform(),
            "Trifinger",
            0,
            0,
            segmentationId=5,
        )
        self.reset_actor()
        return self.actor

    def reset_actor(self):
        max_torque_Nm = 0.36
        # maximum joint velocity (in rad/s) on each actuator
        max_velocity_radps = 10
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)

        robot_dof_props = self.gym.get_asset_dof_properties(self.asset)
        for k, dof_index in enumerate(self.dofs.values()):
            # note: since safety checks are employed, the simulator PD controller is not
            #       used. Instead the torque is computed manually and applied, even if the
            #       command mode is 'position'.
            robot_dof_props["driveMode"][dof_index] = gymapi.DOF_MODE_EFFORT
            robot_dof_props["stiffness"][dof_index] = 0.0
            robot_dof_props["damping"][dof_index] = 0.0
            # set dof limits
            robot_dof_props["effort"][dof_index] = max_torque_Nm
            robot_dof_props["velocity"][dof_index] = max_velocity_radps
            # joint limits
            robot_dof_props["lower"][dof_index] = float(([-0.33, 0.0, -2.7] * 3)[k])
            robot_dof_props["upper"][dof_index] = float(([1.0, 1.57, 0.0] * 3)[k])
            # TODO make this read from strcuture

            # defaults
            dof_states[dof_index, 0] = float(([-0.8, 1.2, -2.7] * 3)[k])
            dof_states[dof_index, 1] = float(([0.0, 0.0, 0.0] * 3)[k])

        self.gym.set_actor_dof_properties(self.env, self.actor, robot_dof_props)

        print("setting dof state")
        self.gym.set_dof_state_tensor(self.sim, _dof_states)

        if self.added_lines:
            self.gym.clear_lines(self.viewer)
        self.added_lines = False
        self.grasp_points = self.grasp_normals = self.grad_ests = None
        self.previous_global_forces = None
        return

    def setup_tensors(self):
        # I didn't know we have to get the tensors every time?
        # segfaults if we only do it once
        _mass_matrix = self.gym.acquire_mass_matrix_tensor(self.sim, "Trifinger")
        self.mass_matrix = gymtorch.wrap_tensor(_mass_matrix)

        # for fixed base
        # jacobian[env_index, link_index - 1, :, dof_index]
        _jac = self.gym.acquire_jacobian_tensor(self.sim, "Trifinger")
        self.jacobian = gymtorch.wrap_tensor(_jac)

        # TODO MAKE this local
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        # (num_dof, 2)
        dof_count = self.gym.get_actor_dof_count(self.env, self.actor)
        dof_start_index = self.gym.get_actor_dof_index(
            self.env, self.actor, 0, gymapi.DOMAIN_SIM
        )

        self.dof_states = gymtorch.wrap_tensor(_dof_states)[
            dof_start_index : dof_start_index + dof_count, :
        ]

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # (num_rigid_bodies, 13)
        rb_count = self.gym.get_actor_rigid_body_count(self.env, self.actor)
        rb_start_index = self.gym.get_actor_rigid_body_index(
            self.env, self.actor, 0, gymapi.DOMAIN_SIM
        )

        # NOTE: simple indexing will return a view of the data but advanced indexing will return a copy breaking the updateing
        self.rb_states = gymtorch.wrap_tensor(_rb_states)[
            rb_start_index : rb_start_index + rb_count, :
        ]

    def sample_grasp(self, obj, residual_dirs=False, metric="l1"):
        assert obj.nerfs_loaded

        # currently using object centroid to offset grasp, which is loaded
        # using groundtruth mesh model. TODO: switch to nerf mesh
        gp = np.array([[0.09, 0.09, 0.0], [0, -0.125, 0.0], [-0.09, 0.09, 0.0]])
        if obj.use_centroid:
            gp += obj.centroid

        grasp_points = torch.tensor(gp).reshape(1, 3, 3)
        if residual_dirs:
            grasp_dirs = torch.zeros_like(grasp_points)
        else:
            grasp_dirs = -grasp_points

        def constraint(x):
            return torch.all(x.abs() <= 0.1, dim=1)

        mu_0 = torch.cat([grasp_points, grasp_dirs], dim=-1).reshape(-1).cuda()
        Sigma_0 = torch.diag(
            torch.cat(
                [torch.tensor([5e-3, 5e-3, 5e-3, 1e-3, 1e-3, 1e-3]) for _ in range(3)]
            )
        ).cuda()

        grasp_points = grasp_opt.get_points_cem(
            3,
            obj.model,
            mu_0=mu_0,
            Sigma_0=Sigma_0,
            constraint=constraint,
            cost_fn=metric,
            residual_dirs=residual_dirs,
            num_iters=self.cem_iters,
            num_samples=self.cem_samples,
        )
        # get grasp distribution from grasp_points
        _, _, weights, z_vals = grasp_utils.get_grasp_distribution(
            grasp_points.reshape(1, 3, 6),
            obj.model,
            residual_dirs=residual_dirs,
        )

        rays_o, rays_d = grasp_points[:, :3], grasp_points[:, 3:]

        centroid = torch.tensor(obj.centroid[:, None]).to(rays_o)
        if not obj.use_centroid:
            centroid *= 0.0

        if residual_dirs:
            dirs = lietorch.SO3.exp(rays_d).matrix()[:, :3, :3] @ (
                centroid - rays_o.unsqueeze(-1)
            )
            rays_d = dirs.squeeze()
        # if obj.use_centroid:
        #     rays_o += torch.tensor(obj.centroid, device=rays_o.device)
        rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
        # run z-val corrections on origins
        des_z_dist = 0.1
        # z_correction shape: (number of fingers, 1)
        z_correction = des_z_dist - torch.sum(weights * z_vals, dim=-1).reshape(3, 1)
        rays_o += z_correction * rays_d

        self.visualize_grasp_normals(rays_o, rays_d, des_z_dist)
        return rays_o, rays_d

    def visualize_grasp_normals(self, rays_o, rays_d, des_z_dist=0.1):
        ro, rd = rays_o.detach().cpu().numpy(), rays_d.detach().cpu().numpy()
        vertices = []
        for i in range(3):
            vertices.append(ro[i])
            vertices.append(ro[i] + rd[i] * des_z_dist)
        vertices = np.stack(vertices, axis=0)
        colors = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype="float32")
        if self.added_lines:
            self.gym.clear_lines(self.viewer)

        self.gym.add_lines(
            self.viewer,
            self.env,
            3,
            vertices,
            colors,
        )
        # self.gym.draw_env_rigid_contacts(
        #     self.viewer, self.env, gymapi.Vec3(0, 0, 1), 1.0, True
        # )
        self.added_lines = True

    @staticmethod
    def eval_grasps(obj, n_grasps=50, residual_dirs=False):
        assert obj.nerfs_loaded

    def get_tip_positions(self, offset_dir=None, radius=0.009):
        tip_indices = [
            self.fingertips_frames[f"finger_tip_link_{finger_pos}"]
            for finger_pos in [0, 120, 240]
        ]
        tip_positions = self.rb_states[tip_indices, :3]
        if offset_dir is not None:
            tip_positions -= offset_dir.to(tip_positions.device) * radius
        return tip_positions

    def control(self, interation, obj):

        # safe_pos = torch.tensor( [[ 0.0,  0.10, 0.05,],
        #                          [ 0.05,-0.10, 0.05,],
        #                          [-0.05,-0.10, 0.05,]])
        if not obj.nerfs_loaded:
            obj.load_nerf_models()

        if self.use_nerf_grasping:
            if self.grasp_points is None:
                self.grasp_points, self.grasp_normals = self.sample_grasp(
                    obj, residual_dirs=self.use_residual_dirs, metric=self.metric
                )
            grasp_points = self.grasp_points
            grasp_normals = self.grasp_normals
        else:
            grasp_points = obj.grasp_points
            grasp_normals = obj.grasp_normals
            grasp_normals /= grasp_normals.norm(dim=1, keepdim=True)

        # safe grasp point moves slightly off of contact surface
        safe_pos = grasp_points - grasp_normals * 0.1

        interation = interation % 1000

        mode = "off"
        if interation < 30:
            mode = "off"  # Box needs this to get ot graps position - bear can't have it
        elif interation < 60:
            mode = "safe"
        elif interation < 140:
            mode = "pos"
        elif interation < 200:
            mode = "vel"
        else:
            mode = "up"

        if interation % 10 == 0:
            logging.info("%s, %s", interation, mode)

        if mode == "off":
            pass
        if mode == "safe":
            self.position_control(safe_pos)
        if mode == "pos":
            self.position_control(grasp_points)
        if mode == "vel":
            # move radialy in along xy plane
            # normal = - (grasp_points - torch.mean(grasp_points, axis=0))
            # normal[:, -1] = 0
            self.vel_control_force_limit(grasp_points, grasp_normals)
        if mode == "up":
            if self.grad_ests is None:
                tip_positions = self.get_tip_positions(
                    offset_dir=grasp_normals, radius=0.009
                ).reshape(1, 3, 3)
                _, grad_ests = grasp_utils.est_grads_vals(
                    obj.model, tip_positions.cuda(), sigma=1e-1
                )
                grad_ests = grad_ests.reshape(3, 3).float()
                grad_ests /= grad_ests.norm(dim=1, keepdim=True)
                tip_positions = tip_positions.reshape(3, 3)
                logging.info("grad_ests: %s", grad_ests)
                logging.info("grasp_pts: %s", grasp_points)
                logging.info("tip_positions: %s", tip_positions)
                self.grad_ests = grad_ests
                print("visualizing grad ests")
                self.visualize_grasp_normals(grasp_points, grad_ests)
            else:
                # Assuming that contact normals stay the same, i.e. static contacts
                grad_ests = self.grad_ests
                tip_positions = self.get_tip_positions(
                    offset_dir=grad_ests, radius=0.009
                ).reshape(3, 3)
            in_normals = grad_ests if self.use_grad_est else grasp_normals
            pos_target = torch.tensor([0, 0, self.target_height])
            self.object_pos_control(grasp_points, in_normals, obj, pos_target)

    def apply_fingertip_forces(self, global_fingertip_forces):
        applied_torque = torch.zeros((9))
        for finger_index, finger_pos in enumerate([0, 120, 240]):
            robot_dof_names = [
                f"finger_base_to_upper_joint_{finger_pos}",
                f"finger_upper_to_middle_joint_{finger_pos}",
                f"finger_middle_to_lower_joint_{finger_pos}",
            ]

            dof_idx = [self.dofs[dof_name] for dof_name in robot_dof_names]
            tip_index = self.fingertips_frames[f"finger_tip_link_{finger_pos}"]

            local_jacobian = self.jacobian[0, tip_index - 1, :3, dof_idx]

            joint_torques = (
                torch.t(local_jacobian) @ global_fingertip_forces[finger_index, :]
            )
            applied_torque[dof_idx] = joint_torques

        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(applied_torque)
        )

    def position_control(self, grasp_points):
        applied_torque = torch.zeros((9))
        for finger_index, finger_pos in enumerate([0, 120, 240]):
            robot_dof_names = [
                f"finger_base_to_upper_joint_{finger_pos}",
                f"finger_upper_to_middle_joint_{finger_pos}",
                f"finger_middle_to_lower_joint_{finger_pos}",
            ]

            dof_idx = [self.dofs[dof_name] for dof_name in robot_dof_names]
            tip_index = self.fingertips_frames[f"finger_tip_link_{finger_pos}"]

            # dof_idx = [self.gym.find_actor_dof_index(self.env, self.actor, dof_name, gymapi.DOMAIN_ACTOR) for dof_name in robot_dof_names]
            # tip_index =  self.gym.find_actor_rigid_body_index(self.env, self.actor, f"finger_tip_link_{finger_pos}", gymapi.DOMAIN_ACTOR)

            # only care about tip position
            local_jacobian = self.jacobian[0, tip_index - 1, :3, dof_idx]
            tip_state = self.rb_states[tip_index, :]

            tip_pos = tip_state[:3]
            tip_vel = tip_state[7:10]

            # pos_target = rot_matrix_finger @ torch.tensor([0.0 , 0.15, 0.09])
            pos_target = grasp_points[finger_index, :].cpu()

            # PD controller in xyz space
            pos_error = tip_pos - pos_target
            xyz_force = -5.0 * pos_error - 1.0 * tip_vel

            joint_torques = torch.t(local_jacobian) @ xyz_force
            applied_torque[dof_idx] = joint_torques

        # TODO use global_fingertip_forces
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(applied_torque)
        )

    def vel_control_force_limit(
        self, grasp_points, grasp_normals, target_vel=0.05, max_force=1.5
    ):
        applied_torque = torch.zeros((9))
        for finger_index, finger_pos in enumerate([0, 120, 240]):
            robot_dof_names = [
                f"finger_base_to_upper_joint_{finger_pos}",
                f"finger_upper_to_middle_joint_{finger_pos}",
                f"finger_middle_to_lower_joint_{finger_pos}",
            ]

            dof_idx = [
                self.gym.find_actor_dof_index(
                    self.env, self.actor, dof_name, gymapi.DOMAIN_SIM
                )
                for dof_name in robot_dof_names
            ]
            tip_index = self.gym.find_actor_rigid_body_index(
                self.env, self.actor, f"finger_tip_link_{finger_pos}", gymapi.DOMAIN_SIM
            )

            # only care about tip position
            local_jacobian = self.jacobian[0, tip_index - 1, :3, dof_idx]
            tip_state = self.rb_states[tip_index, :]

            tip_pos = tip_state[:3]
            tip_vel = tip_state[7:10]
            grasp_points = grasp_points.to(tip_pos.device)
            grasp_normals = grasp_normals.to(tip_pos.device)

            # pos_target = rot_matrix_finger @ torch.tensor([0.0 , 0.15, 0.09])
            pos_target = grasp_points[finger_index, :]

            # define vector along which endeffector should close
            # will be position controlled perpendicular to motion
            # and velocity controlled along. Force along with clamped to prevent crushing
            start_point = pos_target

            # normal      = rot_matrix_finger @ torch.tensor([0.0 , -0.05, 0])
            # normal = torch.tensor([ 0., 0., pos_target[-1]]) - pos_target #TODO HACK
            normal = grasp_normals[finger_index, :]
            # normal /= normal.norm()
            # normal = grasp_normals / torch.norm(grasp_normals, axis=-1, keepdim=True)

            pos_relative = tip_pos - start_point

            perp_xyz_force = -5.0 * pos_relative - 1.0 * tip_vel
            perp_xyz_force = perp_xyz_force - normal * normal.dot(perp_xyz_force)

            vel_error = normal.dot(tip_vel) - target_vel

            parallel_xyz_force_mag = -5.0 * vel_error
            parallel_xyz_force = torch.clamp(
                parallel_xyz_force_mag, -max_force, max_force
            )

            xyz_force = parallel_xyz_force * normal + perp_xyz_force

            joint_torques = torch.t(local_jacobian) @ xyz_force
            applied_torque[dof_idx] = joint_torques

        # TODO use global_fingertip_forces
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(applied_torque)
        )

    def object_pos_control(self, grasp_points, in_normal, obj, target_pos):
        # TODO grasp points should in object frame (rotate with erros in object rotation)
        pos = obj.rb_states[0, 0:3]
        quat = obj.rb_states[0, 3:7]
        vel = obj.rb_states[0, 7:10]
        angular_vel = obj.rb_states[0, 10:13]

        quat = Quaternion.fromWLast(quat)
        target_quat = Quaternion.Identity()

        # cg_pos = pos
        cg_pos = obj.get_CG()  # thoughts it eliminates the pengulum effect? possibly?
        # target_pos = torch.tensor([0, 0, self.target_height])  # TEMP

        # logging.debug(f"pos={pos}")
        # logging.debug(f"CG={obj.CG}")
        # logging.debug(f"cg_pos={cg_pos}")
        # logging.debug(f"quat={quat}")
        # logging.debug(f"target_pos={target_pos}")
        # logging.debug(f"target_quat={target_quat}")

        pos_error = cg_pos - target_pos

        if not hasattr(self, "zpos_error_integral"):
            self.zpos_error_integral = 0
        ki = 0.02
        self.zpos_error_integral += pos_error[2] * ki

        object_weight_comp = obj.mass * 9.8 * torch.tensor([0, 0, 1])
        # object_weight_comp = - self.zpos_error_integral * torch.tensor([0, 0, 1])

        # Box tunning - tunned without moving CG and compensated normals
        target_force = object_weight_comp - 0.2 * pos_error - 0.10 * vel
        target_torque = (
            -0.4 * (quat @ target_quat.T).to_tangent_space() - 0.01 * angular_vel
        )

        # Bear tunning
        # target_force = object_weight_comp - 0.2 * pos_error - 0.10 * vel
        # target_torque = (
        #     -0.4 * (quat @ target_quat.T).to_tangent_space() - 0.01 * angular_vel
        # )

        # banana tunigng
        target_force = object_weight_comp - 0.9 * pos_error - 0.40 * vel
        target_torque = (
            -0.04 * (quat @ target_quat.T).to_tangent_space() - 0.0001 * angular_vel
        )

        # target_torque = torch.zeros((3))
        # target_force = 1.1 * obj.mass * 9.8 * torch.tensor([0,0,1])

        # target_torque = torch.tensor([0, 0, -0.01])
        # target_force = torch.zeros((3))

        # logging.debug(f"target_force={target_force}")
        # logging.debug(f"target_torque={target_torque}")

        # print("global_cg.shape", global_cg.shape)
        tip_positions = self.get_tip_positions()
        # grad_ests = torch.stack([quat.rotate(x) for x in self.grad_ests], axis=0)

        # self.visualize_grasp_normals(tip_positions, grad_ests)

        # not necessary for box - changes tunning parameters
        # makes the grasp points and normals follow the tip positions and object rotation
        grasp_points = tip_positions - cg_pos
        in_normal = torch.stack([quat.rotate(x) for x in in_normal], axis=0)

        try:
            global_forces = calculate_grip_forces(
                grasp_points, in_normal, target_force, target_torque
            )
        except AssertionError:
            logging.warning("solve failed, maintaining previous forces")
            global_forces = (
                self.previous_global_forces
            )  # will fail if we failed solve on first iteration
            assert global_forces is not None
        else:
            self.previous_global_forces = global_forces

        # logging.debug("per finger force: %s", global_forces)
        # logging.debug("applied force: %s", torch.sum(global_forces, dim=0))

        # _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        # net_cf = gymtorch.wrap_tensor(_net_cf)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # print(f"net_cf={net_cf[obj.index,:]}")
        # logging.debug(f"net_cf={net_cf}")

        # logging.debug("grasp_points: %s", grasp_points)
        # logging.debug("in_normal: %s", in_normal)
        self.apply_fingertip_forces(global_forces)


class TriFingerEnv:
    def __init__(
        self, viewer=True, robot=True, Obj=None, save_cameras=False, **robot_kwargs
    ):
        self.args = gymutils.parse_arguments(description="Trifinger test")
        self.gym = gymapi.acquire_gym()

        self.setup_sim()
        self.setup_envs(robot=robot, Obj=Obj, **robot_kwargs)

        if viewer:
            self.setup_viewer()
        else:
            self.viewer = None

        if save_cameras:
            self.setup_cameras(self.env)
        else:
            self.camera_handles = []

        self.gym.prepare_sim(self.sim)
        self.image_idx = 0

    def setup_sim(self):
        # only tested with this one
        assert self.args.physics_engine == gymapi.SIM_PHYSX

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0

        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = self.args.num_threads
        sim_params.physx.use_gpu = self.args.use_gpu
        # sim_params.physx.use_gpu = True

        # sim_params.use_gpu_pipeline = True
        sim_params.use_gpu_pipeline = False
        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            self.args.physics_engine,
            sim_params,
        )
        assert self.sim is not None

        # intensity = 0.01 # for nerf generation
        # ambient = 0.21 / intensity
        intensity = 0.5
        ambient = 0.10 / intensity
        intensity = gymapi.Vec3(intensity, intensity, intensity)
        ambient = gymapi.Vec3(ambient, ambient, ambient)

        self.gym.set_light_parameters(
            self.sim, 0, intensity, ambient, gymapi.Vec3(0.5, 1, 1)
        )
        self.gym.set_light_parameters(
            self.sim, 1, intensity, ambient, gymapi.Vec3(1, 0, 1)
        )
        self.gym.set_light_parameters(
            self.sim, 2, intensity, ambient, gymapi.Vec3(0.5, -1, 1)
        )
        self.gym.set_light_parameters(
            self.sim, 3, intensity, ambient, gymapi.Vec3(0, 0, 1)
        )

    def setup_envs(self, robot, Obj, **robot_kwargs):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        self.gym.add_ground(self.sim, plane_params)

        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        env = self.gym.create_env(self.sim, env_lower, env_upper, 0)

        self.env = env  # used only when there is one env
        self.envs = [env]

        if robot:
            self.robot = Robot(self.gym, self.sim, self.env, **robot_kwargs)
        else:
            self.robot = Mock()

        self.setup_stage(env)

        if Obj is not None:
            self.object = Obj(self.gym, self.sim, self.env)
        else:
            self.object = Mock()

        self.robot.setup_tensors()
        self.object.setup_tensors()

    def setup_stage(self, env):
        # this one is convex decomposed
        stage_urdf_file = (
            "trifinger/robot_properties_fingers/urdf/high_table_boundary.urdf"
        )
        # stage_urdf_file = "trifinger/robot_properties_fingers/urdf/trifinger_stage.urdf"
        # stage_urdf_file = "trifinger/robot_properties_fingers/urdf/stage.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True
        asset_options.thickness = 0.001

        stage_asset = self.gym.load_asset(
            self.sim, asset_dir, stage_urdf_file, asset_options
        )
        self.gym.create_actor(
            env, stage_asset, gymapi.Transform(), "Stage", 0, 0, segmentationId=1
        )

    def setup_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.robot.viewer = self.viewer
        assert self.viewer is not None

        cam_pos = gymapi.Vec3(0.7, 0.175, 0.6)
        cam_target = gymapi.Vec3(0, 0, 0.2)
        self.gym.viewer_camera_look_at(self.viewer, self.env, cam_pos, cam_target)

    def setup_cameras(self, env):
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = 35.0
        camera_props.width = 400
        camera_props.height = 400

        # generates cameara positions along rings around object
        heights = [0.6, 0.3, 0.9, 1.0]
        distances = [0.25, 0.4, 0.5, 0.1]
        counts = [56, 104, 96, 1]
        target_z = [0.0, 0.1, 0.2, 0.1]

        camera_positions = []
        for h, d, c, z in zip(heights, distances, counts, target_z):
            for alpha in np.linspace(0, 2 * np.pi, c, endpoint=False):
                camera_positions.append(([d * np.sin(alpha), d * np.cos(alpha), h], z))

        self.camera_handles = []
        for pos, z in camera_positions:
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            self.gym.set_camera_location(
                camera_handle, env, gymapi.Vec3(*pos), gymapi.Vec3(0, 0, z)
            )

            self.camera_handles.append(camera_handle)

    def setup_save_dir(self, folder, overwrite=False):
        path = Path(folder)

        if path.exists():
            print(path, "already exists!")
            if overwrite:
                shutil.rmtree(path)
            elif input("Clear it before continuing? [y/N]:").lower() == "y":
                shutil.rmtree(path)

        path.mkdir()
        return path

    def save_viewer_frame(self, save_dir, save_freq=10):
        """Saves frame from viewer to"""
        self.gym.render_all_camera_sensors(self.sim)
        path = Path(save_dir)
        if not path.exists():
            path = self.setup_save_dir(save_dir)
        if self.image_idx % save_freq == 0:
            self.gym.write_viewer_image_to_file(
                self.viewer, str(path / f"img{self.image_idx}.png")
            )
        self.image_idx += 1

    def save_images(self, folder, overwrite=False):
        self.gym.render_all_camera_sensors(self.sim)

        path = self.setup_save_dir(folder, overwrite)

        for i, camera_handle in enumerate(self.camera_handles):
            print(f"saving camera {i}")

            color_image = self.gym.get_camera_image(
                self.sim, self.env, camera_handle, gymapi.IMAGE_COLOR
            )
            color_image = color_image.reshape(400, 400, -1)
            Image.fromarray(color_image).save(path / f"col_{i}.png")

            color_image = self.gym.get_camera_image(
                self.sim, self.env, camera_handle, gymapi.IMAGE_SEGMENTATION
            )
            color_image = color_image.reshape(400, 400) * 30
            Image.fromarray(color_image, "I").convert("L").save(path / f"seg_{i}.png")

            color_image = self.gym.get_camera_image(
                self.sim, self.env, camera_handle, gymapi.IMAGE_DEPTH
            )
            # distance in units I think
            color_image = -color_image.reshape(400, 400)
            color_image = (np.clip(color_image, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(color_image).convert("L").save(path / f"dep_{i}.png")

            pos, quat = get_fixed_camera_transfrom(
                self.gym, self.sim, self.env, camera_handle
            )

            with open(path / f"pos_xyz_quat_xyzw_{i}.txt", "w+") as f:
                data = [*pos.tolist(), *quat.q[1:].tolist(), quat.q[0].tolist()]
                json.dump(data, f)

    def save_images_nerf_ready(self, folder, overwrite=False):
        self.gym.render_all_camera_sensors(self.sim)

        path = Path(folder)

        if path.exists():
            print(path, "already exists!")
            if overwrite:
                shutil.rmtree(path)
            elif input("Clear it before continuing? [y/N]:").lower() == "y":
                shutil.rmtree(path)

        path.mkdir()
        (path / "train").mkdir()
        (path / "test").mkdir()
        (path / "val").mkdir()

        json_meta = {"camera_angle_x": np.radians(self.fov), "frames": []}

        for i, camera_handle in enumerate(self.camera_handles):
            print(f"saving camera {i}")

            color_image = self.gym.get_camera_image(
                self.sim, self.env, camera_handle, gymapi.IMAGE_COLOR
            )
            color_image = color_image.reshape(400, 400, -1)
            Image.fromarray(color_image).save(path / "train" / f"col_{i}.png")

            pos, quat = get_fixed_camera_transfrom(
                self.gym, self.sim, self.env, camera_handle
            )

            rot_matrix = quat.get_matrix()
            transform_matrix = torch.vstack(
                [torch.hstack([rot_matrix, pos[:, None]]), torch.tensor([0, 0, 0, 1])]
            )

            image_data = {
                "file_path": f"./train/col_{i}",  # note the lack of ".png" it gets added in the load script
                "transform_matrix": transform_matrix.tolist(),
            }

            json_meta["frames"].append(image_data)

        with open(path / "transforms_train.json", "w+") as f:
            json.dump(json_meta, f, indent=4)

        empty_meta = {"camera_angle_x": np.radians(self.fov), "frames": []}

        with open(path / "transforms_test.json", "w+") as f:
            json.dump(empty_meta, f)

        with open(path / "transforms_val.json", "w+") as f:
            json.dump(empty_meta, f)

    def refresh_tensors(self):
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def step_gym(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.step_graphics(self.sim)
        if self.viewer is not None:
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        self.refresh_tensors()

    def reset(self):
        # reset object after robot actor
        self.robot.reset_actor()
        # reset object actor
        self.object.reset_actor()
        self.refresh_tensors()
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.step_gym()

        self.image_idx = 0


def get_nerf_training(viewer):
    # Obj = None
    # Obj = Box
    # Obj = TeddyBear
    Obj = PowerDrill  # put verticaly?
    # Obj = Banana
    # Obj = BleachCleanser # too big - put on side?
    # Obj = Spatula
    # Obj = Mug

    tf = TriFingerEnv(viewer=viewer, robot=False, Obj=Obj, save_cameras=True)
    for _ in range(500):
        tf.step_gym()
        if Obj is not None:
            print(tf.object.rb_states[0, :7])

    name = "blank" if Obj is None else Obj.name
    tf.save_images("./nerf_shared/data/isaac_" + name, overwrite=False)


def run_robot_control(viewer, Obj, **robot_kwargs):
    tf = TriFingerEnv(viewer=viewer, robot=True, Obj=Obj, **robot_kwargs)
    count = 0
    while not tf.gym.query_viewer_has_closed(tf.viewer):
        try:
            count += 1
            # force = torch.tensor([0,0,1]) * 9.8 *  tf.object.mass * 1.0
            # force = torch.stack( [force, force], dim = 0)
            # force = gymtorch.unwrap_tensor(force)
            # tf.gym.apply_rigid_body_force_tensors(tf.sim, force , None, gymapi.ENV_SPACE)
            tf.step_gym()
            tf.robot.control(count, tf.object)
        except KeyboardInterrupt:
            import pdb

            pdb.set_trace()
        finally:
            pass
    print("closed!")


if __name__ == "__main__":
    # get_nerf_training(viewer=False)
    # Obj = Box
    # Obj = TeddyBear
    # Obj = PowerDrill
    Obj = Banana
    # Obj = BleachCleanser # too big - put on side?
    # Obj = Spatula
    # Obj = Mug
    run_robot_control(
        viewer=True,
        Obj=Obj,
        use_nerf_grasping=False,
        use_residual_dirs=False,
        metric="l1",
    )
