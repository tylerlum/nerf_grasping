import logging

from isaacgym import gymapi, gymtorch
from pathlib import Path
import os
import pdb
import numpy as np
import torch
import cvxpy
from typing import Tuple, Any, Optional
from nerf_grasping import grasp_opt, grasp_utils, nerf_utils
from nerf_grasping.config import RobotConfig
from nerf_grasping.control import force_opt
from nerf_grasping.sim import ig_utils, ig_objects

from nerf_grasping.quaternions import Quaternion

root_dir = Path(os.path.abspath(__file__)).parents[2]
asset_dir = f"{root_dir}/assets"


class FingertipRobot:
    """Robot controlling 3 (or n) spheres as "fingers" to grasp an lift an object"""

    def __init__(
        self,
        config: RobotConfig,
    ):
        """
        Initializes Fingertip-only point mass robot
        """
        self.controller_params = config.controller_params
        self.target_height = config.target_height
        self.use_true_normals = config.gt_normals
        self.norm_start_offset = config.des_z_dist
        self.mu = config.mu
        self.sphere_radius = config.sphere_radius
        self.verbose = config.verbose
        self.debug = self.verbose
        self.grad_config = config.grad_config
        self.actors = []
        self.initialized_actors = False

    def setup_gym(
        self,
        gym: gymapi.Gym,
        sim: Any,
        env: Any,
        grasp_vars: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.actors = self.create_spheres(grasp_vars)
        self.initialized_actors = True

    def create_spheres(self, grasp_vars=None):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.slices_per_cylinder = 40
        asset_options.disable_gravity = True
        asset_options.density = 1000
        asset_options.override_inertia = False
        asset_options.default_dof_drive_mode = gymapi.DOF_TRANSLATION
        markers = []
        actors = []
        colors = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype="float32")

        for i, pos in enumerate(self.get_ftip_start_pos(grasp_vars)):
            color = colors[i]
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(pos[0], pos[1], pos[2])
            marker_asset = self.gym.load_asset(
                self.sim, asset_dir, "objects/urdf/ball.urdf", asset_options
            )
            # marker_asset = self.gym.create_sphere(
            #     self.sim, self.sphere_radius, asset_options
            # )
            markers.append(marker_asset)
            actor_handle = self.gym.create_actor(
                self.env,
                marker_asset,
                pose,
                f"fingertip_{i}",
            )
            actors.append(actor_handle)
            self.gym.set_rigid_body_color(
                self.env,
                actor_handle,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(*color),
            )
        for asset in markers:
            rs_props = self.gym.get_asset_rigid_shape_properties(asset)
            for p in rs_props:
                p.friction = self.mu
                p.torsion_friction = self.mu
                p.restitution = 0.0
        return actors

    def setup_tensors(self):
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # (num_rigid_bodies, 13)
        rb_count = self.gym.get_actor_rigid_body_count(self.env, self.actors[0])
        rb_start_index = self.gym.get_actor_rigid_body_index(
            self.env, self.actors[0], 0, gymapi.DOMAIN_SIM
        )
        print(f"actor rb_count: {rb_count}, start index: {rb_start_index}")

        # NOTE: simple indexing will return a view of the data but advanced indexing will return a copy breaking the updateing
        self.rb_states = gymtorch.wrap_tensor(_rb_states)[
            rb_start_index : rb_start_index + rb_count * len(self.actors), :
        ]

    def get_position(self):
        """Returns tip positions"""
        return self.rb_states[:, 0:3]

    def get_velocity(self):
        """Returns linear velocity"""
        return self.rb_states[:, 7:10]

    def get_contact_points(self, grasp_normals=None):
        """Returns IG-frame contact points, given current position + grasp normals."""
        contact_pts = self.position  # IG frame.
        if grasp_normals is not None:
            contact_pts += self.sphere_radius * grasp_normals  # IG frame.
        return contact_pts

    def get_nerf_grad_ests(self, obj, tip_position):
        if not obj.nerf_loaded:
            obj.load_nerf_model()

        # Express tip position in nerf (Y-up, relative to world origin).
        nerf_tip_pos = grasp_utils.ig_to_nerf(
            tip_position, return_tensor=True
        ) + torch.from_numpy(obj.translation)

        # Query NeRF for gradient estimates
        _, grad_ests = nerf_utils.est_grads_vals(
            obj.model, nerf_tip_pos.view(1, -1, 3), self.grad_config
        )
        grad_ests = grad_ests.reshape(3, 3).float()
        grad_ests /= grad_ests.norm(dim=1, keepdim=True)

        # Rotate back to IG frame.
        grad_ests = grasp_utils.nerf_to_ig(grad_ests)
        return grad_ests

    def apply_fingertip_forces(self, global_fingertip_forces):
        """Applies forces to individual actors"""
        assert global_fingertip_forces.shape == (
            3,
            3,
        ), f"actualshape:{global_fingertip_forces.shape}"
        self.previous_global_forces = global_fingertip_forces
        global_fingertip_forces = torch.clamp(global_fingertip_forces, -10, 10)
        nan_indices = torch.isnan(global_fingertip_forces)
        if nan_indices.any():
            global_fingertip_forces[nan_indices] = 0.0
            logging.warning("global_fingertip_forces contains nans!")
        for f, actor_handle in zip(global_fingertip_forces, self.actors):
            rb_handle = self.gym.get_actor_rigid_body_handle(self.env, actor_handle, 0)
            fx, fy, fz = f
            self.gym.apply_body_forces(
                self.env,
                rb_handle,
                gymapi.Vec3(fx, fy, fz),  # force
                gymapi.Vec3(0.0, 0.0, 0.0),  # torque
                gymapi.CoordinateSpace.ENV_SPACE,
            )

        return

    def reset_actor(self, grasp_vars):
        """Resets fingertips to points on grasp point lines"""
        # Erases last cvx opt solution
        self.previous_global_forces = None

        # Finds valid starting fingertip positions
        ftip_start_pos = self.get_ftip_start_pos(grasp_vars)

        init_states = []
        # Sets actor rigid body states of spheres
        for pos, handle in zip(ftip_start_pos, self.actors):
            state = self.gym.get_actor_rigid_body_states(
                self.env, handle, gymapi.STATE_POS
            )
            state["pose"]["p"].fill(tuple(pos))

            assert self.gym.set_actor_rigid_body_states(
                self.env, handle, state, gymapi.STATE_POS
            ), "gym.set_actor_rigid_body_states failed"
            state = self.gym.get_actor_rigid_body_states(
                self.env, handle, gymapi.STATE_VEL
            )
            state["vel"]["linear"].fill((0.0, 0.0, 0.0))
            state["vel"]["angular"].fill((0.0, 0.0, 0.0))
            assert self.gym.set_actor_rigid_body_states(
                self.env, handle, state, gymapi.STATE_VEL
            ), "gym.set_actor_rigid_body_states failed"
            init_states.append(state)
        return init_states

    def position_control(self, desired_position):
        """Computes joint torques using tip link jacobian to achieve desired tip position"""
        assert self.position.shape == desired_position.shape
        kp = self.controller_params.kp_reach
        kd = self.controller_params.kd_reach
        position_error = self.position - desired_position
        tip_velocities = self.velocity
        xyz_force = -kp * position_error - kd * tip_velocities
        return xyz_force

    def grasping_control(self, desired_position, grasp_normals):
        """Computes joint torques using tip link jacobian to achieve desired tip position"""
        assert self.position.shape == desired_position.shape
        kp = self.controller_params.kp_grasp
        kd = self.controller_params.kd_grasp
        normal_scale = self.controller_params.normal_scale_grasp
        position_error = self.position - desired_position
        tip_velocities = self.velocity
        xyz_force = (
            -kp * position_error - kd * tip_velocities + normal_scale * grasp_normals
        )
        return xyz_force

    def object_pos_control(
        self,
        obj,
        in_normals_ig_frame,
        target_position,
    ):
        """Object position control for lifting trajectory"""
        # Get controller params
        target_normal = self.controller_params.target_normal
        kp = self.controller_params.kp_lift
        kd = self.controller_params.kd_lift
        kp_rot = self.controller_params.kp_rot_lift
        kd_rot = self.controller_params.kd_rot_lift

        vel = obj.velocity  # IG frame.
        angular_vel = obj.angular_velocity  # IG frame.
        quat = Quaternion.fromWLast(obj.orientation)  # obj to IG frame.

        target_quat = Quaternion.Identity()
        # print('rot offset: ', (quat @ target_quat.T).to_tangent_space())

        pos_err = obj.position - target_position
        object_weight_comp = obj.mass * 9.81 * torch.tensor([0, 0, 1])
        # target_force = object_weight_comp - 0.9 * pos_err- 0.4 * vel
        # banana tuning
        # print('pd force: ', -kp*pos_err-kd * vel)

        target_force = object_weight_comp - kp * pos_err - kd * vel
        target_torque = (
            -kp_rot * (quat @ target_quat.T).to_tangent_space() - kd_rot * angular_vel
        )
        # Transform desired force/torque to object frame.
        target_force_obj_frame = quat.T.rotate(target_force)
        target_torque_obj_frame = quat.T.rotate(target_torque)

        # Transform inward normals to obj frame.
        in_normals_obj_frame = torch.stack(
            [quat.T.rotate(x) for x in in_normals_ig_frame]
        )

        # Get contact points in IG frame.
        grasp_points_ig_frame = self.get_contact_points(in_normals_ig_frame)

        # Transform them back to obj frame.
        grasp_points_obj_frame = torch.stack(
            [quat.T.rotate(x - obj.position) for x in grasp_points_ig_frame]
        )

        # print('grasp_points obj frame:', grasp_points_obj_frame)
        # print('inward normals obj frame: ', in_normals_obj_frame)

        try:
            if self.verbose:
                force_opt.check_force_closure(
                    grasp_points_obj_frame, in_normals_obj_frame, obj.mu
                )
            global_forces_obj_frame, success = force_opt.calculate_grip_forces(
                grasp_points_obj_frame,
                in_normals_obj_frame,
                target_force_obj_frame,
                target_torque_obj_frame,
                target_normal,
                obj.mu,
            )
        except cvxpy.error.SolverError:
            if self.debug:
                pdb.set_trace()
            global_forces_obj_frame, success = None, False

        if not success:
            logging.warning("solve failed, maintaining previous forces")
            global_forces_ig_frame = (
                self.previous_global_forces
            )  # will fail if we failed solve on first iteration
            assert global_forces_ig_frame is not None
        else:
            global_forces_ig_frame = torch.stack(
                [quat.rotate(x) for x in global_forces_obj_frame]
            )
            # print('desired forces: ', torch.sum(global_forces_ig_frame, dim=0))
            self.previous_global_forces = global_forces_ig_frame

        return global_forces_ig_frame, target_force, target_torque, success

    def get_ftip_start_pos(self, grasp_vars=None):
        if grasp_vars is None:
            grasp_points = ig_objects.Box.grasp_points.clone()
            grasp_normals = ig_objects.Box.grasp_normals.clone()
        else:
            grasp_points, grasp_normals = grasp_vars
        gn = grasp_normals.clone()
        gn[:, 2] *= 0
        gn = gn / gn.norm(dim=1, keepdim=True)
        ftip_start_pos = grasp_points - gn * self.norm_start_offset
        ftip_start_pos[:, 2] = np.clip(
            grasp_points[:, 2], self.sphere_radius * 1.15, np.inf
        )
        return ftip_start_pos

    @property
    def position(self):
        return self.get_position()

    @property
    def velocity(self):
        return self.get_velocity()

    @property
    def mass(self):
        masses = []
        for actor in self.actors:
            rigid_body_props = self.gym.get_actor_rigid_body_properties(self.env, actor)
            masses.append(sum([x.mass for x in rigid_body_props]))
        return masses
