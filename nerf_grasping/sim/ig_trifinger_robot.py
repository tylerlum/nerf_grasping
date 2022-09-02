import logging

from isaacgym import gymapi, gymtorch
from pathlib import Path
import os
import numpy as np
import torch

try:
    import pinocchio as pin
except ImportError:
    pin = None
    print("WARNING: Unable to import pinocchio, skipping import")

from nerf_grasping import grasp_opt, grasp_utils, nerf_utils
from nerf_grasping.config import RobotConfig
from nerf_grasping.control import pos_control, force_opt
from nerf_grasping.sim import ig_viz_utils

from nerf_grasping.quaternions import Quaternion

root_dir = Path(os.path.abspath(__file__)).parents[2]
asset_dir = f"{root_dir}/assets"


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
        config: RobotConfig,
    ):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.viewer = None

        self.urdf_filename = (
            "trifinger/robot_properties_fingers/urdf/pro/trifingerpro.urdf"
        )

        self.asset = self.create_asset()
        self.actor = self.configure_actor()
        self.target_height = config.target_height
        self.use_true_normals = config.gt_normals
        self.des_z_dist = config.des_z_dist

        print(os.path.exists("assets/" + self.urdf_filename))

        if pin is not None:
            self.pin_model = pin.buildModelFromUrdf("assets/" + self.urdf_filename)
            self.pin_data = self.pin_model.createData()

            pin.forwardKinematics(self.pin_model, self.pin_data, np.random.randn(9))
            pin.forwardKinematics(
                self.pin_model, self.pin_data, np.random.randn(9), np.random.randn(9)
            )

        self.pos_control_config = pos_control.load_config()

    def create_asset(self):

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True
        asset_options.disable_gravity = (
            False  # to make things easier - will eventually compensate ourselves
        )

        robot_asset = self.gym.load_asset(
            self.sim, asset_dir, self.urdf_filename, asset_options
        )

        trifinger_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        # print([rbp.mass for rbp in self.gym.get_asset_rigid_body_properties(robot_asset)])
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

    def reset_actor(self, grasp_vars=None):
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

        self.gym.clear_lines(self.viewer)
        self.added_lines = False
        self.previous_global_forces = None
        self.grad_ests = None
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

    def get_tip_positions(self, offset_dir=None, radius=0.009):
        tip_indices = [
            self.fingertips_frames[f"finger_tip_link_{finger_pos}"]
            for finger_pos in [0, 120, 240]
        ]
        tip_positions = self.rb_states[tip_indices, :3]
        if offset_dir is not None:
            tip_positions -= offset_dir.to(tip_positions.device) * radius
        return tip_positions

    def assign_closest_fingertips(self, rays_o, rays_d):
        tip_positions = self.get_tip_positions().to(rays_o)

        # computes tip/fingertip assignment with the minimum net distance to grasp contact point
        pdist = []
        for i in range(len(tip_positions)):
            pdist.append(
                torch.nn.functional.pairwise_distance(
                    tip_positions[i], rays_o + rays_d * self.des_z_dist
                )
            )
        pdist = torch.stack(pdist, dim=1)
        perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        net_dists = [sum([pdist[i][j] for i, j in enumerate(p)]) for p in perms]
        tip_idx = perms[torch.argmin(torch.tensor(net_dists)).numpy().item()]
        rays_o, rays_d = rays_o[tip_idx], rays_d[tip_idx]
        # colors = [[0, 1, 0]] * 3
        # ig_viz_utils.visualize_markers(self.gym, self.env, self.sim, gp, colors)

        if self.added_lines:
            self.gym.clear_lines(self.viewer)
        ig_viz_utils.visualize_grasp_normals(
            self.gym, self.viewer, self.env, rays_o, rays_d, self.des_z_dist
        )
        self.added_lines = True
        return rays_o, rays_d

    def get_grad_ests(self, obj, grasp_points, grasp_normals):
        tip_positions = self.get_tip_positions(offset_dir=grasp_normals, radius=0.009)
        if self.grad_ests is None:
            # tip_positions = grasp_points
            tip_positions = grasp_points.cuda() + grasp_normals.cuda() * 0.01
            nerf_tip_pos = grasp_utils.ig_to_nerf(tip_positions.cpu().detach().numpy())
            _, grad_ests = nerf_utils.est_grads_vals(
                obj.model,
                nerf_tip_pos.reshape(1, -1, 3).cuda(),
                sigma=5e-3,
                num_samples=1000,
                method="gaussian",
            )
            grad_ests = grad_ests.reshape(3, 3).float()
            grad_ests /= grad_ests.norm(dim=1, keepdim=True)
            # rotate surface normals to IG world coordinates
            grad_ests = grasp_utils.nerf_to_ig(grad_ests.cpu().detach().numpy())
            logging.info("grad_ests: %s", grad_ests)
            logging.info("grasp_pts: %s", grasp_points)
            logging.info("tip_positions: %s", tip_positions)
            self.grad_ests = grad_ests
            print("visualizing grad ests")
        else:
            tip_positions = self.get_tip_positions(
                offset_dir=self.grad_ests, radius=0.009
            )
        # Assuming that contact normals stay the same, i.e. static contacts
        grad_ests = self.grad_ests
        if self.added_lines:
            self.gym.clear_lines(self.viewer)
        ig_viz_utils.visualize_grasp_normals(
            self.gym, self.viewer, self.env, tip_positions, -grad_ests
        )
        self.added_lines = True
        return grad_ests

    def control(self, state_dict, obj):
        timestep = state_dict.get("timestep")
        timestep = timestep % 1000

        mode = "off"
        if timestep < 0:
            mode = "off"  # Box needs this to get ot graps position - bear can't have it
        elif timestep < 100:
            mode = "safe"
        elif timestep < 200:
            mode = "pos"
        elif timestep < 300:
            mode = "vel"
        else:
            mode = "up"

        if timestep % 10 == 0:
            logging.info("%s, %s", timestep, mode)

        # safe_pos = torch.tensor( [[ 0.0,  0.10, 0.05,],
        #                          [ 0.05,-0.10, 0.05,],
        #                          [-0.05,-0.10, 0.05,]])
        if not obj.nerf_loaded:
            obj.load_nerf_model()
        if obj.gt_mesh is None:
            obj.load_trimesh()

        grasp_points = state_dict.get("grasp_points")
        grasp_normals = state_dict.get("grasp_normals")
        grasp_points, grasp_normals = self.assign_closest_fingertips(
            grasp_points, grasp_normals
        )
        tip_positions = self.get_tip_positions()

        # safe grasp point moves slightly off of contact surface
        safe_pos = grasp_points - grasp_normals * self.norm_start_offset

        if mode == "off":
            pass
        if mode == "safe":
            self.position_control(safe_pos)
        if mode == "pos":
            self.position_control(grasp_points)
        if mode == "vel":
            # move radially in along xy plane
            # normal = - (grasp_points - torch.mean(grasp_points, axis=0))
            # normal[:, -1] = 0
            self.vel_control_force_limit(grasp_points, grasp_normals)
            if self.added_lines:
                self.gym.clear_lines(self.viewer)
            ig_viz_utils.visualize_grasp_normals(
                self.gym, self.viewer, self.env, tip_positions, -grasp_normals
            )
            self.added_lines = True
        if mode == "up":
            if not self.use_true_normals:
                in_normals = self.get_grad_ests(obj, grasp_points, grasp_normals)
            else:
                in_normals = grasp_normals
            pos_target = torch.tensor([0, 0, self.target_height])
            if self.added_lines:
                self.gym.clear_lines(self.viewer)
            ig_viz_utils.visualize_grasp_normals(
                self.gym, self.viewer, self.env, tip_positions, -grasp_normals
            )
            self.added_lines = True
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

    def position_control(self, p_des):
        # Build container for joint torques.
        applied_torque = torch.zeros((9))

        # Build list of joint indices to query joint angles + vels.
        dof_idx = []
        for finger_index, finger_pos in enumerate([0, 120, 240]):
            robot_dof_names = [
                f"finger_base_to_upper_joint_{finger_pos}",
                f"finger_upper_to_middle_joint_{finger_pos}",
                f"finger_middle_to_lower_joint_{finger_pos}",
            ]

            dof_idx += [self.dofs[dof_name] for dof_name in robot_dof_names]

        # Unpack joint angles + velocities.
        q, dq = self.dof_states[dof_idx, 0], self.dof_states[dof_idx, 1]

        # Cast everything to numpy for compatibility with pin.
        q = q.cpu().numpy()
        dq = dq.cpu().numpy()
        p_des = p_des.reshape(-1).cpu().numpy()

        # Call position controller.
        joint_torques = pos_control.get_joint_torques(
            p_des.reshape(-1),
            self.pin_model,
            self.pin_data,
            q,
            dq,
            self.pos_control_config,
        )

        # Fill in torque vector; cast back to torch.
        applied_torque[dof_idx] = torch.from_numpy(joint_torques).to(applied_torque)

        # Set joint torques in simulator.
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
        # pos = obj.rb_states[0, 0:3]
        quat = obj.rb_states[0, 3:7]
        vel = obj.rb_states[0, 7:10]
        angular_vel = obj.rb_states[0, 10:13]

        quat = Quaternion.fromWLast(quat)
        target_quat = Quaternion.Identity()

        # cg_pos = pos
        cg_pos = obj.get_CG()  # thoughts it eliminates the pendulum effect? possibly?
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
        # target_force = object_weight_comp - 0.2 * pos_error - 0.10 * vel
        # target_torque = (
        #     -0.4 * (quat @ target_quat.T).to_tangent_space() - 0.01 * angular_vel
        # )

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

        # not necessary for box - changes tunning parameters
        # makes the grasp points and normals follow the tip positions and object rotation
        # TODO grasp points should in object frame
        grasp_points = tip_positions - cg_pos

        in_normal = torch.stack([quat.rotate(x) for x in in_normal], axis=0)
        global_forces, success = force_opt.calculate_grip_forces(
            grasp_points, in_normal, target_force, target_torque, mu=obj.mu
        )
        if not success:
            logging.warning("solve failed, maintaining previous forces")
            global_forces = (
                self.previous_global_forces
            )  # will fail if we failed solve on first iteration
            assert global_forces is not None
        else:
            self.previous_global_forces = global_forces

        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.apply_fingertip_forces(global_forces)
        return global_forces, target_force, target_torque, success
