from isaacgym import gymapi, gymtorch, torch_utils
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock
from nerf_grasping import grasp_utils, mesh_utils, nerf_utils, config
from nerf_grasping.sim import ig_utils, ig_objects, ig_robot, ig_viz_utils
from nerf_grasping.quaternions import Quaternion
from typing import Union, Tuple, Optional

import dcargs
import os
import numpy as np
import torch
import trimesh


# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
asset_dir = f"{root_dir}/assets"


class FingertipEnv:
    def __init__(
        self,
        exp_config: config.EvalExperiment,
        init_grasp_vars=Tuple[torch.Tensor, torch.Tensor],
    ):
        self.args = ig_utils.parse_arguments(description="Trifinger test")
        self.added_lines = False
        self.setup_gym()
        if exp_config.visualize:
            self.setup_viewer()
        else:
            self.viewer = None
        self.setup_robot_obj(exp_config)
        self.gym.prepare_sim(self.sim)
        self.image_idx = 0

    def setup_gym(self):
        self.gym = gymapi.acquire_gym()

        self.sim = ig_utils.setup_sim(self.gym)
        self.env = ig_utils.setup_env(self.gym, self.sim)
        return self.gym, self.sim, self.env

    def setup_robot_obj(self, exp_config):
        # Creates robot
        self.robot = ig_robot.FingertipRobot(exp_config.robot_config)
        self.robot.setup_gym(self.gym, self.sim, self.env)

        self.marker_handles = ig_viz_utils.visualize_markers(
            self.gym,
            self.env,
            self.sim,
            [np.zeros(3)] * 3,
            colors=[[1.0, 0.5, 0.0]] * 3,
            marker_handles=[],
        )

        # Creates object and loads nerf and object mesh
        self.obj = ig_objects.load_object(exp_config)
        self.obj.setup_gym(self.gym, self.sim, self.env)
        ig_utils.setup_stage(self.gym, self.sim, self.env)

        # Loads mesh, checking if EvalExperiment using nerf grasps
        if isinstance(exp_config.model_config, config.Nerf):
            self.obj.load_nerf_model()
            self.mesh = None
        else:
            if exp_config.level_set is None:
                mesh_path = f"{asset_dir}/objects/meshes/{self.obj.name}/textured.obj"
                raw_mesh_path = (
                    f"{asset_dir}/objects/raw_meshes/{self.obj.name}/textured.obj"
                )

                raw_mesh = trimesh.load(raw_mesh_path, force="mesh")
                print("Raw mesh extents:", raw_mesh.extents)
                print("GT mesh extents:", self.obj.gt_mesh.extents)
                print(f"LOADED MESH for estimating normals: {mesh_path}")
                self.mesh = self.obj.gt_mesh
            else:
                print(
                    f"LOADED MESH for estimating normals: {config.mesh_file(exp_config)}"
                )
                self.mesh = mesh_utils.get_mesh(exp_config, self.obj)

        # Create root state tensor for resetting env
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            -1, 13
        )
        self.robot.setup_tensors()
        self.obj.setup_tensors()
        self.actor_indices = None
        grasp_vars = (self.obj.grasp_points, self.obj.grasp_normals)

        # resets object and actor, sets robot start pose
        self.reset_actors(grasp_vars)

        # set object start pose, which is always reset to the same position
        object_start_pose = self.gym.get_actor_rigid_body_states(
            self.env, self.obj.actor, gymapi.STATE_ALL
        )["pose"]

        # Get actor indices to reset actor_root_tensor
        actor_indices = []
        for a_handle in self.robot.actors + self.marker_handles + [self.obj.actor]:
            actor_indices.append(
                self.gym.get_actor_index(self.env, a_handle, gymapi.DOMAIN_SIM)
            )
        self.actor_indices = torch_utils.to_torch(
            actor_indices,
            dtype=torch.long,
            device="cpu",
        )
        self.robot_idx = self.actor_indices[:3]
        self.marker_idx = self.actor_indices[3:6]
        self.object_idx = self.actor_indices[-1]
        self.object_init_state = torch.tensor(
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

    def setup_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        # self.robot.viewer = self.viewer
        assert self.viewer is not None

        # position outside stage
        cam_pos = gymapi.Vec3(0.7, 0.175, 0.6)
        # position above banana
        cam_pos = gymapi.Vec3(0.1, 0.02, 0.4)
        cam_target = gymapi.Vec3(0, 0, 0.2)
        self.gym.viewer_camera_look_at(self.viewer, self.env, cam_pos, cam_target)

    def setup_save_dir(self, folder, overwrite=False):
        path = Path(folder)

        if path.exists():
            print(path, "already exists!")
        else:
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

    def refresh_tensors(self):
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def step_gym(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.step_graphics(self.sim)
        if self.viewer is not None:
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        self.refresh_tensors()

    def debug_grasp_visualization(self, grasp_vars, net_obj_force):
        tip_positions = self.robot.get_contact_points(grasp_vars)
        gp, gn, _ = ig_utils.get_mesh_contacts(
            self.obj.gt_mesh, tip_positions, self.obj.position, self.obj.orientation
        )
        if self.added_lines:
            self.gym.clear_lines(self.viewer)
            self.added_lines = False
        ig_viz_utils.visualize_grasp_normals(
            self.gym, self.viewer, self.env, gp, -net_obj_force
        )
        self.added_lines = True

    def set_robot_init_state(self, grasp_vars):
        start_ftip_pos = self.robot.get_ftip_start_pos(grasp_vars)
        robot_start_poses = [
            self.gym.get_actor_rigid_body_states(self.env, actor, gymapi.STATE_ALL)[
                "pose"
            ]
            for actor in self.robot.actors
        ]
        self.robot_init_state = torch.stack(
            [
                torch.tensor(
                    [
                        grasp_pt[0],
                        grasp_pt[1],
                        grasp_pt[2],
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
                for grasp_pt, start_pose in zip(start_ftip_pos, robot_start_poses)
            ]
        )
        return

    def reset_actors(self, grasp_vars):
        # reset_actor sets actor rigid body states
        self.robot.reset_actor(grasp_vars)
        self.obj.reset_actor()
        if self.added_lines:
            self.gym.clear_lines(self.viewer)
            self.added_lines = False

        for i in range(50):
            self.step_gym()

        self.set_robot_init_state(grasp_vars)

        if self.actor_indices is not None:
            # reset object and robot state tensor
            self.root_state_tensor[self.object_idx] = self.object_init_state.clone()
            self.root_state_tensor[self.robot_idx] = self.robot_init_state.clone()
            # self.root_state_tensor[self.marker_idx] = 0.0
            actor_indices = self.actor_indices.to(torch.int32)
            assert self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(actor_indices),
                len(actor_indices),
            ), "resetting actor_root_state_tensor failed"

            # step_gym calls gym.simulate, then refreshes tensors
            for i in range(50):
                self.step_gym()

        # optionally do a second reset?
        # self.robot.reset_actor(grasp_vars)
        # self.obj.reset_actor()
        # # step_gym calls gym.simulate, then refreshes tensors
        # self.step_gym()
        # self.image_idx = 0

    def run_control(self, mode, grasp_vars, timestep):
        grasp_points, grasp_normals = grasp_vars  # IG frame.
        succ = True
        closest_points = ig_utils.closest_point(
            grasp_points, grasp_points + grasp_normals, self.robot.position
        )  # IG frame.
        target_obj_pos = np.array(
            [0, 0, self.obj.translation[-1] + self.robot.target_height]
        )  # IG frame.
        # get mesh
        if self.mesh is None or self.robot.use_true_normals:
            mesh = self.obj.gt_mesh
        else:
            mesh = self.mesh

        if mode == "reach":
            # position control to reach contact points
            f = self.robot.position_control(closest_points)
            pos_err = grasp_points - self.robot.position
        elif mode == "grasp":
            # position + velocity control to grasp object
            f = self.robot.grasping_control(closest_points, grasp_normals)
            pos_err = closest_points - self.robot.position
        elif mode == "lift":
            # grasp force optimization
            closest_points[:, 2] = self.obj.position[2]
            pos_err = closest_points - self.robot.position
            contact_pts = self.robot.get_contact_points(grasp_normals)

            if self.mesh is None and not self.robot.use_true_normals:
                quat = Quaternion.fromWLast(self.obj.orientation)
                contact_pts_obj_frame = np.stack(
                    [quat.T.rotate(x - self.obj.position) for x in contact_pts]
                )
                # get estimated normal once
                ge_ig_frame = (
                    self.robot.get_nerf_grad_ests(self.obj, contact_pts_obj_frame)
                    .cpu()
                    .float()
                )
                ig_viz_utils.visualize_grasp_normals(
                    self.gym, self.viewer, self.env, contact_pts, ge_ig_frame
                )
                # add markers to robot actors to fix reset_actors
                self.marker_handles = ig_viz_utils.visualize_markers(
                    self.gym,
                    self.env,
                    self.sim,
                    contact_pts,
                    colors=[[1.0, 0.5, 0.0]] * 3,
                    marker_handles=self.marker_handles,
                )
            else:
                contact_pts_obj_frame, ge_ig_frame, _ = ig_utils.get_mesh_contacts(
                    mesh,
                    contact_pts,
                    pos_offset=self.obj.position,
                    rot_offset=self.obj.orientation,
                )  # IG frame.
                ge_ig_frame = torch.tensor(
                    ge_ig_frame, dtype=torch.float32
                )  # IG frame.

            f, target_force, _, succ = self.robot.object_pos_control(
                self.obj, ge_ig_frame, target_obj_pos
            )
        else:
            f = torch.zeros((3, 3))  # no-op
            pos_err = closest_points - self.robot.position

        self.robot.apply_fingertip_forces(f)
        self.step_gym()
        state = {}

        # DEBUG PRINTS FOR CONTACT FORCES.
        # obj_handle = self.gym.get_actor_rigid_body_handle(self.env, self.obj.actor, 0)
        # self.gym.apply_body_forces(self.env, obj_handle, gymapi.Vec3(*torch.sum(f, dim=0).data))
        net_obj_force = self.gym.get_rigid_contact_forces(self.sim)[self.obj.actor]
        if mode == "lift":
            obj_pos_err = self.obj.position - target_obj_pos
            contact_pos_err = np.linalg.norm(
                contact_pts_obj_frame - closest_points.cpu().numpy(), axis=1
            )
            state = {
                "obj_pos_err-z": obj_pos_err[2],
                "obj_vel-z": self.obj.velocity[2],
                "max_contact_pos_err": contact_pos_err.max().item(),
            }

        self.gym.clear_lines(self.viewer)
        self.gym.draw_env_rigid_contacts(
            self.viewer, self.env, gymapi.Vec3(0.0, 0.0, 0.0), 1.0, True
        )
        ig_viz_utils.visualize_obj_com(self.gym, self.viewer, self.env, self.obj)
        mesh = self.mesh or self.obj.gt_mesh
        ig_viz_utils.visualize_mesh_bbox(
            self.gym,
            self.viewer,
            self.env,
            self.obj,
            mesh,
        )
        # print('asset forces:', self.obj.force_sensor.get_forces().force)

        state.update(
            dict(
                mode=mode,
                ftip_pos_err=pos_err,
                max_ftip_pos_err=pos_err.norm(dim=1).max(),
                ftip_vel=self.robot.velocity,
                ftip_forces=f,
                force_mag=f.norm(dim=1),
                net_obj_force=net_obj_force,
                grasp_opt_success=succ,
            )
        )
        return state


def get_mode(timestep):
    if timestep % 1000 < 50:
        return "reach"
    if timestep % 1000 < 200:
        return "grasp"
    if timestep % 1000 < 1000:
        return "lift"


def run_robot_control(exp_config, grasp_vars=None):
    env = FingertipEnv(exp_config)
    count = 0
    while not env.gym.query_viewer_has_closed(env.viewer):
        try:
            env.control(get_mode(count), grasp_vars)
            count += 1
        finally:
            pass
    print("closed!")


def main():
    exp_config = dcargs.cli(config.EvalExperiment)
    run_robot_control(exp_config)