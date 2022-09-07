from isaacgym import gymapi, gymtorch, torch_utils
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock
from nerf_grasping import grasp_utils, nerf_utils, config
from nerf_grasping.sim import ig_utils, ig_objects, ig_robot, ig_viz_utils
from nerf_grasping.quaternions import Quaternion
from typing import Union, Tuple, Optional

import dcargs
import os
import numpy as np
import torch
import trimesh


# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

root_dir = os.path.dirname(os.path.abspath(__file__))
asset_dir = f"{root_dir}/assets"


class FingertipEnv:
    def __init__(self, exp_config: config.EvalExperiment):
        self.args = ig_utils.parse_arguments(description="Trifinger test")
        self.setup_gym()
        if exp_config.visualize:
            self.setup_viewer()
        else:
            self.viewer = None
        self.setup_envs(exp_config)
        self.gym.prepare_sim(self.sim)
        self.image_idx = 0

    def setup_gym(self):
        self.gym = gymapi.acquire_gym()

        self.sim = ig_utils.setup_sim(self.gym)
        self.env = ig_utils.setup_env(self.gym, self.sim)
        return self.gym, self.sim, self.env

    def setup_envs(self, exp_config, grasp_vars=None):
        self.robot = ig_robot.FingertipRobot(exp_config.robot_config)
        self.robot.setup_gym(self.gym, self.sim, self.env, grasp_vars)

        # Creates object and loads nerf and object mesh
        self.obj = ig_objects.load_object(exp_config)
        self.obj.setup_gym(self.gym, self.sim, self.env)
        self.obj.load_trimesh()
        ig_utils.setup_stage(self.gym, self.sim, self.env)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            -1, 13
        )
        self.robot.setup_tensors()
        self.obj.setup_tensors()
        robot_start_poses = [
            self.gym.get_actor_rigid_body_states(self.env, actor, gymapi.STATE_ALL)[
                "pose"
            ]
            for actor in self.robot.actors
        ]
        object_start_pose = self.gym.get_actor_rigid_body_states(
            self.env, self.obj.actor, gymapi.STATE_ALL
        )["pose"]
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
        self.robot_init_state = torch.stack(
            [
                torch.tensor(
                    [
                        start_pose["p"]["x"][0],
                        start_pose["p"]["y"][0],
                        start_pose["p"]["z"][0],
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
                for start_pose in robot_start_poses
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

        self.overhead_camera_handle = self.gym.create_camera_sensor(env, camera_props)
        self.gym.set_camera_location(
            self.overhead_camera_handle,
            env,
            gymapi.Vec3(0, 0.001, 0.5),
            gymapi.Vec3(0, 0, 0.01),
        )

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

    def debug_grasp_visualization(self):
        if len(self.marker_handles) == 0:
            tip_positions = self.obj.grasp_points.cuda().reshape(3, 3)
            if not self.obj.nerf_loaded:
                self.obj.load_nerf_model()
            # get grasp points into nerf frame
            tip_positions = tip_positions + self.obj.grasp_normals.cuda() * 0.01
            nerf_tip_pos = grasp_utils.ig_to_nerf(tip_positions)
            _, grad_ests = nerf_utils.est_grads_vals(
                self.obj.model,
                nerf_tip_pos.reshape(1, 3, 3),
                sigma=5e-3,
                method="gaussian",
                num_samples=1000,
            )
            grad_ests = grad_ests.reshape(3, 3).float()
            grad_ests /= grad_ests.norm(dim=1, keepdim=True)
            # get normal estimates and gradient estimates back in IG world frame
            grad_ests = grasp_utils.nerf_to_ig(grad_ests.cpu().detach().numpy())
            self.grad_ests = grad_ests
            # self.visualize_grasp_normals(tip_positions, -grad_ests)
            # self.marker_handles += self.plot_circle(self.gym, self.env, self.sim, self.object)
            # densities = nerf_utils.nerf_densities(
            #     self.object.model, nerf_tip_pos.reshape(1, 3, 3)
            # )
            # densities = densities.cpu().detach().numpy() / 355
            # densities = densities.flatten()
        if len(self.marker_handles) == 0:
            tip_positions = self.obj.grasp_points.cpu().numpy().reshape(3, 3)
            colors = [[0, 1, 0]] * 3  # green colored markers
            # self.marker_handles = ig_viz_utils.visualize_markers(
            #     self.gym, self.env, self.sim, tip_positions, colors
            # )
            pos_offset = self.obj.rb_states[0, :3].cpu().numpy()
            rot_offset = None  # Quaternion.fromWLast(self.obj.rb_states[0, 3:7])
            gp, gn = ig_utils.get_mesh_contacts(
                self.obj.gt_mesh, tip_positions, pos_offset, rot_offset
            )
            if self.added_lines:
                self.gym.clear_lines(self.viewer)
            ig_viz_utils.visualize_grasp_normals(
                self.gym, self.viewer, self.env, gp, -gn
            )
            self.added_lines = True
            colors = [[1, 0, 0]] * 3  # red colored markers
            self.marker_handles += ig_viz_utils.visualize_markers(
                self.gym, self.env, self.sim, gp, colors
            )

    def reset(self, grasp_vars=None):

        # reset_actor sets actor rigid body states
        self.robot.reset_actor(grasp_vars)
        self.obj.reset_actor()
        # reset object state tensor
        object_idx = self.gym.get_actor_index(
            self.env, self.obj.actor, gymapi.DOMAIN_SIM
        )
        self.root_state_tensor[object_idx] = self.object_init_state.clone()
        actor_indices = []
        for i, actor in enumerate(self.robot.actors):
            actor_idx = self.gym.get_actor_index(self.env, actor, gymapi.DOMAIN_SIM)
            actor_indices.append(actor_idx)
        self.root_state_tensor[actor_indices] = self.robot_init_state.clone()
        actor_indices = torch_utils.to_torch(
            actor_indices + [object_idx], dtype=torch.long, device="cpu"
        ).to(torch.int32)
        assert self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices),
        ), "resetting actor_root_state_tensor failed"
        # step_gym calls gym.simulate, then refreshes tensors
        self.step_gym()

        self.robot.reset_actor(grasp_vars)
        self.obj.reset_actor()
        # step_gym calls gym.simulate, then refreshes tensors
        self.step_gym()
        self.image_idx = 0


def run_robot_control(exp_config, grasp_vars):
    env = FingertipEnv(exp_config)
    count = 0
    while not env.gym.query_viewer_has_closed(env.viewer):
        try:
            count += 1
            env.step_gym()
            env.robot.control(count, env.object, grasp_vars)
        finally:
            pass
    print("closed!")


def main():
    exp_config = dcargs.cli(config.EvalExperiment)
    run_robot_control(exp_config)
