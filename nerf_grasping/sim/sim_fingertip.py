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
    def __init__(
        self,
        exp_config: config.EvalExperiment,
        init_grasp_vars=Tuple[torch.Tensor, torch.Tensor],
    ):
        self.args = ig_utils.parse_arguments(description="Trifinger test")
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

    def setup_robot_obj(self, exp_config, grasp_vars=None):
        # Creates robot
        self.robot = ig_robot.FingertipRobot(exp_config.robot_config)
        self.robot.setup_gym(self.gym, self.sim, self.env, grasp_vars)

        # Creates object and loads nerf and object mesh
        self.obj = ig_objects.load_object(exp_config)
        self.obj.setup_gym(self.gym, self.sim, self.env)
        self.obj.load_trimesh()
        ig_utils.setup_stage(self.gym, self.sim, self.env)

        # Loads mesh, checking if EvalExperiment using nerf grasps
        if isinstance(exp_config.model_config, config.Nerf):
            self.obj.load_nerf_model()
            self.mesh = None
        else:
            mesh_path = config.mesh_file(exp_config)
            self.mesh = trimesh.load(mesh_path)

        # Create root state tensor for resetting env
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

        # Get actor indices to reset actor_root_tensor
        actor_indices = []
        for a_handle in self.robot.actors + [self.obj.actor]:
            actor_indices.append(
                self.gym.get_actor_index(self.env, a_handle, gymapi.DOMAIN_SIM)
            )
        self.actor_indices = torch_utils.to_torch(
            actor_indices,
            dtype=torch.long,
            device="cpu",
        )
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
            gp, gn, _ = ig_utils.get_mesh_contacts(
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

    def reset_actors(self, grasp_vars):
        self.fail_count = 0
        # reset_actor sets actor rigid body states
        self.robot.reset_actor(grasp_vars)
        self.obj.reset_actor()
        # reset object and robot state tensor
        object_idx = self.actor_indices[-1]
        self.root_state_tensor[object_idx] = self.object_init_state.clone()
        self.root_state_tensor[self.actor_indices[:3]] = self.robot_init_state.clone()
        actor_indices = self.actor_indices.to(torch.int32)
        assert self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices),
        ), "resetting actor_root_state_tensor failed"
        # step_gym calls gym.simulate, then refreshes tensors
        self.step_gym()

        # optionally do a second reset?
        # self.robot.reset_actor(grasp_vars)
        # self.obj.reset_actor()
        # # step_gym calls gym.simulate, then refreshes tensors
        # self.step_gym()
        # self.image_idx = 0

    def control(self, mode, grasp_vars):
        grasp_points, grasp_normals = grasp_vars
        succ = True
        closest_points = ig_utils.closest_point(
            grasp_points, grasp_points + grasp_normals, self.robot.position
        )
        if mode == "reach":
            # position control to reach contact points
            f = self.robot.position_control(grasp_points)
            pos_err = grasp_points - self.robot.position
        elif mode == "grasp":
            # position + velocity control to grasp object
            f = self.robot.grasping_control(closest_points, grasp_normals)
            pos_err = closest_points - self.robot.position
        elif mode == "lift":
            # grasp force optimization
            mode = "lift"
            closest_points[:, 2] = self.obj.position[2] + 0.005
            poos_err = closest_points - self.robot.position
            contact_pts = self.robot.get_contact_points(grasp_normals)
            if self.mesh is None:
                # get estimated normal once
                ge = self.robot.get_grad_ests(self.obj, contact_pts)
            else:
                gp, ge, _ = ig_utils.get_mesh_contacts(
                    self.mesh,
                    contact_pts,
                    pos_offset=self.obj.position,
                    rot_offset=self.obj.orientation,
                )
                ge = torch.tensor(ge, dtype=torch.float32)
            f, _, _, succ = self.robot.object_pos_control(
                self.robot,
                self.obj,
                ge,
            )

            if not succ:
                self.fail_count += 1
        self.robot.apply_fingertip_forces(f)
        self.step_gym()
        net_obj_force = self.gym.get_rigid_contact_forces(self.sim)[self.obj.actor]

        state = dict(
            mode=mode,
            pos_err=pos_err,
            velocity=self.robot.velocity,
            force=f,
            force_mag=f.norm(dim=1),
            net_obj_force=net_obj_force,
            grasp_opt_success=succ,
        )
        return state


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
