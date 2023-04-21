# isort:skip_file
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

from isaacgym import gymapi, gymutil, gymtorch
import os
import numpy as np
import torch

# import mathutils
from PIL import Image
import math

from nerf_grasping import grasp_utils, nerf_utils
from nerf_grasping.sim import (
    ig_utils,
    ig_objects,
    ig_robot,
    ig_viz_utils,
    acronym_objects,
)
import trimesh

from nerf_grasping.quaternions import Quaternion
import argparse


# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

root_dir = os.path.dirname(os.path.abspath(__file__))
asset_dir = f"{root_dir}/assets"
CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH = 400, 400
CAMERA_HORIZONTAL_FOV_DEG = 35.0
CAMERA_VERTICAL_FOV_DEG = (
    CAMERA_IMG_HEIGHT / CAMERA_IMG_WIDTH
) * CAMERA_HORIZONTAL_FOV_DEG

RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)

NUM_FINGERS = 3
NUM_XYZ = 3


def get_mesh_contacts(gt_mesh, grasp_points, pos_offset=None, rot_offset=None):
    if pos_offset is not None:
        # project grasp_points into object frame
        grasp_points -= pos_offset
    if rot_offset is not None:
        grasp_points = np.stack([rot_offset.rotate(gp) for gp in grasp_points])
    points, _, index = trimesh.proximity.closest_point(gt_mesh, grasp_points)
    # grasp normals follow convention that points into surface,
    # trimesh computes normals pointing out of surface
    grasp_normals = -gt_mesh.face_normals[index]
    if pos_offset is not None:
        # project back into world frame
        points += pos_offset
    if rot_offset is not None:
        grasp_normals = np.stack([rot_offset.T.rotate(x) for x in grasp_normals])
    return points, grasp_normals


def get_fixed_camera_transform(gym, sim, env, camera):
    # currently x+ is pointing down camera view axis - other degree of freedom is messed up
    # output will have x+ be optical axis, y+ pointing left (looking down camera) and z+ pointing up
    t = gym.get_camera_transform(sim, env, camera)
    pos = torch.tensor([t.p.x, t.p.y, t.p.z])
    quat = Quaternion.fromWLast([t.r.x, t.r.y, t.r.z, t.r.w])

    x_axis = torch.tensor([1.0, 0, 0])
    # y_axis = torch.tensor([0, 1.0, 0])
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


class TriFingerEnv:
    def __init__(
        self,
        viewer=True,
        robot_type="trifinger",
        Obj=None,
        disable_gravity=False,
        save_cameras=False,
        **robot_kwargs,
    ):
        self.args = ig_utils.parse_arguments(description="Trifinger test")
        self.gym = gymapi.acquire_gym()
        self.robot_type = robot_type

        self.setup_sim(disable_gravity)
        self.setup_envs(robot_type=robot_type, Obj=Obj, **robot_kwargs)

        if viewer:
            self.setup_viewer()
        else:
            self.viewer = None

        if save_cameras:
            self.setup_cameras(self.env)
        else:
            self.camera_handles = []

        self.marker_handles = []
        self.gym.prepare_sim(self.sim)
        self.image_idx = 0

    def setup_sim(self, disable_gravity):
        # only tested with this one
        assert self.args.physics_engine == gymapi.SIM_PHYSX

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 6000.0  # HACK TODO

        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = (
            gymapi.Vec3(0.0, 0.0, -9.8)
            if not disable_gravity
            else gymapi.Vec3(0.0, 0.0, 0.0)
        )

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

    def setup_envs(self, robot_type, Obj, **robot_kwargs):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        self.gym.add_ground(self.sim, plane_params)

        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        env = self.gym.create_env(self.sim, env_lower, env_upper, 0)

        self.env = env  # used only when there is one env
        self.envs = [env]

        if robot_type == "trifinger":
            self.robot = ig_robot.FingertipRobot(
                self.gym, self.sim, self.env, **robot_kwargs
            )
        elif robot_type == "spheres":
            self.robot = ig_robot.FingertipRobot(
                self.gym, self.sim, self.env, **robot_kwargs
            )
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
        asset_options.disable_gravity = False
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

        # position outside stage
        cam_pos = gymapi.Vec3(0.7, 0.175, 0.6)
        # position above banana
        cam_pos = gymapi.Vec3(0.1, 0.02, 0.4)
        cam_target = gymapi.Vec3(0, 0, 0.2)
        self.gym.viewer_camera_look_at(self.viewer, self.env, cam_pos, cam_target)

    def setup_cameras(self, env):
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = CAMERA_HORIZONTAL_FOV_DEG
        camera_props.width = CAMERA_IMG_WIDTH
        camera_props.height = CAMERA_IMG_HEIGHT

        # generates cameara positions along rings around object
        heights = [0.1, 0.3, 0.25, 0.35]
        distances = [0.05, 0.125, 0.3, 0.3]
        counts = [56, 104, 96, 1]
        target_z = [0.0, 0.1, 0.0, 0.1]

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
            if overwrite:
                shutil.rmtree(path)
            elif input("Clear it before continuing? [y/N]:").lower() == "y":
                shutil.rmtree(path)

        path.mkdir()
        return path

    def save_viewer_frame(self, save_dir, save_freq=10):
        """Saves frame from viewer to"""
        self.gym.render_all_camera_sensors(self.sim)

        path = self.setup_save_dir(save_dir)
        if self.image_idx % save_freq == 0:
            self.gym.write_viewer_image_to_file(
                self.viewer, str(path / f"img{self.image_idx}.png")
            )
        self.image_idx += 1

    def save_single_image(self, path, ii, camera_handle, numpy_depth=False):
        print(f"saving camera {ii}")

        color_image = self.gym.get_camera_image(
            self.sim, self.env, camera_handle, gymapi.IMAGE_COLOR
        )
        color_image = color_image.reshape(CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH, -1)
        Image.fromarray(color_image).save(path / f"col_{ii}.png")

        segmentation_image = self.gym.get_camera_image(
            self.sim, self.env, camera_handle, gymapi.IMAGE_SEGMENTATION
        )
        segmentation_image = segmentation_image == ig_objects.OBJ_SEGMENTATION_ID
        segmentation_image = (
            segmentation_image.reshape(CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH) * 255
        ).astype(np.uint8)
        Image.fromarray(segmentation_image).convert("L").save(path / f"seg_{ii}.png")

        depth_image = self.gym.get_camera_image(
            self.sim, self.env, camera_handle, gymapi.IMAGE_DEPTH
        )
        # distance in units I think
        depth_image = -depth_image.reshape(CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH)
        if numpy_depth:
            np.save(path / f"dep_{ii}.npy", depth_image)
        else:
            depth_image = (np.clip(depth_image, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(depth_image).convert("L").save(path / f"dep_{ii}.png")

        pos, quat = get_fixed_camera_transform(
            self.gym, self.sim, self.env, camera_handle
        )

        with open(path / f"pos_xyz_quat_xyzw_{ii}.txt", "w+") as f:
            data = [*pos.tolist(), *quat.q[1:].tolist(), quat.q[0].tolist()]
            json.dump(data, f)

    def save_images(self, folder, overwrite=False):
        self.gym.render_all_camera_sensors(self.sim)

        path = self.setup_save_dir(folder, overwrite)

        for ii, camera_handle in enumerate(self.camera_handles):
            self.save_single_image(path, ii, camera_handle)

        self.save_single_image(
            path, "overhead", self.overhead_camera_handle, numpy_depth=True
        )

    def create_train_val_test_split(self, folder, train_frac, val_frac):
        num_imgs = len(self.camera_handles)
        num_train = int(train_frac * num_imgs)
        num_val = int(val_frac * num_imgs)
        num_test = num_imgs - num_train - num_val
        print(f"num_imgs = {num_imgs}")
        print(f"num_train = {num_train}")
        print(f"num_val = {num_val}")
        print(f"num_test = {num_test}")
        print()

        img_range = np.arange(num_imgs)

        np.random.shuffle(img_range)
        train_range = img_range[:num_train]
        test_range = img_range[num_train : (num_train + num_test)]
        val_range = img_range[(num_train + num_test) :]

        self._create_one_split(
            split_name="train", split_range=train_range, folder=folder
        )
        self._create_one_split(split_name="val", split_range=val_range, folder=folder)
        self._create_one_split(split_name="test", split_range=test_range, folder=folder)

    def _create_one_split(self, split_name, split_range, folder):
        import scipy

        json_dict = {
            "camera_angle_x": math.radians(CAMERA_HORIZONTAL_FOV_DEG),
            "camera_angle_y": math.radians(CAMERA_VERTICAL_FOV_DEG),
            "frames": [],
        }
        for ii in split_range:
            pose_file = os.path.join(folder, f"pos_xyz_quat_xyzw_{ii}.txt")
            with open(pose_file) as file:
                raw_pose_str = file.readline()[1:-1]  # Remove brackets
                pose = np.fromstring(raw_pose_str, sep=",")

                transform_mat = np.eye(4)
                pos, quat = pose[:3], pose[-4:]
                R = scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()
                R = (
                    R
                    @ scipy.spatial.transform.Rotation.from_euler(
                        "YZ", [-np.pi / 2, -np.pi / 2]
                    ).as_matrix()
                )
                transform_mat[:3, :3] = R
                transform_mat[:3, -1] = pos

                source_img = "col_" + str(ii)

                new_folder = os.path.join(folder, split_name)
                os.makedirs(new_folder, exist_ok=True)

                source_img = os.path.join(folder, f"col_{ii}.png")
                target_img = os.path.join(new_folder, f"{ii}.png")
                shutil.copyfile(source_img, target_img)

                # Remove the first part of the path
                target_img_split = target_img.split("/")
                target_img = os.path.join(
                    *target_img_split[target_img_split.index(split_name) :]
                )

                json_dict["frames"].append(
                    {
                        "transform_matrix": transform_mat.tolist(),
                        "file_path": os.path.splitext(target_img)[
                            0
                        ],  # Exclude ext because adds it in load
                    }
                )

        with open(
            os.path.join(folder, f"transforms_{split_name}.json"), "w"
        ) as outfile:
            outfile.write(json.dumps(json_dict))

    def refresh_tensors(self):
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def step_gym(self):
        print("Forcing object to be in place")
        print(f"self.object.rb_states.shape = {self.object.rb_states.shape}")
        self.object.rb_states[0, 0] = self.object.object_start_pos.x
        self.object.rb_states[0, 1] = self.object.object_start_pos.y
        self.object.rb_states[0, 2] = self.object.object_start_pos.z

        for i in range(3, 13):
            if i == 6:
                self.object.rb_states[0, i] = 1.0
            else:
                self.object.rb_states[0, i] = 0.0
        self.gym.set_rigid_body_state_tensor(self.sim, gymtorch.unwrap_tensor(self.object.rb_states_all))

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.step_graphics(self.sim)
        if self.viewer is not None:
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        self.refresh_tensors()
        if self.object.gt_mesh is None:
            self.object.load_trimesh()

    def draw_acronym_grasps(self):
        if not hasattr(self.object, "acronym_file"):
            return

        # Clear previous lines
        self.gym.clear_lines(self.viewer)

        # Read in grasp transforms and successes
        import h5py

        assumed_acronym_root = "/juno/u/tylerlum/github_repos/acronym/data/grasps"
        acronym_filepath = os.path.join(assumed_acronym_root, self.object.acronym_file)
        acronym_data = h5py.File(acronym_filepath, "r")
        grasp_transforms = np.array(acronym_data["grasps/transforms"])
        grasp_successes = np.array(
            acronym_data["grasps/qualities/flex/object_in_gripper"]
        )

        num_grasps = 3
        successful_grasp_transforms = grasp_transforms[grasp_successes == 1][
            :num_grasps
        ]
        failed_grasp_transforms = grasp_transforms[grasp_successes == 0][:num_grasps]

        # Get transformation matrix from object frame to world frame
        import scipy

        object_to_world_transform = np.eye(4)
        pos, quat = self.object.position, self.object.orientation
        R = scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()
        object_to_world_transform[:3, :3] = R
        object_to_world_transform[:3, -1] = pos

        print(f"Drawing {num_grasps} successful and failed grasps")
        for grasp_tranforms, color in [
            (successful_grasp_transforms, GREEN),
            (failed_grasp_transforms, RED),
        ]:
            sphere = gymutil.WireframeSphereGeometry(
                radius=0.002, num_lats=10, num_lons=10, color=color
            )
            for grasp_transform in grasp_tranforms:
                # Get left and right tip positions from transform (in object frame)
                raw_left_tip = [4.10000000e-02, -7.27595772e-12, 1.12169998e-01]
                raw_right_tip = [-4.10000000e-02, -7.27595772e-12, 1.12169998e-01]
                raw_left_knuckle = [4.10000000e-02, -7.27595772e-12, 6.59999996e-02]
                raw_right_knuckle = [-4.10000000e-02, -7.27595772e-12, 6.59999996e-02]
                raw_hand_origin = [0.0, 0.0, 0.0]

                left_tip = (grasp_transform @ np.array([*raw_left_tip, 1.0]))[:3]
                right_tip = (grasp_transform @ np.array([*raw_right_tip, 1.0]))[:3]
                left_knuckle = (grasp_transform @ np.array([*raw_left_knuckle, 1.0]))[
                    :3
                ]
                right_knuckle = (grasp_transform @ np.array([*raw_right_knuckle, 1.0]))[
                    :3
                ]
                hand_origin = (grasp_transform @ np.array([*raw_hand_origin, 1.0]))[:3]

                # Transform to world frame
                left_tip = (object_to_world_transform @ np.array([*left_tip, 1.0]))[:3]
                right_tip = (object_to_world_transform @ np.array([*right_tip, 1.0]))[
                    :3
                ]
                left_knuckle = (
                    object_to_world_transform @ np.array([*left_knuckle, 1.0])
                )[:3]
                right_knuckle = (
                    object_to_world_transform @ np.array([*right_knuckle, 1.0])
                )[:3]
                hand_origin = (
                    object_to_world_transform @ np.array([*hand_origin, 1.0])
                )[:3]

                # Draw spheres at tips
                # Draw lines from tips to knuckles, knuckles to knuckles, and between knuckles to hand origin
                left_tip_pose = gymapi.Transform(gymapi.Vec3(*left_tip), r=None)
                right_tip_pose = gymapi.Transform(gymapi.Vec3(*right_tip), r=None)
                gymutil.draw_lines(
                    geom=sphere,
                    gym=self.gym,
                    viewer=self.viewer,
                    env=self.env,
                    pose=left_tip_pose,
                )
                gymutil.draw_lines(
                    geom=sphere,
                    gym=self.gym,
                    viewer=self.viewer,
                    env=self.env,
                    pose=right_tip_pose,
                )
                gymutil.draw_line(
                    p1=gymapi.Vec3(*left_tip),
                    p2=gymapi.Vec3(*left_knuckle),
                    color=gymapi.Vec3(*color),
                    gym=self.gym,
                    viewer=self.viewer,
                    env=self.env,
                )
                gymutil.draw_line(
                    p1=gymapi.Vec3(*right_tip),
                    p2=gymapi.Vec3(*right_knuckle),
                    color=gymapi.Vec3(*color),
                    gym=self.gym,
                    viewer=self.viewer,
                    env=self.env,
                )
                gymutil.draw_line(
                    p1=gymapi.Vec3(*left_knuckle),
                    p2=gymapi.Vec3(*right_knuckle),
                    color=gymapi.Vec3(*color),
                    gym=self.gym,
                    viewer=self.viewer,
                    env=self.env,
                )
                btwn_knuckles = (left_knuckle + right_knuckle) / 2
                gymutil.draw_line(
                    p1=gymapi.Vec3(*btwn_knuckles),
                    p2=gymapi.Vec3(*hand_origin),
                    color=gymapi.Vec3(*color),
                    gym=self.gym,
                    viewer=self.viewer,
                    env=self.env,
                )

    def visualize_example_grasp(self):
        # if self.added_lines:
        # self.gym.clear_lines(self.viewer)
        # self._visualize_grasp_normals()
        self._visualize_grasp_points()

    def _visualize_grasp_normals(self):
        # Already visualized
        #  if len(self.marker_handles) > 0:
        #      return

        # Load nerf model if not already loaded
        print("Loading nerf model...")
        if not self.object.nerf_loaded:
            self.object.load_nerf_model()
        print("Done loading nerf model...")

        # Get example grasp positions in nerf frame (slightly move in normal dir)
        tip_positions = self.object.grasp_points.cuda().reshape(NUM_FINGERS, NUM_XYZ)
        tip_positions = tip_positions + self.object.grasp_normals.cuda() * 0.01
        nerf_tip_pos = grasp_utils.ig_to_nerf(tip_positions)

        # Visualize grasp normals
        from nerf_grasping import config

        GRAD_CONFIG = config.grad_configs["sim"]
        _, grad_ests = nerf_utils.est_grads_vals(
            nerf=self.object.model,
            grasp_points=nerf_tip_pos.reshape(1, NUM_FINGERS, NUM_XYZ),
            grad_config=GRAD_CONFIG,
        )
        grad_ests = grad_ests.reshape(NUM_FINGERS, NUM_XYZ).float()
        grad_ests /= grad_ests.norm(dim=1, keepdim=True)

        # get normal estimates and gradient estimates back in IG world frame
        grad_ests = grasp_utils.nerf_to_ig(grad_ests.cpu().detach().numpy())

        self.grad_ests = grad_ests

        ig_viz_utils.visualize_grasp_normals(
            self.gym, self.viewer, self.env, tip_positions, -grad_ests
        )
        self.marker_handles += ig_viz_utils.visualize_markers(
            self.gym, self.env, self.sim, tip_positions, [GREEN] * NUM_FINGERS
        )
        densities = nerf_utils.nerf_densities(
            self.object.model, nerf_tip_pos.reshape(1, NUM_FINGERS, NUM_XYZ)
        )
        densities = densities.cpu().detach().numpy() / 355
        densities = densities.flatten()

    def _visualize_grasp_points(self):
        # Already visualized
        # if len(self.marker_handles) > 0:
        #     return

        tip_positions = (
            self.object.grasp_points.cpu().numpy().reshape(NUM_FINGERS, NUM_XYZ)
        )
        colors = [GREEN] * NUM_FINGERS
        self.marker_handles = ig_viz_utils.visualize_markers(
            self.gym, self.env, self.sim, tip_positions, colors
        )

        pos_offset = self.object.position.cpu().numpy()
        rot_offset = None  # Quaternion.fromWLast(self.object.orientation)
        gp, gn = get_mesh_contacts(
            self.object.gt_mesh, tip_positions, pos_offset, rot_offset
        )
        ig_viz_utils.visualize_grasp_normals(self.gym, self.viewer, self.env, gp, -gn)
        colors = [RED] * NUM_FINGERS
        self.marker_handles += ig_viz_utils.visualize_markers(
            self.gym, self.env, self.sim, gp, colors
        )

    def reset(self, grasp_vars=None):
        # reset object after robot actor
        self.robot.reset_actor(grasp_vars=grasp_vars)

        # reset object actor
        self.object.reset_actor()

        self.refresh_tensors()
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.step_gym()

        self.image_idx = 0


def get_nerf_training_data(Obj, num_steps_before_collecting, viewer, overwrite):
    tf = TriFingerEnv(
        viewer=viewer, robot_type="", Obj=Obj, disable_gravity=True, save_cameras=True
    )
    for _ in range(num_steps_before_collecting):
        tf.step_gym()
        if Obj is not None:
            print(f"tf.object.position = {tf.object.position}")
            print(f"tf.object.orientation = {tf.object.orientation}")
            print()

    # name = "blank" if Obj is None else Obj.name
    save_folder = "./torch-ngp/data/isaac_" + Obj.name
    tf.save_images(save_folder, overwrite=overwrite)
    tf.create_train_val_test_split(save_folder, train_frac=0.8, val_frac=0.1)


def visualize_acronym_grasps(Obj):
    tf = TriFingerEnv(viewer=True, robot_type="", Obj=Obj, save_cameras=True)
    for _ in range(500):
        tf.step_gym()
        tf.draw_acronym_grasps()


def visualize_example_grasp(Obj):
    tf = TriFingerEnv(viewer=True, robot_type="", Obj=Obj, save_cameras=True)
    for _ in range(500):
        tf.step_gym()
        tf.visualize_example_grasp()


def run_robot_control(viewer, Obj, robot_type, **robot_kwargs):
    tf = TriFingerEnv(viewer=viewer, robot_type=robot_type, Obj=Obj, **robot_kwargs)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default="Banana")
    parser.add_argument("--get_nerf_training_data", action="store_true")
    parser.add_argument("--run_robot_control", action="store_true")
    parser.add_argument("--visualize_acronym_grasps", action="store_true")
    parser.add_argument("--visualize_example_grasp", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--num_steps_before_collecting", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80)

    if (
        sum(
            [
                args.get_nerf_training_data,
                args.run_robot_control,
                args.visualize_acronym_grasps,
                args.visualize_example_grasp,
            ]
        )
        != 1
    ):
        raise ValueError(
            "Must specify only one of --get_nerf_training_data, --run_robot_control, --visualize_acronym_grasps, --visualize_example_grasp"
        )

    # Object
    if hasattr(ig_objects, args.obj):
        print(f"Found object {args.obj} in ig_objects")
        Obj = getattr(ig_objects, args.obj)
    elif hasattr(acronym_objects, args.obj):
        print(f"Found object {args.obj} in acronym_objects")
        Obj = getattr(acronym_objects, args.obj)
    else:
        raise ValueError(f"Object {args.obj} not found")
    print("Obj", Obj.name, Obj().gt_mesh.extents)

    if args.get_nerf_training_data:
        get_nerf_training_data(
            Obj,
            num_steps_before_collecting=args.num_steps_before_collecting,
            viewer=args.viewer,
            overwrite=args.overwrite,
        )

    elif args.run_robot_control:
        run_robot_control(
            viewer=args.viewer,
            Obj=Obj,
            robot_type="trifinger",
            use_nerf_grasping=False,
            use_residual_dirs=True,
            use_true_normals=False,
            use_grad_est=True,
            metric="psv",
        )

    elif args.visualize_acronym_grasps:
        visualize_acronym_grasps(
            Obj=Obj,
        )

    elif args.visualize_example_grasp:
        visualize_example_grasp(
            Obj=Obj,
        )

    else:
        raise ValueError(
            "Must specify one of --get_nerf_training_data or --run_robot_control"
        )
