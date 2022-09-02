# isort:skip_file
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

from isaacgym import gymapi
import os
import numpy as np
import torch

# import mathutils
from PIL import Image

from nerf_grasping import grasp_utils, nerf_utils
from nerf_grasping.sim import ig_utils, ig_objects, ig_robot, ig_viz_utils
import trimesh

from nerf_grasping.quaternions import Quaternion

# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

root_dir = os.path.dirname(os.path.abspath(__file__))
asset_dir = f"{root_dir}/assets"


def get_mesh_contacts(gt_mesh, grasp_points, pos_offset=None, rot_offset=None):
    if pos_offset is not None:
        # project grasp_points into object frame
        grasp_points -= pos_offset
        grasp_points = np.stack([rot_offset.rotate(gp) for gp in grasp_points])
    points, _, index = trimesh.proximity.closest_point(gt_mesh, grasp_points)
    # grasp normals follow convention that points into surface,
    # trimesh computes normals pointing out of surface
    grasp_normals = -gt_mesh.face_normals[index]
    if pos_offset is not None:
        # project back into world frame
        points += pos_offset
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
        save_cameras=False,
        **robot_kwargs,
    ):
        self.args = ig_utils.parse_arguments(description="Trifinger test")
        self.gym = gymapi.acquire_gym()
        self.robot_type = robot_type

        self.setup_sim()
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
            self.robot = ig_robot.Robot(self.gym, self.sim, self.env, **robot_kwargs)
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

    def save_single_image(self, path, ii, camera_handle, numpy_depth=False):
        print(f"saving camera {ii}")

        color_image = self.gym.get_camera_image(
            self.sim, self.env, camera_handle, gymapi.IMAGE_COLOR
        )
        color_image = color_image.reshape(400, 400, -1)
        Image.fromarray(color_image).save(path / f"col_{ii}.png")

        segmentation_image = self.gym.get_camera_image(
            self.sim, self.env, camera_handle, gymapi.IMAGE_SEGMENTATION
        )
        segmentation_image = segmentation_image == 2
        segmentation_image = (segmentation_image.reshape(400, 400) * 255).astype(
            np.uint8
        )
        Image.fromarray(segmentation_image).convert("L").save(path / f"seg_{ii}.png")

        depth_image = self.gym.get_camera_image(
            self.sim, self.env, camera_handle, gymapi.IMAGE_DEPTH
        )
        # distance in units I think
        depth_image = -depth_image.reshape(400, 400)
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

        for i, camera_handle in enumerate(self.camera_handles[:-1]):
            print(f"saving camera {i}")

            color_image = self.gym.get_camera_image(
                self.sim, self.env, camera_handle, gymapi.IMAGE_COLOR
            )
            color_image = color_image.reshape(400, 400, -1)
            Image.fromarray(color_image).save(path / "train" / f"col_{i}.png")

            pos, quat = get_fixed_camera_transform(
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
        if self.object.gt_mesh is None:
            self.object.load_trimesh()

        # if isinstance(self.robot, Mock):
        #     self.debug_grasp_visualization()

    def debug_grasp_visualization(self):
        if len(self.marker_handles) == 0:
            tip_positions = self.object.grasp_points.cuda().reshape(3, 3)
            if not self.object.nerf_loaded:
                self.object.load_nerf_model()
            # get grasp points into nerf frame
            tip_positions = tip_positions + self.object.grasp_normals.cuda() * 0.01
            nerf_tip_pos = grasp_utils.ig_to_nerf(tip_positions)
            _, grad_ests = nerf_utils.est_grads_vals(
                self.object.model,
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
            tip_positions = self.object.grasp_points.cpu().numpy().reshape(3, 3)
            colors = [[0, 1, 0]] * 3  # green colored markers
            # self.marker_handles = ig_viz_utils.visualize_markers(
            #     self.gym, self.env, self.sim, tip_positions, colors
            # )
            pos_offset = self.object.rb_states[0, :3].cpu().numpy()
            rot_offset = None  # Quaternion.fromWLast(self.object.rb_states[0, 3:7])
            gp, gn = get_mesh_contacts(
                self.object.gt_mesh, tip_positions, pos_offset, rot_offset
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
        # reset object after robot actor
        self.robot.reset_actor(grasp_vars=grasp_vars)
        # reset object actor
        self.object.reset_actor()
        self.refresh_tensors()
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.step_gym()

        self.image_idx = 0


def get_nerf_training(Obj, viewer):
    tf = TriFingerEnv(viewer=viewer, robot_type="", Obj=Obj, save_cameras=True)
    for _ in range(500):
        tf.step_gym()
        if Obj is not None:
            print(tf.object.rb_states[0, :7])

    name = "blank" if Obj is None else Obj.name
    tf.save_images("./torch-ngp/data/isaac_" + name, overwrite=False)


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
    # Obj = ig_objects.Box
    # Obj = ig_objects.TeddyBear
    # Obj = ig_objects.PowerDrill
    # Obj = ig_objects.Box
    Obj = ig_objects.BleachCleanser  # too big - put on side?
    # Obj = ig_objects.Spatula
    # Obj = ig_objects.Mug
    get_nerf_training(Obj, viewer=False)
    # run_robot_control(
    #     viewer=True,
    #     Obj=Obj,
    #     robot_type="trifinger",
    #     use_nerf_grasping=False,
    #     use_residual_dirs=True,
    #     use_true_normals=False,
    #     use_grad_est=True,
    #     metric="psv",
    # )
