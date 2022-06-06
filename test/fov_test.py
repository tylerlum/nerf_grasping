import os
import json
import math
import numpy as np

# import mathutils
from PIL import Image
from pathlib import Path
import shutil

from isaacgym import gymapi, gymutil  # , gymtorch

import matplotlib.pyplot as plt

# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs


class TestEnv:

    def __init__(self, viewer=True):
        self.args = gymutil.parse_arguments(description="Trifinger test", )
        self.gym = gymapi.acquire_gym()

        self.setup_sim()
        self.setup_envs()

        if viewer:
            self.setup_viewer()
        else:
            self.viewer = None

        self.gym.prepare_sim(self.sim)

    def setup_sim(self):
        # only tested with this one
        assert self.args.physics_engine == gymapi.SIM_PHYSX

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = dt = 1.0 / 60.0

        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = self.args.num_threads
        # sim_params.physx.use_gpu = self.args.use_gpu
        sim_params.physx.use_gpu = False

        # sim_params.use_gpu_pipeline = True
        sim_params.use_gpu_pipeline = False
        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            self.args.physics_engine,
            sim_params,
        )
        assert self.sim != None

        intensity = gymapi.Vec3(0.3, 0.3, 0.3)
        ambient = gymapi.Vec3(0.2, 0.2, 0.2)

        self.gym.set_light_parameters(self.sim, 0, intensity, ambient,
                                      gymapi.Vec3(0.5, 1, 1))
        self.gym.set_light_parameters(self.sim, 1, intensity, ambient,
                                      gymapi.Vec3(1, 0, 1))
        self.gym.set_light_parameters(self.sim, 2, intensity, ambient,
                                      gymapi.Vec3(0.5, -1, 1))
        self.gym.set_light_parameters(self.sim, 3, intensity, ambient,
                                      gymapi.Vec3(0, 0, 1))

    def setup_envs(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        self.gym.add_ground(self.sim, plane_params)

        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        env = self.gym.create_env(self.sim, env_lower, env_upper, 0)

        self.env = env  # used only when there is one env
        self.envs = [env]

        self.setup_object(env)
        self.setup_cameras(env)

    def setup_object(self, env):

        asset_options = gymapi.AssetOptions()
        asset_options.density = 10.0
        asset_options.disable_gravity = True

        # sphere_asset = self.gym.create_sphere(self.sim, 0.25 , asset_options)

        sphere_asset = self.gym.create_box(self.sim, 0.01, 1, 1, asset_options)
        quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.25 * np.pi)
        self.gym.create_actor(
            env,
            sphere_asset,
            gymapi.Transform(p=gymapi.Vec3(0, 0.0, 1), r=quat),
            "sphere",
            0,
            0,
        )

        # sphere_asset = self.gym.create_capsule(self.sim, 0.05, 1 , asset_options)
        # sphere_asset = self.gym.create_sphere(self.sim, 1 * np.tan(np.radians( 35/2)) , asset_options)

        # self.gym.create_actor(env, sphere_asset, gymapi.Transform(p=gymapi.Vec3(0, 0., 1)), "sphere", 0, 0)

    def setup_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim,
                                             gymapi.CameraProperties())
        assert self.viewer != None

        cam_pos = gymapi.Vec3(0.8, 0.2, 0.9)
        cam_target = gymapi.Vec3(0, 0, 0.2)
        self.gym.viewer_camera_look_at(self.viewer, self.env, cam_pos,
                                       cam_target)

    def setup_cameras(self, env):

        self.camera_handles = []
        for fov in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = float(fov)
            camera_props.width = 400
            camera_props.height = 400

            camera_handle = self.gym.create_camera_sensor(env, camera_props)

            fudge_fov = 1 * fov
            distance = np.sqrt(2) / 2 / np.tan(np.radians(fudge_fov / 2))

            self.gym.set_camera_location(camera_handle, env,
                                         gymapi.Vec3(distance, 0, 1),
                                         gymapi.Vec3(0, 0, 1))

            self.camera_handles.append(camera_handle)

    def save_images(self, folder):
        self.gym.render_all_camera_sensors(self.sim)

        path = Path(folder)

        if path.exists():
            print(path, "already exists!")
            if input("Clear it before continuing? [y/N]:").lower() == "y":
                shutil.rmtree(path)

        path.mkdir()

        for i, camera_handle in enumerate(self.camera_handles):
            color_image = self.gym.get_camera_image(self.sim, self.env,
                                                    camera_handle,
                                                    gymapi.IMAGE_COLOR)
            color_image = color_image.reshape(400, 400, 4)

            Image.fromarray(color_image).save(path / f"{i}.png")

            transform = self.gym.get_camera_transform(self.sim, self.env,
                                                      camera_handle)

            # identity = np.array([gymapi.Vec3(1,0,0),
            #                          gymapi.Vec3(0,1,0),
            #                          gymapi.Vec3(0,0,1),
            #                          gymapi.Vec3(0,0,0),])[None,:]

            # print(type(identity))

            # output = transform.transform_points( identity )
            # matrix = mathutils.Matrix.LocRotScale(transform.p , mathutils.Quaternion(transform.q) , None)

            with open(path / f"{i}.txt", "w+") as f:
                # f.write( str(matrix) )
                # json.dump([ [v.x, v.y, v.z] for v in output ], f)

                data = [
                    transform.p.x,
                    transform.p.y,
                    transform.p.z,
                    transform.r.x,
                    transform.r.y,
                    transform.r.z,
                    transform.r.w,
                ]
                json.dump(data, f)

            # plt.imshow(color_image.reshape(400,400,4))
            # plt.show()

    def get_images(self):
        pass

    def get_object_pose(self):
        pass

    def step_gym(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.step_graphics(self.sim)

        if self.viewer != None:
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)


def main():
    tf = TestEnv()

    while not tf.gym.query_viewer_has_closed(tf.viewer):
        tf.step_gym()

    tf.save_images("/home/mikadam/Desktop/fov_test")


if __name__ == "__main__":
    main()
