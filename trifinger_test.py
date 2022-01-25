
import os
import json
import math
import numpy as np
# import mathutils
from PIL import Image
from pathlib import Path
import shutil

from isaacgym import gymapi, gymutil#, gymtorch

import matplotlib.pyplot as plt

# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

class TriFingerEnv:

    def __init__(self, viewer = True, robot=True):
        self.args = gymutil.parse_arguments( description="Trifinger test",)
        self.gym = gymapi.acquire_gym()
        self.robot = robot

        self.setup_sim()
        self.setup_envs()

        if viewer:
            self.setup_viewer()
        else:
            self.viewer = None

        self.gym.prepare_sim(self.sim)

    def setup_sim(self):
        #only tested with this one
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
        sim_params.physx.use_gpu = self.args.use_gpu

        # allows for non-convex objects but has other issues
        # self.args.physics_engine = gymapi.SIM_FLEX
        # sim_params.flex.solver_type = 5
        # sim_params.flex.num_outer_iterations = 4
        # sim_params.flex.num_inner_iterations = 20
        # sim_params.flex.relaxation = 0.8
        # sim_params.flex.warm_start = 0.5

        # sim_params.use_gpu_pipeline = True
        sim_params.use_gpu_pipeline = False
        self.sim = self.gym.create_sim(self.args.compute_device_id,
                                       self.args.graphics_device_id,
                                       self.args.physics_engine,
                                       sim_params)
        assert self.sim != None



        intensity = gymapi.Vec3( 0.3, 0.3, 0.3)
        ambient   = gymapi.Vec3( 0.5, 0.5, 0.5)

        self.gym.set_light_parameters(self.sim, 0, intensity, ambient, gymapi.Vec3( 0.5, 1,  1))
        self.gym.set_light_parameters(self.sim, 1, intensity, ambient, gymapi.Vec3( 1, 0,  1))
        self.gym.set_light_parameters(self.sim, 2, intensity, ambient, gymapi.Vec3( 0.5, -1,  1))
        self.gym.set_light_parameters(self.sim, 3, intensity, ambient, gymapi.Vec3( 0, 0,  1))

    def setup_envs(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        self.gym.add_ground(self.sim, plane_params)

        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        env = self.gym.create_env(self.sim, env_lower, env_upper, 0)

        self.env = env # used only when there is one env
        self.envs = [env]

        #TODO asset setup and adding to enviroment should be seperated
        if self.robot:
            self.setup_robot(env)
        self.setup_stage(env)
        self.setup_object(env)
        self.setup_cameras(env)


    def setup_stage(self, env):
        asset_dir = 'assets'

        stage_urdf_file = "trifinger/robot_properties_fingers/urdf/high_table_boundary.urdf" #this one is convex decomposed
        # stage_urdf_file = "trifinger/robot_properties_fingers/urdf/trifinger_stage.urdf"
        # stage_urdf_file = "trifinger/robot_properties_fingers/urdf/stage.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True
        asset_options.thickness = 0.001

        stage_asset = self.gym.load_asset(self.sim, asset_dir, stage_urdf_file, asset_options)
        self.gym.create_actor(env, stage_asset, gymapi.Transform(), "Stage", 0, 0)

    def setup_robot(self, env):
        asset_dir = 'assets'
        robot_urdf_file = "trifinger/robot_properties_fingers/urdf/pro/trifingerpro.urdf"
        # robot_urdf_file = "trifinger/robot_properties_fingers/urdf/trifinger_with_stage.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True
        asset_options.thickness = 0.001

        robot_asset = self.gym.load_asset(self.sim, asset_dir, robot_urdf_file, asset_options)

        trifinger_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        for p in trifinger_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.8
        self.gym.set_asset_rigid_shape_properties(robot_asset, trifinger_props)

        fingertips_frames = ["finger_tip_link_0", "finger_tip_link_120", "finger_tip_link_240"]
        self.fingertips_frames = {}

        for frame_name in fingertips_frames:
            frame_handle = self.gym.find_asset_rigid_body_index(robot_asset, frame_name)
            assert frame_handle != gymapi.INVALID_HANDLE
            self.fingertips_frames[frame_name] = frame_handle

        robot_dof_names = []
        for finger_pos in ['0', '120', '240']:
            robot_dof_names += [f'finger_base_to_upper_joint_{finger_pos}',
                                f'finger_upper_to_middle_joint_{finger_pos}',
                                f'finger_middle_to_lower_joint_{finger_pos}']

        self.dofs = {}
        for dof_name in robot_dof_names:
            dof_handle = self.gym.find_asset_dof_index(robot_asset, dof_name)
            assert dof_handle != gymapi.INVALID_HANDLE
            self.dofs[dof_name] = dof_handle

        max_torque_Nm = 0.36
        # maximum joint velocity (in rad/s) on each actuator
        max_velocity_radps = 10

        self.robot_actor = self.gym.create_actor(env, robot_asset, gymapi.Transform(), "Trifinger", 0, 0)

        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        for k, dof_index in enumerate(self.dofs.values()):
            # note: since safety checks are employed, the simulator PD controller is not
            #       used. Instead the torque is computed manually and applied, even if the
            #       command mode is 'position'.
            robot_dof_props['driveMode'][dof_index] = gymapi.DOF_MODE_EFFORT
            robot_dof_props['stiffness'][dof_index] = 0.0
            robot_dof_props['damping'][dof_index] = 0.0
            # set dof limits
            robot_dof_props['effort'][dof_index] = max_torque_Nm
            robot_dof_props['velocity'][dof_index] = max_velocity_radps
            # joint limits
            robot_dof_props['lower'][dof_index] = float(([-0.33, 0.0, -2.7] * 3)[k])
            robot_dof_props['upper'][dof_index] = float(([ 1.0,  1.57, 0.0] * 3)[k])
            #TODO make this read from strcuture

        self.gym.set_actor_dof_properties(env, self.robot_actor, robot_dof_props)


    def setup_object(self, env):
        asset_dir = 'assets'
        teady_bear_file = "objects/urdf/teady_bear.urdf"

        asset_options = gymapi.AssetOptions()
        # asset_options.thickness = 0.001

        asset_options.vhacd_enabled = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True

        asset_options.vhacd_params.mode = 0
        asset_options.vhacd_params.resolution = 300000
        asset_options.vhacd_params.max_convex_hulls = 10
        asset_options.vhacd_params.max_num_vertices_per_ch = 16

        sphere_asset = self.gym.create_sphere(self.sim, 0.1, asset_options)
        teady_bear_asset = self.gym.load_asset(self.sim, asset_dir, teady_bear_file, asset_options)

        # gym.create_actor(env, sphere_asset, gymapi.Transform(p=gymapi.Vec3(0., 0., 1.)), "sphere", 0, 0)
        self.gym.create_actor(env, teady_bear_asset, gymapi.Transform(p=gymapi.Vec3(0., 0., 0.1)), "teady bear", 0, 0)

    def setup_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        assert self.viewer != None

        cam_pos = gymapi.Vec3(0.8, 0.2, 0.9)
        cam_target = gymapi.Vec3(0, 0, 0.2)
        self.gym.viewer_camera_look_at(self.viewer, self.env, cam_pos, cam_target)

    def setup_cameras(self, env):
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = 35.0
        camera_props.width = 400
        camera_props.height = 400

        props = ['far_plane', 'height', 'horizontal_fov', 'near_plane', 'supersampling_horizontal', 'supersampling_vertical', 'use_collision_geometry', 'width']

        for prop in props:
            print(prop + ": " + str(getattr(camera_props,prop)))

        # generates cameara positions along rings around object
        heights   = [0.6, 0.3, 0.9, 1.]
        distances = [0.25, 0.4, 0.5, 0.1]
        counts    = [7,   13,   12,    1]
        target_z  = [0.0, 0.1,0.2, 0.1]

        # heights   = [0.3, 0.9, 1.]
        # distances = [0.4, 0.5, 0.1]
        # counts    = [6,   5,    1]
        # target_z  = [0.1,0.1, 0.1]

        camera_positions = []
        for h,d,c,z in zip(heights, distances, counts, target_z):
            for alpha in np.linspace(0, 2*np.pi, c, endpoint=False):
                camera_positions.append( ([d* np.sin(alpha), d*np.cos(alpha), h], z) )

        self.camera_handles = []
        for pos,z in camera_positions:
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            self.gym.set_camera_location(camera_handle, env, gymapi.Vec3(*pos), gymapi.Vec3(0,0,z))

            self.camera_handles.append(camera_handle)

    def save_images(self, folder):
        self.gym.render_all_camera_sensors(self.sim)

        path = Path(folder)

        if path.exists():
            print(path, "already exists!")
            if input("Clear it before continuing? [y/N]:").lower() == "y":
                shutil.rmtree(path)

        path.mkdir()

        for i,camera_handle in enumerate(self.camera_handles):
            color_image = self.gym.get_camera_image(self.sim, self.env, camera_handle, gymapi.IMAGE_COLOR)
            color_image = color_image.reshape(400,400,4)

            Image.fromarray(color_image).save(path / f"{i}.png")

            transform = self.gym.get_camera_transform(self.sim, self.env, camera_handle)

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

                data = [transform.p.x, transform.p.y, transform.p.z, transform.r.x, transform.r.y, transform.r.z, transform.r.w]
                json.dump(data, f)

            # plt.imshow(color_image.reshape(400,400,4))
            # plt.show()

    def get_images(self):
        pass


    def get_object_pose(self):
        pass

    def get_robot_state(self):
        dof_states = self.gym.get_actor_dof_states(self.env, self.robot_actor, gymapi.STATE_ALL)
        print(dof_states)

    def do_robot_action(self, action):
        applied_torque = np.array([ action * 0.3, 0.3 , -0.3,
                                    0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0], dtype=np.float32)

#         applied_torque = gymapi.Tensor([ 0.0, 0.9,-0.0,
#                                     0.0, 0.0, 0.3,
#                                     0.0, 0.0, 0.0])

        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(applied_torque))
        # self.gym.set_dof_actuation_force_tensor(self.sim, applied_torque)
        # self.gym.set_dof_actuation_force_tensor(self.sim, applied_torque)
        self.gym.apply_actor_dof_efforts(self.env, self.robot_actor, applied_torque)

    def step_gym(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.step_graphics(self.sim)

        if self.viewer != None:
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)


def get_nerf_training():
    tf = TriFingerEnv(robot = False)

    # for _ in range(500):
    while not tf.gym.query_viewer_has_closed(tf.viewer):
        tf.step_gym()

    # tf.save_images("/home/mikadam/Desktop/test")

def run_robot_control():
    tf = TriFingerEnv()

    direction = 1
    count = 0
    while not tf.gym.query_viewer_has_closed(tf.viewer):
        count += 1
        if count == 100:
            print("flip")
            direction = - direction
            count = 0

        # prototype of inerface
        tf.get_robot_state()
        tf.do_robot_action(direction)

        tf.step_gym()

    print("closed!")


if __name__ == "__main__":
    get_nerf_training()
    # run_robot_control()




