
import os
import json
import math
from turtle import color
import numpy as np
# import mathutils
from PIL import Image
from pathlib import Path
import shutil

from isaacgym import gymapi, gymutil, gymtorch

import matplotlib.pyplot as plt
import cvxpy as cp

import torch

# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

# grasp poitns
# tensor([[ 0.0128, -0.0739, -0.0815],
#     [-0.0380, -0.0607,  0.0463],
#     [ 0.0466, -0.0137, -0.0653]], requires_grad=True)


# def get_fixed_camera_transfrom(camera)

def skew_matrix(vectors):
    skew = np.zeros( vectors.shape[:-1] + (3,3) )

    skew[..., 0, 1] = -vectors[..., 2]
    skew[..., 1, 2] = -vectors[..., 0]
    skew[..., 2, 0] = -vectors[..., 1]
    skew[..., 1, 0] =  vectors[..., 2]
    skew[..., 2, 1] =  vectors[..., 0]
    skew[..., 0, 2] =  vectors[..., 1]
    
    return skew

def example_rotation_transform(normals):
    #hopefully no one will try grabing directly under or above
    global_z_axis = np.array([0,0,1])

    #  n,3, 1      3, 3                       n, 3, 1
    local_x = skew_matrix(global_z_axis) @ normals[..., None]

    #  n,3,1         n,3,3              n,3,1
    local_y = skew_matrix(normals) @ local_x

    rotations = np.stack([ local_x, local_y, normals[..., None] ], axis=-1)[...,0,:]
    return rotations

def calculate_grip_forces(positions, normals, target_force):
    """ positions are relative to object CG if we want unbalanced torques"""
    mu = 0.9

    torch_input = type(positions) == torch.Tensor
    if torch_input:
        assert type(normals) == torch.Tensor, "numpy vs torch needs to be consistant"
        assert type(target_force) == torch.Tensor, "numpy vs torch needs to be consistant"
        positions = positions.numpy()
        normals = normals.numpy()
        target_force = target_force.numpy()

    n, _ = positions.shape
    assert normals.shape == (n, 3)
    assert target_force.shape == (3,)

    F = cp.Variable( (n, 3) )
    constraints = []

    normals = normals /  np.linalg.norm( normals, axis=-1, keepdims=True)

    total_force = np.zeros((3))
    total_torque = np.zeros((3))

    Q = []

    for pos,norm,f in zip(positions, normals, F):
        q = example_rotation_transform(norm)
        Q.append(q)

        total_force += q @ f
        total_torque += skew_matrix(pos) @ q @ f

    constraints.append( total_force == target_force )
    constraints.append( total_torque == 0 )

    friction_cone = cp.norm(F[:,:2], axis=1) <= mu * F[:,2]
    constraints.append( friction_cone )

    force_magnitudes = cp.norm(F, axis=1)
    # friction_magnitudes = cp.norm(F[:,2], axis=1)
    prob = cp.Problem(cp.Minimize(cp.max(force_magnitudes)), constraints)
    prob.solve()

    global_forces = np.zeros_like(F.value)
    for i in range(n):
        global_forces[i, :] = Q[i] @ F.value[i,:]

    if torch_input:
        global_forces = torch.Tensor(global_forces)

    return global_forces


class Robot:

    dof_min = None
    dof_max = None
    dof_default = None

    def __init__(self):
        pass

    def control(self):
        pass

    def position_control(self):
        pass

    def vel_control_force_limit(self):
        pass

    def object_control(self):
        pass

class TriFingerEnv:

    def __init__(self, viewer = True, robot=True, obj = True):
        self.args = gymutil.parse_arguments( description="Trifinger test",)
        self.gym = gymapi.acquire_gym()

        self.setup_sim()
        self.setup_envs(robot = robot,  obj = obj)

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
        # sim_params.physx.use_gpu = True

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



        # intensity = 0.01 # for nerf generation
        # ambient = 0.21 / intensity
        intensity = 0.5
        ambient = 0.10 / intensity
        intensity = gymapi.Vec3( intensity, intensity, intensity)
        ambient   = gymapi.Vec3( ambient, ambient, ambient)

        self.gym.set_light_parameters(self.sim, 0, intensity, ambient, gymapi.Vec3( 0.5, 1,  1))
        self.gym.set_light_parameters(self.sim, 1, intensity, ambient, gymapi.Vec3( 1, 0,  1))
        self.gym.set_light_parameters(self.sim, 2, intensity, ambient, gymapi.Vec3( 0.5, -1,  1))
        self.gym.set_light_parameters(self.sim, 3, intensity, ambient, gymapi.Vec3( 0, 0,  1))

    def setup_envs(self, robot, obj):
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
        if robot:
            self.setup_robot(env)
        self.setup_stage(env)
        if obj:
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
        self.gym.create_actor(env, stage_asset, gymapi.Transform(), "Stage", 0, 0, segmentationId=1)

    def setup_robot(self, env):
        asset_dir = 'assets'
        robot_urdf_file = "trifinger/robot_properties_fingers/urdf/pro/trifingerpro.urdf"
        # robot_urdf_file = "trifinger/robot_properties_fingers/urdf/trifinger_with_stage.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True
        asset_options.disable_gravity = True # to make things easier - will eventually compensate ourselves

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

        self.dofs = {} #TODO fix asset vs actor index differnce
        for dof_name in robot_dof_names:
            dof_handle = self.gym.find_asset_dof_index(robot_asset, dof_name)
            assert dof_handle != gymapi.INVALID_HANDLE
            self.dofs[dof_name] = dof_handle

        max_torque_Nm = 0.36
        # maximum joint velocity (in rad/s) on each actuator
        max_velocity_radps = 10

        self.robot_actor = self.gym.create_actor(env, robot_asset, gymapi.Transform(), "Trifinger", 0, 0, segmentationId=5)


        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)

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

            #defaults
            dof_states[dof_index, 0] = float(([-0.8, 1.2, -2.7] * 3)[k])
            dof_states[dof_index, 1] = float(([0.0, 0.0, 0.0] * 3)[k])

        self.gym.set_actor_dof_properties(env, self.robot_actor, robot_dof_props)

        print("setting dof state")
        self.gym.set_dof_state_tensor(self.sim, _dof_states)


    def setup_object(self, env):
        asset_dir = 'assets'
        teady_bear_file = "objects/urdf/teady_bear.urdf"

        asset_options = gymapi.AssetOptions()
        # asset_options.thickness = 0.001

        asset_options.vhacd_enabled = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.density = 100

        asset_options.vhacd_params.mode = 0
        asset_options.vhacd_params.resolution = 300000
        asset_options.vhacd_params.max_convex_hulls = 10
        asset_options.vhacd_params.max_num_vertices_per_ch = 16

        # teady_bear_asset = self.gym.load_asset(self.sim, asset_dir, teady_bear_file, asset_options)
        # self.teady = self.gym.create_actor(env, teady_bear_asset, gymapi.Transform(p=gymapi.Vec3(0., 0., 0.1)), "teady bear", 0, 0, segmentationId=2)

        box_asset     = self.gym.create_box(self.sim, 0.1, 0.1, 0.1, asset_options)
        rs_props = self.gym.get_asset_rigid_shape_properties(box_asset)
        for p in rs_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.8
        self.gym.set_asset_rigid_shape_properties(box_asset, rs_props)

        self.teady = self.gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(0., 0., 0.101)), "teady bear", 0, 0, segmentationId=2)
        self.gym.set_rigid_body_color(self.env, self.teady, 0 , gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.2, 0.3))

        # sphere_asset     = self.gym.create_sphere(self.sim, 0.1, asset_options)
        # self.teady = self.gym.create_actor(env, sphere_asset, gymapi.Transform(p=gymapi.Vec3(0., 0., 0.105)), "teady bear", 0, 0, segmentationId=2)
        # self.gym.set_rigid_body_color(self.env, self.teady, 0 , gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.2, 0.3))

    def setup_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        assert self.viewer != None

        cam_pos = gymapi.Vec3(0.8, 0.2, 0.7)
        cam_target = gymapi.Vec3(0, 0, 0.2)
        self.gym.viewer_camera_look_at(self.viewer, self.env, cam_pos, cam_target)

    def setup_cameras(self, env):
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = 35.0
        camera_props.width = 400
        camera_props.height = 400
        
        # generates cameara positions along rings around object
        heights   = [0.6, 0.3, 0.9, 1.]
        distances = [0.25, 0.4, 0.5, 0.1]
        counts    = [7,   13,   12,    1]
        target_z  = [0.0, 0.1,0.2, 0.1]

        camera_positions = []
        for h,d,c,z in zip(heights, distances, counts, target_z):
            for alpha in np.linspace(0, 2*np.pi, c, endpoint=False):
                camera_positions.append( ([d* np.sin(alpha), d*np.cos(alpha), h], z) )

        self.camera_handles = []
        for pos,z in camera_positions:
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            self.gym.set_camera_location(camera_handle, env, gymapi.Vec3(*pos), gymapi.Vec3(0,0,z))

            self.camera_handles.append(camera_handle)

    def save_images(self, folder, overwrite = False):
        self.gym.render_all_camera_sensors(self.sim)

        path = Path(folder)

        if path.exists():
            print(path, "already exists!")
            if overwrite:
                shutil.rmtree(path)
            elif input("Clear it before continuing? [y/N]:").lower() == "y":
                shutil.rmtree(path)

        path.mkdir()

        for i,camera_handle in enumerate(self.camera_handles):
            color_image = self.gym.get_camera_image(self.sim, self.env, camera_handle, gymapi.IMAGE_COLOR)
            color_image = color_image.reshape(400,400,-1)
            Image.fromarray(color_image).save(path / f"col_{i}.png")


            color_image = self.gym.get_camera_image(self.sim, self.env, camera_handle, gymapi.IMAGE_SEGMENTATION)
            color_image = color_image.reshape(400,400) * 30
            Image.fromarray(color_image, "I").convert("L").save(path / f"seg_{i}.png")

            color_image = self.gym.get_camera_image(self.sim, self.env, camera_handle, gymapi.IMAGE_DEPTH)
            color_image = -color_image.reshape(400,400) # distance in units I think
            color_image = (np.clip(color_image, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(color_image).convert("L").save(path / f"dep_{i}.png")

            transform = self.gym.get_camera_transform(self.sim, self.env, camera_handle)

            # identity = np.array([gymapi.Vec3(1,0,0),
            #                          gymapi.Vec3(0,1,0),
            #                          gymapi.Vec3(0,0,1),
            #                          gymapi.Vec3(0,0,0),])[None,:]

            # print(type(identity))

            # output = transform.transform_points( identity )
            # matrix = mathutils.Matrix.LocRotScale(transform.p , mathutils.Quaternion(transform.q) , None)

            with open(path / f"quat_{i}.txt", "w+") as f:
                # f.write( str(matrix) )
                # json.dump([ [v.x, v.y, v.z] for v in output ], f)

                data = [transform.p.x, transform.p.y, transform.p.z, transform.r.x, transform.r.y, transform.r.z, transform.r.w]
                json.dump(data, f)

            # plt.imshow(color_image.reshape(400,400,4))
            # plt.show()

    def get_images(self):
        pass


    def get_object_pose(self):
        transform = self.gym.get_rigid_transform(self.env, self.teady)
        data = [transform.p.x, transform.p.y, transform.p.z, transform.r.x, transform.r.y, transform.r.z, transform.r.w]
        print(data)

    def robot_control(self, count):
        _mass_matrix = self.gym.acquire_mass_matrix_tensor(self.sim, "Trifinger")
        mass_matrix = gymtorch.wrap_tensor(_mass_matrix)

        # for fixed base
        # jacobian[env_index, link_index - 1, :, dof_index]
        _jac = self.gym.acquire_jacobian_tensor(self.sim, "Trifinger")
        jacobian = gymtorch.wrap_tensor(_jac)

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)

        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        applied_torque = torch.zeros((9))
        # finger_pos = 0

        # teady points
        # grasp_points = np.array( [[ 0.035, 0.058, 0.101,],
        #                           [0.0, -0.048, 0.083,],
        #                           [-0.039, 0.058, 0.101,]], dtype=np.float32)

        # box points
        grasp_points = torch.Tensor( [[ 0.0,  0.05, 0.05,],
                                  [ 0.03,-0.05, 0.05,],
                                  [-0.03,-0.05, 0.05,]])

        graps_normals = torch.Tensor( [[ 0.0,-1.0, 0.0,],
                                  [ 0.0, 1.0, 0.0,],
                                  [ 0.0, 1.0, 0.0,]])

        # print(grasp_points.dtype)

        for finger_index, finger_pos in enumerate([0, 120, 240]):
            robot_dof_names = [f'finger_base_to_upper_joint_{finger_pos}',
                                f'finger_upper_to_middle_joint_{finger_pos}',
                                f'finger_middle_to_lower_joint_{finger_pos}']

            dof_idx = [self.gym.find_actor_dof_index(self.env, self.robot_actor, dof_name, gymapi.DOMAIN_SIM) for dof_name in robot_dof_names]
            tip_index =  self.gym.find_actor_rigid_body_index(self.env, self.robot_actor, f"finger_tip_link_{finger_pos}", gymapi.DOMAIN_SIM)

            # only care about tip position
            local_jacobian = jacobian[0, tip_index - 1, :3, dof_idx]
            
            # print("dof_states", dof_states[dof_idx, :])

            tip_state = rb_states[tip_index, :]
            # print("rb_states", tip_state[:3], tip_state[6:9]) #position, orientation, linear velocity, and angular velocity

            # pos_target = torch.Tensor([0.1*np.cos(2*np.pi*count/200) , 0.05, 0.1])

            tip_pos = tip_state[:3]
            tip_vel = tip_state[7:10]

            c = np.cos(2*np.pi * finger_pos / 360)
            s = np.sin(2*np.pi * finger_pos / 360)
            rot_matrix_finger = torch.Tensor([[ c, s, 0],
                                              [-s, c, 0],
                                              [ 0, 0, 1]])

            print("dof state", dof_states[:, 0] )
            print("finger pos", tip_pos)

            count = count % 1000

            mode = "pos"
            if   count < 30:  mode = "off"
            elif count < 120: mode = "pos"
            elif count < 350: mode = "vel"
            else:             mode = "up"

            print(count, mode)

            # pos_target = rot_matrix_finger @ torch.Tensor([0.0 , 0.15, 0.09])
            pos_target = grasp_points[ finger_index, :]
            if mode == "off":
                #go far to not intersect

                pos_target = torch.Tensor( [[ 0.0,  0.10, 0.05,],
                                            [ 0.05,-0.10, 0.05,],
                                            [-0.05,-0.10, 0.05,]]) [ finger_index, :]

                # xyz_force = torch.zeros((3))
                pos_error = tip_pos - pos_target

                print("pos_target", pos_target)
                print("pos erro", pos_error)

                xyz_force = - 5.0 * pos_error - 1.0 * tip_vel

            elif mode == "pos":
                # PD controller in xyz space
                pos_error = tip_pos - pos_target

                print("pos_target", pos_target)
                print("pos erro", pos_error)

                xyz_force = - 5.0 * pos_error - 1.0 * tip_vel

            elif mode == "vel":
                # define vector along which endeffector should close
                # will be position controlled perpendicular to motion
                # and velocity controlled along. Force along with clamped to prevent crushing
                start_point = pos_target
                # normal      = rot_matrix_finger @ torch.Tensor([0.0 , -0.05, 0])

                normal = graps_normals[finger_index,:]
                target_vel  = 0.05
                max_force   = 0.5

                normal /= normal.norm()

                pos_relative = tip_pos - start_point

                perp_xyz_force = - 5.0 * pos_relative - 1.0 * tip_vel
                perp_xyz_force = perp_xyz_force - normal * normal.dot(perp_xyz_force)

                vel_error = normal.dot(tip_vel) - target_vel

                parallel_xyz_force_mag = -1.0 * vel_error
                parallel_xyz_force = torch.clamp(parallel_xyz_force_mag, -max_force, max_force)

                xyz_force = parallel_xyz_force * normal + perp_xyz_force

            elif mode == "up":
                # ugly runs the calculation 3 times unnecesarily
                target_force = 1.1 * torch.Tensor([0,0,1])

                global_forces = calculate_grip_forces(grasp_points, graps_normals, target_force)
                xyz_force = global_forces[finger_index, :]

            else:
                assert False, "Unknown mode"

            joint_torques = torch.t( local_jacobian ) @ xyz_force
            applied_torque[dof_idx] = joint_torques

            print(finger_index, "xyz_force", xyz_force)
            # print("jacobian", local_jacobian)
            # print("joint_torques", joint_torques)
            # print()


        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(applied_torque))

    def do_robot_action(self, action):
        applied_torque = np.array([ action * 0.3, 0.3 , -0.3,
                                    0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0], dtype=np.float32)

#         applied_torque = gymapi.Tensor([ 0.0, 0.9,-0.0,
#                                     0.0, 0.0, 0.3,
#                                     0.0, 0.0, 0.0])

        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(applied_torque))
        self.gym.apply_actor_dof_efforts(self.env, self.robot_actor, applied_torque)

    def step_gym(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.step_graphics(self.sim)

        if self.viewer != None:
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)


def get_nerf_training(viewer):
    # tf = TriFingerEnv(viewer=viewer, robot = False, obj=True)

    # for _ in range(500):
    # # while not tf.gym.query_viewer_has_closed(tf.viewer):
    #     tf.step_gym()
    #     tf.get_object_pose()

    # tf.save_images("/media/data/mikadam/outputs/test", overwrite=True)

    blank = TriFingerEnv(viewer=viewer, robot = False, obj=False)
    blank.save_images("/media/data/mikadam/outputs/blank", overwrite=True)

def run_robot_control(viewer):
    tf = TriFingerEnv(viewer= viewer, robot = True, obj= True)

    count = 0
    # for _ in range(500):
    while not tf.gym.query_viewer_has_closed(tf.viewer):
        count += 1

        # prototype of inerface
        tf.robot_control(count)
        # tf.do_robot_action(direction)

        tf.step_gym()

    print("closed!")


if __name__ == "__main__":
    # get_nerf_training(viewer = False)
    run_robot_control(viewer = True)




