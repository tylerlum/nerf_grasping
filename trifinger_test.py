


import math
import numpy as np
from isaacgym import gymapi, gymutil


args = gymutil.parse_arguments( description="Trifinger test",)

gym = gymapi.acquire_gym()


#only tested with this one
assert args.physics_engine == gymapi.SIM_PHYSX

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

# args.physics_engine = gymapi.SIM_FLEX
# sim_params.flex.solver_type = 5
# sim_params.flex.num_outer_iterations = 4
# sim_params.flex.num_inner_iterations = 20
# sim_params.flex.relaxation = 0.8
# sim_params.flex.warm_start = 0.5

sim_params.use_gpu_pipeline = True

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
gym.add_ground(sim, plane_params)


spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, env_lower, env_upper, 0)

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.use_mesh_materials = True
asset_options.thickness = 0.001


asset_dir = 'assets'
robot_urdf_file = "trifinger/robot_properties_fingers/urdf/pro/trifingerpro.urdf"
# robot_urdf_file = "trifinger/robot_properties_fingers/urdf/trifinger_with_stage.urdf"

teady_bear_file = "objects/urdf/teady_bear.urdf"


stage_urdf_file = "trifinger/robot_properties_fingers/urdf/high_table_boundary.urdf"
# stage_urdf_file = "trifinger/robot_properties_fingers/urdf/trifinger_stage.urdf"
# stage_urdf_file = "trifinger/robot_properties_fingers/urdf/stage.urdf"


# asset = gym.load_asset(sim, '../isaacgym/assets', "urdf/cartpole.urdf", asset_options)
stage_asset = gym.load_asset(sim, asset_dir, stage_urdf_file, asset_options)

robot_asset = gym.load_asset(sim, asset_dir, robot_urdf_file, asset_options)
# robot_asset = gym.load_asset(sim, asset_dir, stage_urdf_file, asset_options)


gym.create_actor(env, stage_asset, gymapi.Transform(p=gymapi.Vec3(0., 0., 0.0)), "Stage", 0, 0)
gym.create_actor(env, robot_asset, gymapi.Transform(p=gymapi.Vec3(0., 0., 0.0)), "Trifinger", 0, 0)


asset_options = gymapi.AssetOptions()
# asset_options.thickness = 0.001

asset_options.vhacd_enabled = True
# asset_options.thickness = 0.001

asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
asset_options.override_inertia = True
asset_options.override_com = True

asset_options.vhacd_params.mode = 0
asset_options.vhacd_params.resolution = 300000
# asset_options.vhacd_params.max_convex_hulls = 10
asset_options.vhacd_params.max_num_vertices_per_ch = 16

sphere_asset = gym.create_sphere(sim, 0.1, asset_options)

# gym.create_actor(env, sphere_asset, gymapi.Transform(p=gymapi.Vec3(0., 0., 1.)), "test asset", 0, 0)


teady_bear_asset = gym.load_asset(sim, asset_dir, teady_bear_file, asset_options)
gym.create_actor(env, teady_bear_asset, gymapi.Transform(p=gymapi.Vec3(0., 0., 0.1)), "teady asset", 0, 0)


camera_props = gymapi.CameraProperties()
camera_props.width = 128
camera_props.height = 128
camera_handle = gym.create_camera_sensor(env, camera_props)

gym.set_camera_location(camera_handle, env, gymapi.Vec3(1,1,1.5), gymapi.Vec3(0,0,0))
# color_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

cam_pos = gymapi.Vec3(0.5, 0.5, 0.7)
# cam_pos = gymapi.Vec3(1, 1, 1.5)
cam_target = gymapi.Vec3(0, 0, 0.2)
gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)


gym.prepare_sim(sim)
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.sync_frame_time(sim)








