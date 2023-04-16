from isaacgym import gymapi, gymtorch
from pathlib import Path
import os
import numpy as np
import torch

import trimesh
import scipy

from nerf_grasping.quaternions import Quaternion
from nerf_grasping import config, grasp_utils, nerf_utils

from nerf import utils

root_dir = Path(os.path.abspath(__file__)).parents[2]
asset_dir = f"{root_dir}/assets"
gd_mesh_dir = f"{root_dir}/grasp_data/meshes"
OBJ_SEGMENTATION_ID = 2
print("root_dir", root_dir)


def load_nerf(workspace, bound, scale, obj_translation):
    parser = utils.get_config_parser()
    args = parser.parse_args(
        [
            "--workspace",
            # f"{root_dir}/nerf_checkpoints/{workspace}",
            f"{root_dir}/torch-ngp/isaac_banana_nerf",
            "--fp16",
            "--test",
            "--bound",
            f"{bound}",
            "--scale",
            f"{scale}",
            "--mode",
            "blender",
            f"{root_dir}/torch-ngp",
        ]
    )
    model = nerf_utils.load_nerf(args, obj_translation)
    return model


def load_object(exp_config: config.Experiment):
    if exp_config.object == config.ObjectType.BANANA:
        obj = Banana()
    if exp_config.object == config.ObjectType.BIG_BANANA:
        obj = BigBanana()
    elif exp_config.object == config.ObjectType.BOX:
        obj = Box()
    elif exp_config.object == config.ObjectType.TEDDY_BEAR:
        obj = TeddyBear()
        obj.use_centroid = True
    elif exp_config.object == config.ObjectType.POWER_DRILL:
        obj = PowerDrill()
    elif exp_config.object == config.ObjectType.BLEACH_CLEANSER:
        obj = BleachCleanser()
    return obj


class RigidObject:
    scale = 1.0
    bound = 2.0
    centroid = np.zeros((3, 1))
    use_centroid = False
    mu = 1.0
    translation = np.zeros(3)

    def __init__(self, gym=None, sim=None, env=None):
        self.setup_gym(gym, sim, env)
        self.nerf_loaded = False
        self.model = None
        self.gt_mesh = self.load_trimesh()

    def setup_gym(self, gym, sim, env):
        self.gym = gym
        self.sim = sim
        self.env = env
        if self.sim is not None:
            self.asset = self.create_asset()
            self.actor = self.configure_actor(self.gym, self.env)

    def setup_tensors(self):
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # (num_rigid_bodies, 13)
        rb_count = self.gym.get_actor_rigid_body_count(self.env, self.actor)
        rb_start_index = self.gym.get_actor_rigid_body_index(
            self.env, self.actor, 0, gymapi.DOMAIN_SIM
        )

        # TODO TEMP
        self.index = rb_start_index

        # NOTE: simple indexing will return a view of the data but advanced indexing will return a copy breaking the updateing
        self.rb_states = gymtorch.wrap_tensor(_rb_states)[
            rb_start_index : rb_start_index + rb_count, :
        ]

    def load_nerf_model(self):
        if self.nerf_loaded:
            return
        # TODO: whether to use new_translation or translation?
        self.model = load_nerf(self.workspace, self.bound, self.scale, self.translation)
        self.nerf_loaded = True

    def load_trimesh(self, mesh_path=None):
        if mesh_path is None or not os.path.exists(mesh_path):
            asset_path = os.path.join(asset_dir, self.asset_file)
            mesh_path = os.path.join(
                asset_dir, "objects", self._get_mesh_path_from_urdf(asset_path)
            )

        print("mesh path: ", mesh_path)
        # Mesh loaded in Z-up, centered at object centroid.
        mesh = trimesh.load(mesh_path, force="mesh")

        print("mesh extents: ", mesh.extents)

        # IG centroid (when object is loaded into sim) in Nerf frame
        mesh.nerf_centroid = (
            grasp_utils.ig_to_nerf(self.translation.reshape(1, 3))
            .reshape(-1)
            .cpu()
            .numpy()
        )
        return mesh

    def _get_mesh_path_from_urdf(self, urdf_path):
        import xml.etree.ElementTree as ET

        # Load the URDF file
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Find the mesh filename inside the URDF file
        mesh_path = root.find(".//geometry/mesh").get("filename")
        return mesh_path

    def create_asset(self):
        asset_options = gymapi.AssetOptions()

        asset_options.vhacd_enabled = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.density = 100
        asset_options.override_inertia = False
        asset_options.override_com = False

        asset_options.vhacd_params.mode = (
            0  # 0 = tetrahedron, 1 = voxel, was 1, but 0 fixed issue with xbox360
        )
        asset_options.vhacd_params.resolution = 600000
        asset_options.vhacd_params.max_convex_hulls = 16
        asset_options.vhacd_params.max_num_vertices_per_ch = 128

        asset = self.gym.load_asset(self.sim, asset_dir, self.asset_file, asset_options)

        rs_props = self.gym.get_asset_rigid_shape_properties(asset)
        for p in rs_props:
            p.friction = self.mu
            p.torsion_friction = self.mu
            p.restitution = 0.0
        self.gym.set_asset_rigid_shape_properties(asset, rs_props)

        self.force_sensor_idx = self.gym.create_asset_force_sensor(
            asset, 0, gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
        )

        return asset

    def configure_actor(self, gym, env):
        actor = self.gym.create_actor(
            env,
            self.asset,
            gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.1)),
            self.name,
            1,
            0,
            segmentationId=OBJ_SEGMENTATION_ID,
        )

        self.force_sensor = self.gym.get_actor_force_sensor(
            env, actor, self.force_sensor_idx
        )

        rigid_body_props = self.gym.get_actor_rigid_body_properties(self.env, actor)
        self.mass = sum(x.mass for x in rigid_body_props)
        com = rigid_body_props[0].com
        self.CG = torch.tensor([com.x, com.y, com.z])
        return actor

    def reset_actor(self):
        self.rb_states[0] *= 0.0
        self.rb_states[0][6] = 1.0

        (rb_states,) = self.gym.get_actor_rigid_body_states(
            self.env, self.actor, gymapi.STATE_ALL
        )
        rb_states["pose"]["p"].fill((0, 0, 0))
        rb_states["pose"]["r"].fill((0, 0, 0, 1))
        rb_states["vel"]["linear"].fill((0, 0, 0))
        rb_states["vel"]["angular"].fill((0, 0, 0))
        self.gym.set_actor_rigid_body_states(
            self.env,
            self.actor,
            rb_states,
            gymapi.STATE_ALL,
        )
        return rb_states

    @property
    def position(self):
        """Returns object position"""
        return self.rb_states[0, :3]

    @property
    def orientation(self):
        """Returns object orientation"""
        return self.rb_states[0, 3:7]

    @property
    def velocity(self):
        """Returns linear velocity"""
        return self.rb_states[0, 7:10]

    @property
    def angular_velocity(self):
        return self.rb_states[0, 10:13]


class TeddyBear(RigidObject):
    asset_file = "objects/urdf/teddy_bear.urdf"
    name = "teddy_bear"
    workspace = "teddy_bear"

    data_dir = f"{root_dir}/nerf_shared/data/isaac_teddy"
    config_path = f"{root_dir}/nerf_shared/configs/isaac_teddy.txt"
    centroid = np.array([-0.0001444, 0.00412231, 0.08663063])
    use_centroid = False
    translation = np.array([-1.2824e-05, 6.9302e-06, 2.2592e-03])
    new_translation = np.array([1.7978e-08, -6.0033e-08, 2.7280e-03])
    orientation = np.array([3.4469e-06, 4.8506e-06, 6.9347e-07, 1.0000e00])

    # Grasp points are in XYZ (i.e. Z-up) frame
    grasp_points = torch.tensor(
        [[0.0350, 0.0580, 0.1010], [0.0000, -0.0480, 0.0830], [-0.0390, 0.0580, 0.1010]]
    )

    grasp_normals = torch.tensor(
        [
            [-0.0350, -0.0580, 0.0000],
            [0.0000, 1.0000, 0.0000],
            [0.0390, -0.0580, 0.0000],
        ]
    )
    mu = 1.0


class Box(RigidObject):
    name = "box"
    workspace = "box"
    grasp_points = torch.tensor(
        [[0.0, 0.05, 0.05], [0.03, -0.05, 0.05], [-0.03, -0.05, 0.05]]
    )
    grasp_normals = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    translation = np.array([-7.7818e-07, 1.5440e-06, 3.9500e-02])
    new_translation = np.array([7.8142e-07, -1.5576e-06, 3.9500e-02])
    asset_file = "objects/urdf/box.urdf"


#     def create_asset(self):
#         asset_options = gymapi.AssetOptions()

#         asset_options.vhacd_enabled = True
#         asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
#         asset_options.override_inertia = False
#         asset_options.override_com = False
#         asset_options.density = 100

#         asset_options.vhacd_params.mode = 0
#         asset_options.vhacd_params.resolution = 300000
#         asset_options.vhacd_params.max_convex_hulls = 10
#         asset_options.vhacd_params.max_num_vertices_per_ch = 16

#         asset = self.gym.create_box(self.sim, 0.1, 0.1, 0.1, asset_options)

#         rs_props = self.gym.get_asset_rigid_shape_properties(asset)
#         for p in rs_props:
#             p.friction = 1.0
#             p.torsion_friction = 1.0
#             p.restitution = 0.1
#         self.gym.set_asset_rigid_shape_properties(asset, rs_props)

#         return asset

#     def configure_actor(self, gym, env):
#         actor = self.gym.create_actor(
#             env,
#             self.asset,
#             gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.101)),
#             "box",
#             0,
#             0,
#             segmentationId=OBJ_SEGMENTATION_ID,
#         )
#         self.gym.set_rigid_body_color(
#             self.env, actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.2, 0.3)
#         )

#         rigid_body_props = self.gym.get_actor_rigid_body_properties(self.env, actor)
#         self.mass = sum(x.mass for x in rigid_body_props)
#         com = rigid_body_props[0].com
#         self.CG = torch.tensor([com.x, com.y, com.z])

#         return actor


class PowerDrill(RigidObject):
    workspace = "power_drill"
    centroid = np.zeros(3)
    grasp_points = torch.tensor(
        [
            [-0.038539, 0.115021, 0.023878],
            [0.030017, -0.002467, 0.027816],
            [-0.029284, -0.099212, 0.027294],
        ]
    )

    grasp_normals = torch.tensor([[0, -1.0, 0.0], [-1, 0.0, 0.0], [1.0, 1.0, 0.0]])
    bound = 3.0

    asset_file = "objects/urdf/power_drill.urdf"
    name = "power_drill"
    translation = np.array([-1.9098e-06, 1.6409e-06, 2.9233e-02])


class Banana(RigidObject):
    workspace = "banana"
    grasp_points = torch.tensor(
        [
            [-0.00693, 0.085422, 0.013867],
            [0.018317, -0.001611, 0.013867],
            [-0.058538, -0.051027, 0.013867],
        ]
    )
    grasp_normals = torch.tensor([[1, -1.5, 0.0], [-2, 1.0, 0.0], [1, 0.0, 0.0]])
    asset_file = "objects/urdf/banana.urdf"
    name = "banana"
    mu = 1.0
    bound = 2
    translation = np.array([1.7978e-08, -6.0033e-08, 2.7280e-03])


class BigBanana(RigidObject):
    workspace = "big_banana"
    grasp_points = torch.tensor(
        [
            [-0.00693, 0.085422, 0.013867],
            [0.018317, -0.001611, 0.013867],
            [-0.058538, -0.051027, 0.013867],
        ]
    )
    grasp_normals = torch.tensor([[1, -1.5, 0.0], [-2, 1.0, 0.0], [1, 0.0, 0.0]])
    asset_file = "objects/urdf/banana.urdf"
    name = "banana"
    bound = 1.5
    mu = 1.0
    translation = np.array([-1.4408e-05, 3.8640e-06, 2.7102e-03])
    new_translation = np.array([2.3227e-05, -1.6251e-05, 3.5882e-03])
    new_orientation = np.array([6.6045e-03, 1.0165e-02, 1.1789e-05, 9.9993e-01])

    def configure_actor(self, gym, env):
        actor = super().configure_actor(gym, env)
        self.mass *= 1.5
        return actor

    def setup_gym(self, gym, sim, env):
        super().setup_gym(gym, sim, env)


class Spatula(RigidObject):
    asset_file = "objects/urdf/spatula.urdf"
    name = "spatula"


class Mug(RigidObject):
    asset_file = "objects/urdf/mug.urdf"
    data_dir = "./nerf_shared/data/mug"
    config_path = "./nerf_shared/configs/mug.txt"
    name = "mug"


class BleachCleanser(RigidObject):
    asset_file = "objects/urdf/bleach_cleanser.urdf"
    data_dir = "./nerf_shared/data/bleach_cleanser"
    config_path = "./nerf_shared/configs/bleach_cleanser.txt"
    name = "bleach_cleanser"
    workspace = "bleach_cleanser"
    grasp_points = torch.tensor(
        [
            [-0.038539, 0.115021, 0.023878],
            [0.030017, -0.002467, 0.027816],
            [-0.029284, -0.099212, 0.027294],
        ]
    )

    grasp_normals = torch.tensor([[0, -1.0, 0.0], [-1, 0.0, 0.0], [1.0, 1.0, 0.0]])
    translation = np.array([-5.7448e-07, -1.2433e-05, 8.1302e-02])
    mu = 1.0


class Xbox360(RigidObject):
    workspace = "xbox360"
    grasp_points = torch.tensor(
        [
            [-0.00693, 0.085422, 0.013867],
            [0.018317, -0.001611, 0.013867],
            [-0.058538, -0.051027, 0.013867],
        ]
    )
    grasp_normals = torch.tensor([[1, -1.5, 0.0], [-2, 1.0, 0.0], [1, 0.0, 0.0]])
    asset_file = "objects/urdf/Xbox360_14e5dba73b283dc7fe0939859a0b15ea.urdf"
    name = "xbox360"
    mu = 1.0
    bound = 2
    translation = np.array([1.7978e-08, -6.0033e-08, 1.0])
