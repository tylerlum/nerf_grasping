from isaacgym import gymapi, gymtorch
from pathlib import Path
import os
import numpy as np
import torch

import trimesh
import scipy

from nerf_grasping.quaternions import Quaternion
from nerf_grasping import grasp_utils

from nerf import utils

root_dir = Path(os.path.abspath(__file__)).parents[2]
asset_dir = f"{root_dir}/assets"


def load_nerf(workspace, bound, scale):
    parser = utils.get_config_parser()
    args = parser.parse_args(
        [
            "--workspace",
            f"{root_dir}/torch-ngp/logs/{workspace}",
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
    model = grasp_utils.load_nerf(args)
    return model


class RigidObject:

    obj_scale = 1e-2
    scale = 1.0
    bound = 2.0
    centroid = np.zeros((3, 1))
    use_centroid = False
    mu = 1.0

    def __init__(self, gym=None, sim=None, env=None):
        self.gym = gym
        self.sim = sim
        self.env = env

        if self.sim is not None:
            self.asset = self.create_asset()
            self.actor = self.configure_actor(gym, env)

        self.nerf_loaded = False
        self.load_trimesh()

    def get_CG(self):
        pos = self.rb_states[0, :3]
        quat = Quaternion.fromWLast(self.rb_states[0, 3:7])
        return quat.rotate(self.CG) + pos

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
        self.model = load_nerf(self.workspace, self.bound, self.scale)
        self.nerf_loaded = True

    def load_trimesh(self):
        mesh_path = os.path.join(asset_dir, self.mesh_file)
        self.gt_mesh = trimesh.load(mesh_path, force="mesh")
        # R = scipy.spatial.transform.Rotation.from_euler("Y", [-np.pi / 2]).as_matrix()
        # R = (
        #     R
        #     @ scipy.spatial.transform.Rotation.from_euler("X", [-np.pi / 2]).as_matrix()
        # )
        # T_rot = np.eye(4)
        # T_rot[:3, :3] = R
        # self.gt_mesh.apply_transform(T_rot)
        self.gt_mesh.apply_scale(self.obj_scale)
        self.gt_mesh.apply_translation(self.translation)

    def create_asset(self):
        asset_options = gymapi.AssetOptions()

        asset_options.vhacd_enabled = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        # asset_options.density = 100
        asset_options.override_inertia = False
        asset_options.override_com = False

        asset_options.vhacd_params.mode = 0
        asset_options.vhacd_params.resolution = 300000
        asset_options.vhacd_params.max_convex_hulls = 30
        asset_options.vhacd_params.max_num_vertices_per_ch = 16

        asset = self.gym.load_asset(self.sim, asset_dir, self.asset_file, asset_options)

        rs_props = self.gym.get_asset_rigid_shape_properties(asset)
        for p in rs_props:
            p.friction = self.mu
            p.torsion_friction = self.mu
            p.restitution = 0.0
        self.gym.set_asset_rigid_shape_properties(asset, rs_props)

        return asset

    def configure_actor(self, gym, env):
        actor = self.gym.create_actor(
            env,
            self.asset,
            gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.1)),
            self.name,
            0,
            0,
            segmentationId=2,
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
    mesh_file = "objects/meshes/isaac_teddy/isaac_bear.obj"
    obj_scale = 1e-2
    name = "teddy_bear"
    workspace = "teddy_bear"

    data_dir = f"{root_dir}/nerf_shared/data/isaac_teddy"
    config_path = f"{root_dir}/nerf_shared/configs/isaac_teddy.txt"
    centroid = np.array([-0.0001444, 0.00412231, 0.08663063])
    use_centroid = False
    translation = np.array([-1.2824e-05,  6.9302e-06,  2.2592e-03])

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
    obj_scale = 0.075
    translation = np.array([1.6316e-07, -6.7600e-07,  3.9500e-02])

    grasp_normals = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    mesh_file = "objects/meshes/cube_multicolor.obj"
    asset_file = "objects/urdf/cube_multicolor.urdf"

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
#             segmentationId=2,
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

    workspace = "powerdrill"
    centroid = np.zeros(3)
    grasp_points = torch.tensor(
        [
            [-0.038539, 0.115021, 0.023878],
            [0.030017, -0.002467, 0.027816],
            [-0.029284, -0.099212, 0.027294],
        ]
    )

    grasp_normals = torch.tensor([[0, -1.0, 0.0], [-1, 0.0, 0.0], [1.0, 1.0, 0.0]])

    asset_file = "objects/urdf/power_drill.urdf"
    name = "power_drill"


class Banana(RigidObject):
    workspace = "banana"
    grasp_points = torch.tensor(
        [
            [-0.00693, 0.085422, 0.013867],
            [0.018317, -0.001611, 0.013867],
            [-0.058538, -0.051027, 0.013867],
        ]
    )
    obj_scale = 1.0
    grasp_normals = torch.tensor([[1, -1.5, 0.0], [-2, 1.0, 0.0], [1, 0.0, 0.0]])
    asset_file = "objects/urdf/banana.urdf"
    mesh_file = "objects/meshes/banana/textured.obj"
    name = "banana"
    mu = 1.
    translation = np.array([-1.4408e-05,  3.8640e-06,  2.7102e-03])


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
