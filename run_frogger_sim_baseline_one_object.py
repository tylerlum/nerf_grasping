import time
from nerf_grasping.frogger_utils import custom_coll_callback
import tyro
from typing import Optional, Tuple, List, Literal, Callable
from nerfstudio.models.base_model import Model
from nerf_grasping.run_pipeline import (
    transform_point,
    add_transform_matrix_traces,
)
from nerf_grasping.grasp_utils import load_nerf_pipeline
from nerf_grasping.nerfstudio_train import train_nerfs_return_trainer
from nerf_grasping.baselines.nerf_to_mesh import nerf_to_mesh
from nerf_grasping.nerf_utils import (
    compute_centroid_from_nerf,
)
import trimesh
import pathlib
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FroggerConfig:
    num_grasps: int = 10
    object_code: str = "unnamed_object"
    output_folder: pathlib.Path = pathlib.Path(
        "sim_experiments"
    ) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    density_levelset_threshold: float = 15.0
    obj_is_z_up: bool = False
    lb_x: float = -0.2
    lb_y: float = 0.0
    lb_z: float = -0.2
    ub_x: float = 0.2
    ub_y: float = 0.3
    ub_z: float = 0.2
    object_scale: float = 0.9999
    nerf_frame_offset_x: float = 0.65
    visualize: bool = False

    @property
    def lb_N(self) -> np.ndarray:
        return np.array([self.lb_x, self.lb_y, self.lb_z])

    @property
    def ub_N(self) -> np.ndarray:
        return np.array([self.ub_x, self.ub_y, self.ub_z])

    @property
    def X_W_Nz(self) -> np.ndarray:
        return trimesh.transformations.translation_matrix(
            [self.nerf_frame_offset_x, 0, 0]
        )

    @property
    def X_O_Oy(self) -> np.ndarray:
        return (
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            if self.obj_is_z_up
            else np.eye(4)
        )

    @property
    def X_O_Oz(self) -> np.ndarray:
        return (
            np.eye(4)
            if self.obj_is_z_up
            else trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        )

    @property
    def X_N_Nz(self) -> np.ndarray:
        return (
            np.eye(4)
            if self.obj_is_z_up
            else trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        )

    @property
    def object_code_and_scale_str(self) -> str:
        return f"{self.object_code}_{self.object_scale:.4f}".replace(".", "_")


def compute_frogger_grasps_v2(
    nerf_model: Model,
    cfg: FroggerConfig,
    custom_coll_callback: Callable,
    max_time: float,
) -> None:
    print("=" * 80)
    print("Step 1: Figuring out frames")
    print("=" * 80 + "\n")
    print("Frames are W = world, N = nerf, O = object, Oy = object y-up, H = hand")
    print(
        "W is centered at the robot base. N is centered where origin of NeRF data collection is. O is centered at the object centroid. Oy is centered at the object centroid. H is centered at the base of the middle finger"
    )
    print(
        "W, N, O are z-up frames. Oy is y-up. H has z-up along finger and x-up along palm normal"
    )
    print("X_A_B represents 4x4 transformation matrix of frame B wrt A")
    X_W_Nz, X_O_Oy = cfg.X_W_Nz, cfg.X_O_Oy
    X_N_Nz = cfg.X_N_Nz
    X_Nz_N = np.linalg.inv(X_N_Nz)
    X_W_N = X_W_Nz @ X_Nz_N
    lb_N, ub_N = cfg.lb_N, cfg.ub_N

    print("\n" + "=" * 80)
    print("Step 2: Get NERF")
    print("=" * 80 + "\n")
    nerf_field = nerf_model.field

    print("\n" + "=" * 80)
    print("Step 3: Convert NeRF to mesh")
    print("=" * 80 + "\n")
    nerf_to_mesh_folder = cfg.output_folder / "nerf_to_mesh" / cfg.object_code / "coacd"
    nerf_to_mesh_folder.mkdir(parents=True, exist_ok=True)
    mesh_N = nerf_to_mesh(
        field=nerf_field,
        level=cfg.density_levelset_threshold,
        lb=lb_N,
        ub=ub_N,
        save_path=nerf_to_mesh_folder / "decomposed.obj",
    )

    # Save to /tmp/mesh_viz_object.obj as well
    mesh_N.export("/tmp/mesh_viz_object.obj")

    print("\n" + "=" * 80)
    print(
        "Step 4: Compute X_N_Oy (transformation of the object y-up frame wrt the nerf frame)"
    )
    print("=" * 80 + "\n")
    USE_MESH = False
    mesh_centroid_N = mesh_N.centroid
    nerf_centroid_N = compute_centroid_from_nerf(
        nerf_field,
        lb=lb_N,
        ub=ub_N,
        level=cfg.density_levelset_threshold,
        num_pts_x=100,
        num_pts_y=100,
        num_pts_z=100,
    )
    print(f"mesh_centroid_N: {mesh_centroid_N}")
    print(f"nerf_centroid_N: {nerf_centroid_N}")
    centroid_N = mesh_centroid_N if USE_MESH else nerf_centroid_N
    print(f"USE_MESH: {USE_MESH}, centroid_N: {centroid_N}")
    assert centroid_N.shape == (3,), f"centroid_N.shape is {centroid_N.shape}, not (3,)"
    X_N_O = trimesh.transformations.translation_matrix(centroid_N)

    X_N_Oy = X_N_O @ X_O_Oy
    X_Oy_N = np.linalg.inv(X_N_Oy)
    assert X_N_Oy.shape == (4, 4), f"X_N_Oy.shape is {X_N_Oy.shape}, not (4, 4)"

    mesh_W = trimesh.Trimesh(vertices=mesh_N.vertices, faces=mesh_N.faces)
    mesh_W.apply_transform(X_W_N)

    # For debugging
    mesh_Oy = trimesh.Trimesh(vertices=mesh_N.vertices, faces=mesh_N.faces)
    mesh_Oy.apply_transform(X_Oy_N)
    nerf_to_mesh_Oy_folder = (
        cfg.output_folder / "nerf_to_mesh_Oy" / cfg.object_code / "coacd"
    )
    nerf_to_mesh_Oy_folder.mkdir(parents=True, exist_ok=True)
    mesh_Oy.export(nerf_to_mesh_Oy_folder / "decomposed.obj")

    X_W_O = X_W_N @ X_N_O
    X_O_W = np.linalg.inv(X_W_O)
    mesh_O = trimesh.Trimesh(vertices=mesh_W.vertices, faces=mesh_W.faces)
    mesh_O.apply_transform(X_O_W)

    # O is a bit ambiguous, it can be Oy or Oz
    X_O_Oz = cfg.X_O_Oz
    X_Oz_O = np.linalg.inv(X_O_Oz)

    X_W_Oz = X_W_O @ X_O_Oz
    mesh_Oz = trimesh.Trimesh(vertices=mesh_O.vertices, faces=mesh_O.faces)
    mesh_Oz.apply_transform(X_Oz_O)

    if cfg.visualize:
        # Visualize N
        fig_N = go.Figure()
        fig_N.add_trace(
            go.Mesh3d(
                x=mesh_N.vertices[:, 0],
                y=mesh_N.vertices[:, 1],
                z=mesh_N.vertices[:, 2],
                i=mesh_N.faces[:, 0],
                j=mesh_N.faces[:, 1],
                k=mesh_N.faces[:, 2],
                color="lightblue",
                name="Mesh",
                opacity=0.5,
            )
        )
        fig_N.add_trace(
            go.Scatter3d(
                x=[mesh_centroid_N[0]],
                y=[mesh_centroid_N[1]],
                z=[mesh_centroid_N[2]],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Mesh centroid",
            )
        )
        fig_N.add_trace(
            go.Scatter3d(
                x=[nerf_centroid_N[0]],
                y=[nerf_centroid_N[1]],
                z=[nerf_centroid_N[2]],
                mode="markers",
                marker=dict(size=10, color="green"),
                name="NeRF centroid",
            )
        )
        fig_N.update_layout(title="Mesh in nerf frame")
        add_transform_matrix_traces(fig=fig_N, transform_matrix=np.eye(4), length=0.1)
        fig_N.show()

        # Visualize Oz
        fig_Oz = go.Figure()
        fig_Oz.add_trace(
            go.Mesh3d(
                x=mesh_Oz.vertices[:, 0],
                y=mesh_Oz.vertices[:, 1],
                z=mesh_Oz.vertices[:, 2],
                i=mesh_Oz.faces[:, 0],
                j=mesh_Oz.faces[:, 1],
                k=mesh_Oz.faces[:, 2],
                color="lightblue",
                name="Mesh",
                opacity=0.5,
            )
        )
        fig_Oz.update_layout(title="Mesh in object z-up frame")
        add_transform_matrix_traces(fig=fig_Oz, transform_matrix=np.eye(4), length=0.1)
        fig_Oz.show()

    print("\n" + "=" * 80)
    print("Step 5: Run frogger")
    print("=" * 80 + "\n")

    from nerf_grasping import frogger_utils

    # Careful: We must use Oz for frogger
    # The mesh should be in O
    # The X_W_O should be X_W_Oz so it is oriented upwards
    frogger_args = frogger_utils.FroggerArgs(
        obj_filepath=nerf_to_mesh_folder / "decomposed.obj",
        obj_scale=cfg.object_scale,
        obj_name=cfg.object_code,
        obj_is_yup=False,
        num_grasps=cfg.num_grasps,
        output_grasp_config_dicts_folder=cfg.output_folder / "grasp_config_dicts",
        visualize=cfg.visualize,
        grasp_idx_to_visualize=0,
        max_time=max_time,
    )
    frogger_utils.frogger_to_grasp_config_dict(
        args=frogger_args,
        mesh=mesh_N,
        X_W_O=X_W_N,
        custom_coll_callback=custom_coll_callback,
    )
    return


@dataclass
class CommandlineArgs(FroggerConfig):
    nerfdata_path: Optional[pathlib.Path] = None
    nerfcheckpoint_path: Optional[pathlib.Path] = None
    max_num_iterations: int = 400

    def __post_init__(self) -> None:
        if self.nerfdata_path is not None and self.nerfcheckpoint_path is None:
            assert self.nerfdata_path.exists(), f"{self.nerfdata_path} does not exist"
            assert (
                self.nerfdata_path / "transforms.json"
            ).exists(), f"{self.nerfdata_path / 'transforms.json'} does not exist"
            assert (
                self.nerfdata_path / "images"
            ).exists(), f"{self.nerfdata_path / 'images'} does not exist"

            object_code_and_scale_str = self.nerfdata_path.name
        elif self.nerfdata_path is None and self.nerfcheckpoint_path is not None:
            assert (
                self.nerfcheckpoint_path.exists()
            ), f"{self.nerfcheckpoint_path} does not exist"
            assert (
                self.nerfcheckpoint_path.suffix == ".yml"
            ), f"{self.nerfcheckpoint_path} does not have a .yml suffix"
            object_code_and_scale_str = self.nerfcheckpoint_path.parents[2].name
        else:
            raise ValueError(
                "Exactly one of nerfdata_path or nerfcheckpoint_path must be specified"
            )

        # Update object_code and object_scale
        idx = object_code_and_scale_str.index("_0_")
        self.object_code = object_code_and_scale_str[:idx]
        self.object_scale = float(object_code_and_scale_str[idx + 3:].replace("_", "."))



def main() -> None:
    args = tyro.cli(CommandlineArgs)
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")

    # Prepare nerf model
    if args.nerfdata_path is not None:
        start_time = time.time()
        nerf_checkpoints_folder = args.output_folder / "nerfcheckpoints"
        nerf_trainer = train_nerfs_return_trainer.train_nerf(
            args=train_nerfs_return_trainer.Args(
                nerfdata_folder=args.nerfdata_path,
                nerfcheckpoints_folder=nerf_checkpoints_folder,
                max_num_iterations=args.max_num_iterations,
            )
        )
        nerf_model = nerf_trainer.pipeline.model
        end_time = time.time()
        print("@" * 80)
        print(f"Time to train_nerf: {end_time - start_time:.2f}s")
        print("@" * 80 + "\n")
    elif args.nerfcheckpoint_path is not None:
        start_time = time.time()
        nerf_pipeline = load_nerf_pipeline(args.nerfcheckpoint_path)
        nerf_model = nerf_pipeline.model
        end_time = time.time()
        print("@" * 80)
        print(f"Time to load_nerf_pipeline: {end_time - start_time:.2f}s")
        print("@" * 80 + "\n")
    else:
        raise ValueError(
            "Exactly one of nerfdata_path or nerfcheckpoint_path must be specified"
        )

    compute_frogger_grasps_v2(
        nerf_model=nerf_model,
        cfg=args,
        custom_coll_callback=custom_coll_callback,
        max_time=20.0,
    )


if __name__ == "__main__":
    main()
