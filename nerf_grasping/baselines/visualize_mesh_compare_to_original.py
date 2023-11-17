import trimesh
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
import pathlib
import tyro
from typing import Tuple


@dataclass
class Args:
    obj_filepath: pathlib.Path
    opacity: float = 1.0
    original_meshdata_dir_path: pathlib.Path = (
        pathlib.Path("~/github_repos/DexGraspNet/data/meshdata").expanduser().resolve()
    )  # TODO: change this per workstation


def parse_object_code_and_scale(object_code_and_scale_str: str) -> Tuple[str, float]:
    keyword = "_0_"
    idx = object_code_and_scale_str.rfind(keyword)
    object_code = object_code_and_scale_str[:idx]

    idx_offset_for_scale = keyword.index("0")
    object_scale = float(
        object_code_and_scale_str[idx + idx_offset_for_scale :].replace("_", ".")
    )
    return object_code, object_scale


def create_mesh_3d(
    vertices: np.ndarray, faces: np.ndarray, opacity: float = 1.0
) -> go.Mesh3d:
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=opacity,
    )


def main() -> None:
    args = tyro.cli(Args)
    assert args.obj_filepath.exists(), f"{args.obj_filepath} does not exist"
    assert (
        args.original_meshdata_dir_path.exists()
    ), f"{args.original_meshdata_dir_path} does not exist"

    object_code_and_scale_str = (
        args.obj_filepath.parent.parent.name
        if "_0_" in args.obj_filepath.parent.parent.name
        else args.obj_filepath.stem
    )
    object_code, object_scale = parse_object_code_and_scale(object_code_and_scale_str)

    original_mesh_filepath = (
        args.original_meshdata_dir_path / object_code / "coacd" / "decomposed.obj"
    ).resolve()
    assert original_mesh_filepath.exists(), f"{original_mesh_filepath} does not exist"

    # Load your .obj file
    mesh = trimesh.load(str(args.obj_filepath))
    original_mesh = trimesh.load(str(original_mesh_filepath))

    # Extract vertices and faces
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    original_vertices = np.array(original_mesh.vertices)
    original_faces = np.array(original_mesh.faces)

    # Create a 3D Plotly figure
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "mesh3d"}, {"type": "mesh3d"}]],
        subplot_titles=(
            f"{args.obj_filepath.stem}",
            f"{args.obj_filepath.stem} (original)",
        ),
    )
    fig.add_trace(
        create_mesh_3d(vertices, faces, opacity=args.opacity),
        row=1,
        col=1,
    )
    fig.add_trace(
        create_mesh_3d(original_vertices, original_faces, opacity=args.opacity),
        row=1,
        col=2,
    )
    fig.show()


if __name__ == "__main__":
    main()
