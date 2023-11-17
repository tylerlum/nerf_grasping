import trimesh
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
import pathlib
import tyro


@dataclass
class Args:
    obj_filepath: pathlib.Path
    opacity: float = 1.0


def main() -> None:
    args = tyro.cli(Args)

    # Load your .obj file
    mesh = trimesh.load(str(args.obj_filepath))

    # Extract vertices and faces
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Create a 3D Plotly figure
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=args.opacity,
            )
        ]
    )

    object_code_and_scale_str = (
        args.obj_filepath.parent.parent.name
        if "_0_" in args.obj_filepath.parent.parent.name
        else args.obj_filepath.stem
    )

    fig.update_layout(
        title=f"{object_code_and_scale_str}",
    )
    fig.show()


if __name__ == "__main__":
    main()
