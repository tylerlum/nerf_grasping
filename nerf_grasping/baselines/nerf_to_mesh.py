from pathlib import Path
from typing import Callable, Optional, Tuple
from typing_extensions import Unpack

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes
from nerf_grasping.grasp_utils import load_nerf_field, get_nerf_configs


def sdf_to_mesh(
    sdf: Callable[[torch.Tensor], float],
    npts: int = 31,
    lb: np.ndarray = -np.ones(3),
    ub: np.ndarray = np.ones(3),
) -> Unpack[Tuple[np.ndarray, ...]]:
    """Converts an SDF to a mesh using marching cubes.

    Parameters
    ----------
    sdf : Callable[[torch.Tensor], float]
        A signed distance field. The input into this function has shape (B, 3).
    npts : int, default=31
        Number of points used to grid space in each dimension for marching cubes.
    lb : np.ndarray, default=-np.ones(3)
        Lower bound for marching cubes.
    ub : np.ndarray, default=np.ones(3)
        Upper bound for marching cubes.


    Returns
    -------
    verts : np.ndarray, shape=(nverts, 3)
        The vertices of the mesh.
    faces : np.ndarray, shape=(nfaces, 3), type=int
        The vertex indices associated with the corners of each face.
    normals : np.ndarray, shape=(nfaces, 3)
        The surface normals associated with each face.
    """
    # running marching cubes to extract the isosurface
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, Z = np.mgrid[
        lb[0] : ub[0] : npts * 1j,
        lb[1] : ub[1] : npts * 1j,
        lb[2] : ub[2] : npts * 1j,
    ]
    pts_plot = np.stack((X, Y, Z), axis=-1)  # (npts, npts, npts, 3)
    pts_plot_flat = torch.tensor(pts_plot.reshape((-1, 3)), device=device)  # (B, 3)
    vol = np.array(sdf(pts_plot_flat).reshape(pts_plot.shape[:-1]))
    _verts, faces, normals, _ = marching_cubes(vol, 0.0, allow_degenerate=False)
    verts = (ub - lb) * _verts / (np.array(X.shape) - 1) + lb  # scaling verts properly
    return verts, faces, normals


def nerf_to_mesh(
    field,
    level: float,
    npts: int = 31,
    lb: np.ndarray = -np.ones(3),
    ub: np.ndarray = np.ones(3),
    scale: float = 1.0,
    min_len: Optional[float] = None,
    flip_faces: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    """Takes a nerfstudio pipeline field and plots or saves a mesh.

    Parameters
    ----------
    field
        The nerfstudio field.
    level : float
        The density level to treat as the 0-level set.
    npts : int, default=31
        Number of points used to grid space in each dimension for marching cubes.
    lb : np.ndarray, default=-np.ones(3)
        Lower bound for marching cubes.
    ub : np.ndarray, default=np.ones(3)
        Upper bound for marching cubes.
    scale : float, default=1.0
        The scale to apply to the mesh.
    min_len : Optional[float], default=None
        Minimum number of edges to be considered a relevant component, used to remove floaters
    flip_faces : bool, default=True
        Whether to flip the faces, helps get correct signed distance values
        (it appears that the faces are flipped inside out by default)
    save_path : Optional[Path], default=None
        The save path. If None, shows a plot instead.
    """
    # marching cubes
    sdf = lambda x: field.density_fn(x).cpu().detach().numpy() - level
    verts, faces, normals = sdf_to_mesh(
        sdf,
        npts=npts,
        lb=lb,
        ub=ub,
    )

    if flip_faces:
        faces = np.fliplr(faces)

    # making a trimesh mesh
    mesh = trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_normals=normals, process=False
    )
    mesh.apply_transform(trimesh.transformations.scale_matrix(scale))
    if min_len is not None:
        cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=min_len)
        mask = np.zeros(len(mesh.faces), dtype=bool)
        mask[np.concatenate(cc)] = True
        mesh.update_faces(mask)

    # saving/visualizing
    if save_path is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        _ = ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            faces,
            verts[:, 2],
        )
        ax.set_aspect("equal")
        plt.show()
    else:
        mesh.export(save_path)


if __name__ == "__main__":
    NERF_CHECKPOINTS_PATH = Path("data/2023-09-11_20-52-40/nerfcheckpoints")
    IDX = 0
    RADIUS = 0.1
    LEVEL = 10.0

    nerf_configs = get_nerf_configs(
        nerf_checkpoints_path=str(NERF_CHECKPOINTS_PATH),
    )
    nerf_config = nerf_configs[IDX]
    field = load_nerf_field(nerf_config)

    lb = -RADIUS * np.ones(3)
    ub = RADIUS * np.ones(3)

    # [EXAMPLE] seeing a plot of the mesh
    nerf_to_mesh(
        field,
        # level=10.0,  # VERY IMPORTANT TO ADJUST!
        level=LEVEL,
        npts=31,
        # lb=np.array([-0.1, -0.1, 0.0]),  # VERY IMPORTANT TO ADJUST!
        # ub=np.array([0.1, 0.1, 0.3]),  # VERY IMPORTANT TO ADJUST!
        lb=lb,
        ub=ub,
        save_path=None,
    )

    # [EXAMPLE] saving the mesh
    nerf_to_mesh(
        field,
        # level=10.0,
        level=LEVEL,
        npts=31,
        # lb=np.array([-0.1, -0.1, 0.0]),
        # ub=np.array([0.1, 0.1, 0.3]),
        lb=lb,
        ub=ub,
        save_path=Path(f"./{nerf_config.stem}.obj"),
    )
