import pathlib
from typing import Tuple


def get_table_collision_dict() -> dict:
    return {
        "cuboid": {
            "table": {
                "dims": [1.3208, 1.8288, 0.02],
                "pose": [0.50165, 0.0, -0.01, 1.0, 0.0, 0.0, 0.0],
            },
        }
    }


def get_object_collision_dict(
    file_path: pathlib.Path,
    xyz: Tuple[float, float, float],
    quat_wxyz: Tuple[float, float, float, float],
) -> dict:
    return {
        "mesh": {
            "object": {
                "pose": [*xyz, *quat_wxyz],
                "file_path": str(file_path),
            }
        }
    }


def get_dummy_collision_dict() -> dict:
    FAR_AWAY_POS = 10.0
    return {
        "cuboid": {
            "dummy": {
                "dims": [0.1, 0.1, 0.1],
                "pose": [FAR_AWAY_POS, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            },
        }
    }
