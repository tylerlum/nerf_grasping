from dataclasses import dataclass
from typing import Tuple
from collections import OrderedDict
import tyro
from typing import Union


@dataclass(frozen=True)
class BaseFingertipConfig:
    num_pts_x: int
    num_pts_y: int
    num_pts_z: int
    finger_width_mm: float = 10.0
    finger_height_mm: float = 15.0
    grasp_depth_mm: float = 20.0
    n_fingers: int = 4


@dataclass(frozen=True)
class VanillaFingertipConfig(BaseFingertipConfig):
    pass


# Not frozen, since we need to modify the num_pts_x, num_pts_y, num_pts_z in the custom constructor.
@dataclass(frozen=True)
class EvenlySpacedFingertipConfig(BaseFingertipConfig):
    distance_between_pts_mm: float = 0.5

    @classmethod
    def from_dimensions(
        cls,
        distance_between_pts_mm: float = 0.5,
        finger_width_mm: float = 10.0,
        finger_height_mm: float = 15.0,
        grasp_depth_mm: float = 20.0,
    ):
        num_pts_x = int(finger_width_mm / distance_between_pts_mm) + 1
        num_pts_y = int(finger_height_mm / distance_between_pts_mm) + 1
        num_pts_z = int(grasp_depth_mm / distance_between_pts_mm) + 1

        return cls(
            num_pts_x=num_pts_x,
            num_pts_y=num_pts_y,
            num_pts_z=num_pts_z,
            finger_width_mm=finger_width_mm,
            finger_height_mm=finger_height_mm,
            grasp_depth_mm=grasp_depth_mm,
            distance_between_pts_mm=distance_between_pts_mm,
        )


UnionFingertipConfig = Union[VanillaFingertipConfig, EvenlySpacedFingertipConfig]


@dataclass
class TopLevelConfig:
    fingertip_config: UnionFingertipConfig = (
        EvenlySpacedFingertipConfig.from_dimensions()
    )


if __name__ == "__main__":
    cfg = tyro.cli(TopLevelConfig)
    print(cfg)
