from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
from nerf_grasping.optimizer_utils import (
    GraspMetric,
    DepthImageGraspMetric,
    AllegroGraspConfig,
)
from nerf_grasping.config.nerfdata_config import DepthImageNerfDataConfig
import nerf_grasping
import tyro
import pathlib
import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class Args:
    grasp_metric: GraspMetricConfig
    grasp_config_dict_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "data"
        / "2023-11-17_rubikscube_0"
        / "evaled_grasp_config_dicts"
        / "ddg-gd_rubik_cube_poisson_004_0_1000.npy"
    )
    max_num_grasps: Optional[int] = 10


def main(cfg: Args):
    grasp_config_dict = np.load(cfg.grasp_config_dict_path, allow_pickle=True).item()

    if cfg.max_num_grasps is not None:
        print(f"Limiting number of grasps to {cfg.max_num_grasps}")
        for key in grasp_config_dict.keys():
            grasp_config_dict[key] = grasp_config_dict[key][:cfg.max_num_grasps]

    grasp_config = AllegroGraspConfig.from_grasp_config_dict(grasp_config_dict)

    # Create grasp metric
    USE_DEPTH_IMAGES = isinstance(
        cfg.grasp_metric.classifier_config.nerfdata_config, DepthImageNerfDataConfig
    )
    if USE_DEPTH_IMAGES:
        grasp_metric = DepthImageGraspMetric.from_config(
            cfg.grasp_metric,
        )
    else:
        grasp_metric = GraspMetric.from_config(
            cfg.grasp_metric,
        )

    # Evaluate grasp
    scores = grasp_metric.get_failure_probability(grasp_config).tolist()
    print(f"Grasp score: {scores}")

    # Ensure grasp_config was not modified
    output_grasp_config_dict = grasp_config.as_dict()
    assert output_grasp_config_dict.keys()
    for key, val in output_grasp_config_dict.items():
        assert np.allclose(val, grasp_config_dict[key], atol=1e-5, rtol=1e-5), f"Key {key} was modified!"

    # Compare to ground truth
    passed_eval = grasp_config_dict["passed_eval"]
    print(f"Passed eval: {passed_eval}")


if __name__ == "__main__":
    cfg = tyro.cli(Args)
    main(cfg)
