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


def main(cfg: GraspMetricConfig):
    # PARAMS
    grasp_config_dict_path = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "data"
        / "2023-08-29_evaled_grasp_config_dicts_trial_big"
        / "mug_0_1000.npy"
    )
    grasp_config_dict = np.load(grasp_config_dict_path, allow_pickle=True).item()
    grasp_config = AllegroGraspConfig.from_grasp_config_dict(grasp_config_dict)

    # Create grasp metric
    USE_DEPTH_IMAGES = isinstance(
        cfg.classifier_config.nerfdata_config, DepthImageNerfDataConfig
    )
    if USE_DEPTH_IMAGES:
        grasp_metric = DepthImageGraspMetric.from_config(
            cfg,
        )
    else:
        grasp_metric = GraspMetric.from_config(
            cfg,
        )

    # Evaluate grasp
    scores = grasp_metric.get_failure_probability(grasp_config)
    print(f"Grasp score: {scores}")

    # Ensure grasp_config was not modified
    output_grasp_config_dict = grasp_config.as_dict()
    assert grasp_config_dict.keys() == output_grasp_config_dict.keys()
    for key in grasp_config_dict.keys():
        assert np.allclose(grasp_config_dict[key], output_grasp_config_dict[key])

    # Compare to ground truth
    passed_eval = output_grasp_config_dict["passed_eval"]
    print(f"Passed eval: {passed_eval}")


if __name__ == "__main__":
    cfg = tyro.cli(GraspMetricConfig)
    main(cfg)
