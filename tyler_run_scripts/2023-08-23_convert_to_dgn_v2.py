import nerf_grasping
from nerf_grasping.optimizer_utils import AllegroGraspConfig
import pathlib
import numpy as np

def main() -> None:
    object_code = "sem-Wii-effdc659515ff747eb2c6725049f8f"
    graspdata_filepath = (pathlib.Path(nerf_grasping.get_repo_root())
        / "graspdata"
        / f"{object_code}.npy"
    )

    stored_data_dicts = np.load(str(graspdata_filepath), allow_pickle=True)
    configs = AllegroGraspConfig.from_grasp_data(
        graspdata_filepath, batch_size=1
    )

    output_data_dicts = configs.to_dexgraspnet_dicts()

    output_dir = pathlib.Path("dexgraspnet_dicts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # HACKILY ADD OBJECT AND SCALES
    scale = stored_data_dicts[0]['scale']
    output_file = pathlib.Path(output_dir) / f"{object_code}_{str(scale).replace('.', '_')}.npy"
    print(f"Saving to {output_file}")
    np.save(
        str(output_file),
        output_data_dicts,
        allow_pickle=True,
    )


if __name__ == "__main__":
    main()
