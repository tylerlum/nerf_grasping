# %%
from nerf_grasping.nerfstudio_train.train_nerfs_return_trainer import train_nerf, Args
import pathlib

trainer = train_nerf(
    args = Args(
        nerfdata_folder=pathlib.Path(
            "experiments/2024-04-15_DEBUG/nerfdata/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846"
        ),
        nerfcheckpoints_folder=pathlib.Path(
            "experiments/2024-04-15_DEBUG/nerfcheckpoints/"
        ),
        is_real_world=False,
    )
)
# %%
trainer.pipeline.model.field
# %%
[x for x in trainer.config.get_base_dir().iterdir()]

# %%

