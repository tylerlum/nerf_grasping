# %%
import h5py
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# %%
grid_val_path = pathlib.Path("/home/tylerlum/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/grid_dataset_v3/train_dataset.h5")
assert grid_val_path.exists()

# %%
def get_nerf_densities(i: int) -> np.ndarray:
    with h5py.File(grid_val_path, "r") as f:
        densities = f["nerf_densities"][i]
    return densities

# %%
densities = get_nerf_densities(0)
print(densities.shape)
# %%
plt.hist(densities.reshape(-1), bins=100, log=True)
plt.xlabel("Density")
plt.ylabel("Count")

# %%
densities = np.stack([get_nerf_densities(i) for i in range(10000)])
print(densities.shape)

# %%
plt.hist(densities.reshape(-1), bins=100, log=True)
plt.xlabel("Density")
plt.ylabel("Count")

# %%
plt.hist(densities.reshape(-1), bins=100)
plt.xlabel("Density")
plt.ylabel("Count")

# %%
num_low_density = np.sum(densities < 0.1)
print(f"num_low_density: {num_low_density}")
print(f"num_total: {densities.size}")
print(f"fraction_low_density: {num_low_density / densities.size}")


# %%
num_low_density = np.sum(densities < 1)
print(f"num_low_density: {num_low_density}")
print(f"num_total: {densities.size}")
print(f"fraction_low_density: {num_low_density / densities.size}")
# %%
num_low_density = np.sum(densities < 10)
print(f"num_low_density: {num_low_density}")
print(f"num_total: {densities.size}")
print(f"fraction_low_density: {num_low_density / densities.size}")

# %%
num_low_density = np.sum(densities < 100)
print(f"num_low_density: {num_low_density}")
print(f"num_total: {densities.size}")
print(f"fraction_low_density: {num_low_density / densities.size}")
# %%
