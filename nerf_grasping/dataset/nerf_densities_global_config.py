import numpy as np


# HACK: Just some hardcoded values to reference, make into nice config later
(
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
) = (40, 40, 40)

lb_Oy = np.array([-0.2, -0.2, -0.2])
ub_Oy = np.array([0.2, 0.2, 0.2])

(
    NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
    NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
    NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
) = (30, 30, 30)