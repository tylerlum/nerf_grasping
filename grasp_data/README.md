### Grasp Data

These files contain grasps which were optimized offline.

Each file contains a single np array `grasps`, size [B, 3, 6], where B is the batch dimension (so each entry is the result of a different run of CEM).

`grasps[ii, jj, :3]` will be the origin / starting point of finger `jj` for the `ii`th grasp, and `grasps[ii, jj, 3:]` will be the approach direction, expressed in world frame (i.e., not relative; no need to transform).