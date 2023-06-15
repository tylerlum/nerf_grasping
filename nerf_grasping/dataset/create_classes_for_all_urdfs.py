import os
from tqdm import tqdm
import h5py

"""
Example:
INPUT: 

```
{input_dir}
├── objects
│   ├── urdf
│   │   ├── 1Shelves_12a64182bbaee7a12b2444829a3507de.urdf
│   │   ├── 1Shelves_160684937ae737ec5057ad0f363d6ddd.urdf
│   │   ├── 1Shelves_1e3df0ab57e8ca8587f357007f9e75d1.urdf
│   │   ├── 1Shelves_2b9d60c74bc0d18ad8eae9bce48bbeed.urdf
```

```
{input_acronym_dir}
├── 1Shelves_12a64182bbaee7a12b2444829a3507de_0.00914554366969263.h5
├── 1Shelves_160684937ae737ec5057ad0f363d6ddd_0.009562610447288044.h5
├── 1Shelves_1e3df0ab57e8ca8587f357007f9e75d1_0.011099225885734912.h5
├── 1Shelves_2b9d60c74bc0d18ad8eae9bce48bbeed_0.00614208274225087.h5
```

Note: many object categories (like 1Shelves) start with a number, which is not allowed in python class names.
So we need to add a prefix to the class name.

Note 2: We may need to change mu and bound for some objects.

Note 3: some objects have multiple urdfs to account for different scales (from acronym dataset)
We include this in the class name, but replace the "." with an "_" to make it a valid python class name.


OUTPUT:

{output_py_file}

```
class Obj_Xbox360_435f39e98d2260f0d6e21b8525c3f8bb_0_002061950217848804(RigidObject):
    workspace = "Xbox360_435f39e98d2260f0d6e21b8525c3f8bb_0.002061950217848804"
    grasp_points = None
    grasp_normals = None
    asset_file = "objects/urdf/Xbox360_435f39e98d2260f0d6e21b8525c3f8bb.urdf"
    name = "Xbox360_435f39e98d2260f0d6e21b8525c3f8bb_0.002061950217848804"
    mu = 1.0  # From acronym dataset
    bound = 2  # TODO: Need to properly set this
    translation = np.zeros(3)  # TODO: Need to properly set this
    acronym_file = "Xbox360_435f39e98d2260f0d6e21b8525c3f8bb_0.002061950217848804.h5"
    mesh_scale = 0.002061950217848804
...
```

"""


# Create a class for each urdf in the dir
input_dir = "/juno/u/tylerlum/github_repos/nerf_grasping/assets/"
input_acronym_dir = "/juno/u/tylerlum/github_repos/acronym/data/grasps/"
output_py_file = "/juno/u/tylerlum/github_repos/nerf_grasping/acronym_objects.py"

print("=" * 100)
print("PARAMS")
print("=" * 100)
print(f"input_dir: {input_dir}")
print(f"input_acronym_dir: {input_acronym_dir}")
print(f"output_py_file: {output_py_file}")
print()

# Read in all urdf files
urdf_dir = os.path.join(input_dir, "objects", "urdf")

# Check inputs
if not os.path.exists(urdf_dir):
    print(f"urdf_dir: {urdf_dir} does not exist. Exiting.")
    exit()

if not os.path.exists(input_acronym_dir):
    print(f"input_acronym_dir: {input_acronym_dir} does not exist. Exiting.")
    exit()

if os.path.exists(output_py_file):
    print(f"output_py_file: {output_py_file} already exists. Exiting.")
    exit()


print("=" * 100)
print("URDF FILES")
print("=" * 100)
urdf_files = os.listdir(urdf_dir)
print(f"Found {len(urdf_files)} urdf files.")
print(f"First 10 urdf files: {urdf_files[:10]}")
print()

print("=" * 100)
print("ACRONYM FILES")
print("=" * 100)
acronym_files = os.listdir(input_acronym_dir)
print(f"Found {len(acronym_files)} acronym files.")
print(f"First 10 acronym files: {acronym_files[:10]}")
print()

with open(output_py_file, "w") as f:
    # STARTING COMMENT
    f.write(f"### START OF FILE: {output_py_file} ###\n")

    for urdf_file in tqdm(urdf_files):
        urdf_name, ext = os.path.splitext(urdf_file)
        if ext != ".urdf":
            print(f"WARNING: Skipping {urdf_file} because it is not a urdf file.")
            continue

        # Each urdf file may have multiple scales
        related_acronym_files = [f for f in acronym_files if urdf_name in f]
        if len(related_acronym_files) == 0:
            print(
                f"WARNING: Skipping {urdf_file} because it has no related acronym files."
            )
            continue

        for related_acronym_file in related_acronym_files:
            # Get scale from acronym data
            acronym_data = h5py.File(
                os.path.join(input_acronym_dir, related_acronym_file), "r"
            )
            mesh_scale = float(acronym_data["object/scale"][()])

            # Create class
            # Need :f to avoid scientific notation in class name
            urdf_text = "\n".join(
                [
                    f"class Obj_{urdf_name}_{mesh_scale:f}(RigidObject):".replace(
                        ".", "_"
                    ),
                    f'    workspace = "{urdf_name}_{mesh_scale}"',
                    "    grasp_points = None",
                    "    grasp_normals = None",
                    f'    asset_file = "objects/urdf/{urdf_name}.urdf"',
                    f'    name = "{urdf_name}_{mesh_scale}"',
                    "    mu = 1.0  # From acronym dataset",
                    "    bound = 2  # TODO: Need to properly set this",
                    "    translation = np.zeros(3)  # TODO: Need to properly set this",
                    f'    acronym_file = "{related_acronym_file}"',
                    f"    mesh_scale = {mesh_scale}",
                    "",
                    "",
                ]
            )
            f.write(urdf_text)
    f.write(f"### END OF FILE: {output_py_file} ###\n")
