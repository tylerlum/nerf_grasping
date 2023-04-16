import os
from tqdm import tqdm

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

Note: many object categories (like 1Shelves) start with a number, which is not allowed in python class names.
So we need to add a prefix to the class name.

Note 2: We may need to change mu and bound for some objects.

OUTPUT:

{output_py_file}

class Obj_Xbox360_435f39e98d2260f0d6e21b8525c3f8bb(RigidObject):
    workspace = "Xbox360_435f39e98d2260f0d6e21b8525c3f8bb"
    grasp_points = None
    grasp_normals = None
    asset_file = "objects/urdf/Xbox360_435f39e98d2260f0d6e21b8525c3f8bb.urdf"
    name = "Xbox360_435f39e98d2260f0d6e21b8525c3f8bb"
    mu = 1.0
    bound = 2
    translation = np.array([0.0, 0.0, 0.0])  # Need to properly set this


"""

# Create a class for each urdf in the dir
input_dir = "/juno/u/tylerlum/github_repos/nerf_grasping/assets/"
output_py_file = "/juno/u/tylerlum/github_repos/nerf_grasping/acronym_objects.py"

print("=" * 100)
print("PARAMS")
print("=" * 100)
print(f"input_dir: {input_dir}")
print(f"output_py_file: {output_py_file}")
print()

# Read in all urdf files
urdf_dir = os.path.join(input_dir, "objects", "urdf")

# Check inputs
if not os.path.exists(urdf_dir):
    print(f"urdf_dir: {urdf_dir} does not exist. Exiting.")
    exit()

if os.path.exists(output_py_file):
    print(f"output_py_file: {output_py_file} already exists. Exiting.")
    exit()

urdf_files = os.listdir(urdf_dir)
with open(output_py_file, "w") as f:
    # STARTING COMMENT
    f.write(f"### START OF FILE: {output_py_file} ###\n")
    for urdf_file in tqdm(urdf_files):
        urdf_name, ext = os.path.splitext(urdf_file)
        if ext != ".urdf":
            print(f"WARNING: Skipping {urdf_file} because it is not a urdf file.")
            continue

        urdf_text = "\n".join(
            [
                f"class Obj_{urdf_name}(RigidObject):",
                f"    workspace = \"{urdf_name}\"",
                "    grasp_points = None",
                "    grasp_normals = None",
                f"    asset_file = \"objects/urdf/{urdf_name}.urdf\"",
                f"    name = \"{urdf_name}\"",
                "    mu = 1.0",
                "    bound = 2",
                "    translation = np.array([0.0, 0.0, 0.0])  # Need to properly set this",
                "",
                "",
            ]
        )

        f.write(urdf_text)
    f.write(f"### END OF FILE: {output_py_file} ###\n")
