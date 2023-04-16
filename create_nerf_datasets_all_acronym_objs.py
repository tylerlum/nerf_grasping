import subprocess
from nerf_grasping.sim import acronym_objects
from tqdm import tqdm

"""
Create nerf dataset for all objects in the acronym dataset

For example:
```
python sim_trifinger.py --get_nerf_training_data --obj Obj_Fish_5e6656b0f124e0f38af30108ea9ccb6c_0_019030 --num_steps_before_collecting 50 --overwrite
```
"""

acronym_object_classes = [k for k in acronym_objects.__dict__.keys() if k.startswith("Obj_")]
print(f"Found {len(acronym_object_classes)} objects in the acronym dataset")
print(f"First 10: {acronym_object_classes[:10]}")

for acronym_object_class in tqdm(acronym_object_classes):
    command = f"python sim_trifinger.py --get_nerf_training_data --obj {acronym_object_class} --num_steps_before_collecting 50 --overwrite"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)