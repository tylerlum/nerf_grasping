# Grasping with NeRFs

This project focuses on performing grasping and manipulation using
Neural Radiance Fields (NeRFs).

# System Diagram

View diagrams [here](diagrams.md)

# Tyler Updates (2023-06-13)

Mostly follow same README setup with a few changes. Some changes include:

Switch to this fork of torch-ngp (a few small changes): https://github.com/tylerlum/torch-ngp

## 2. ACRONYM Dataset (ONLY IF WORKING WITH ACRONYM DATASET)

Follow instructions from here: https://github.com/tylerlum/acronym

Populate `nerf_grasping/assets/objects/urdf` and `nerf_grasping/assets/objects/meshes`

## 3. ACRONYM NeRF Dataset Collection (ONLY IF WORKING WITH ACRONYM DATASET)

Run:
```
cd nerf_grasping/dataset
python create_classes_for_all_urdfs.py
```

NOTE: Need to change the `input_acronym_dir' to actual path to ACRONYM dataset.

This creates acronym_objects.py (already have one in this repo as example, but you should replace it).

Then run:
```
python create_nerf_datasets_all_acronym_objs.py
```

This creates nerf training data for all objects.

You can also run `Visualize_NeRF_Mesh_Grasps.ipynb` to understand what is going on.

Some paths may be hardcoded or invalid. May need some adjustments. Sorry in advance.

## 4. NeRF Training

Move all nerf data to `torch-ngp/data`. Should have rough structure like:

```
torch-ngp/data/isaac_Knife_208610aa6d607c90e3f7a74e12a274ef_3.93127e-05
├── col_100.png
├── col_101.png
├── col_102.png
├── col_103.png
├── col_104.png
├── col_105.png
├── col_106.png
├── col_107.png
├── col_108.png
├── col_109.png
├── dep_100.png
├── dep_101.png
├── dep_102.png
├── dep_103.png
├── dep_104.png
├── dep_105.png
├── dep_106.png
├── dep_107.png
├── dep_108.png
├── dep_109.png
├── seg_100.png
├── seg_101.png
├── seg_102.png
├── seg_103.png
├── seg_104.png
├── seg_105.png
├── seg_106.png
├── seg_107.png
├── seg_108.png
├── seg_109.png
── test
│   ├── 123.png
│   ├── 133.png
│   ├── 137.png
│   ├── 139.png
│   ├── 141.png
│   ├── 146.png
│   ├── 14.png
│   ├── 183.png
│   ├── 198.png
│   ├── 202.png
│   ├── 206.png
│   ├── 218.png
│   ├── 220.png
│   ├── 221.png
│   ├── 223.png
│   ├── 229.png
│   ├── 239.png
│   ├── 24.png
│   ├── 2.png
│   ├── 38.png
│   ├── 40.png
│   ├── 44.png
│   ├── 49.png
│   ├── 67.png
│   ├── 6.png
│   ├── 76.png
│   └── 99.png
├── train
│   ├── 100.png
│   ├── 101.png
│   ├── 103.png
│   ├── 104.png
│   ├── 105.png
│   ├── 106.png
│   ├── 107.png
│   ├── 108.png
│   ├── 109.png
│   ├── 10.png
│   ├── 110.png
│   ├── 111.png
│   ├── 112.png
│   ├── 113.png
│   ├── 115.png
│   ├── 116.png
│   ├── 117.png
│   ├── 119.png
│   ├── 120.png
│   ├── 122.png
│   ├── 124.png
│   ├── 125.png
│   ├── 127.png
│   ├── 128.png
│   ├── 129.png
│   ├── 130.png
│   ├── 131.png
│   ├── 132.png
│   ├── 134.png
│   ├── 135.png
│   ├── 136.png
│   ├── 138.png
│   ├── 13.png
│   ├── 140.png
│   ├── 142.png
│   ├── 143.png
│   ├── 144.png
│   ├── 145.png
│   ├── 147.png
│   ├── 148.png
│   ├── 149.png
│   ├── 151.png
│   ├── 152.png
│   ├── 153.png
│   ├── 155.png
│   ├── 156.png
│   ├── 157.png
│   ├── 158.png
│   ├── 159.png
│   ├── 15.png
│   ├── 160.png
│   ├── 161.png
│   ├── 162.png
│   ├── 163.png
│   ├── 164.png
│   ├── 165.png
│   ├── 166.png
│   ├── 167.png
│   ├── 168.png
│   ├── 16.png
│   ├── 170.png
│   ├── 171.png
│   ├── 172.png
│   ├── 173.png
│   ├── 174.png
│   ├── 176.png
│   ├── 177.png
│   ├── 178.png
│   ├── 179.png
│   ├── 180.png
│   ├── 181.png
│   ├── 182.png
│   ├── 184.png
│   ├── 185.png
│   ├── 186.png
│   ├── 187.png
│   ├── 188.png
│   ├── 189.png
│   ├── 18.png
│   ├── 190.png
│   ├── 191.png
│   ├── 192.png
│   ├── 193.png
│   ├── 194.png
│   ├── 195.png
│   ├── 196.png
│   ├── 197.png
│   ├── 1.png
│   ├── 200.png
│   ├── 201.png
│   ├── 203.png
│   ├── 204.png
│   ├── 205.png
│   ├── 207.png
│   ├── 208.png
│   ├── 209.png
│   ├── 20.png
│   ├── 210.png
│   ├── 211.png
│   ├── 212.png
│   ├── 213.png
│   ├── 214.png
│   ├── 215.png
│   ├── 216.png
│   ├── 217.png
│   ├── 219.png
│   ├── 21.png
│   ├── 222.png
│   ├── 224.png
│   ├── 225.png
│   ├── 226.png
│   ├── 227.png
│   ├── 228.png
│   ├── 22.png
│   ├── 231.png
│   ├── 232.png
│   ├── 233.png
│   ├── 234.png
│   ├── 235.png
│   ├── 236.png
│   ├── 237.png
│   ├── 238.png
│   ├── 23.png
│   ├── 240.png
│   ├── 241.png
│   ├── 242.png
│   ├── 243.png
│   ├── 244.png
│   ├── 246.png
│   ├── 247.png
│   ├── 248.png
│   ├── 249.png
│   ├── 250.png
│   ├── 251.png
│   ├── 252.png
│   ├── 253.png
│   ├── 254.png
│   ├── 255.png
│   ├── 25.png
│   ├── 27.png
│   ├── 28.png
│   ├── 29.png
│   ├── 30.png
│   ├── 31.png
│   ├── 32.png
│   ├── 33.png
│   ├── 36.png
│   ├── 37.png
│   ├── 39.png
│   ├── 3.png
│   ├── 41.png
│   ├── 42.png
│   ├── 43.png
│   ├── 45.png
│   ├── 46.png
│   ├── 47.png
│   ├── 4.png
│   ├── 50.png
│   ├── 51.png
│   ├── 52.png
│   ├── 53.png
│   ├── 54.png
│   ├── 55.png
│   ├── 56.png
│   ├── 58.png
│   ├── 59.png
│   ├── 5.png
│   ├── 60.png
│   ├── 61.png
│   ├── 62.png
│   ├── 63.png
│   ├── 64.png
│   ├── 65.png
│   ├── 66.png
│   ├── 68.png
│   ├── 69.png
│   ├── 70.png
│   ├── 71.png
│   ├── 72.png
│   ├── 73.png
│   ├── 74.png
│   ├── 75.png
│   ├── 77.png
│   ├── 78.png
│   ├── 79.png
│   ├── 7.png
│   ├── 80.png
│   ├── 81.png
│   ├── 82.png
│   ├── 83.png
│   ├── 84.png
│   ├── 85.png
│   ├── 86.png
│   ├── 88.png
│   ├── 8.png
│   ├── 90.png
│   ├── 91.png
│   ├── 92.png
│   ├── 94.png
│   ├── 95.png
│   ├── 96.png
│   ├── 97.png
│   ├── 98.png
│   └── 9.png
├── transforms_test.json
├── transforms_train.json
├── transforms_val.json
└── val
    ├── 0.png
    ├── 102.png
    ├── 114.png
    ├── 118.png
    ├── 11.png
    ├── 121.png
    ├── 126.png
    ├── 12.png
    ├── 150.png
    ├── 154.png
    ├── 169.png
    ├── 175.png
    ├── 17.png
    ├── 199.png
    ├── 19.png
    ├── 230.png
    ├── 245.png
    ├── 26.png
    ├── 34.png
    ├── 35.png
    ├── 48.png
    ├── 57.png
    ├── 87.png
    ├── 89.png
    └── 93.png
```

Then go to `torch-ngp` and run:

```
python create_nerf_models_all_acronym_objs.py
```

May need to change a few parameters:

* path to the folders of nerf data to be trained
* bound and scale parameters (may be object dependent)

This creates `nerf_checkpoints` with nerf model information.

## 5. Learned Metric Dataset Generation

Run:
```
python Create_NeRF_ACRONYM_Dataset_HDF5.py
```

To create hdf5 file using the above grasp dataset and nerf models for learned metric training.

## 6. Learned Metric Training

Run:
```
python Train_NeRF_Grasp_Metric.py
```

Check the config for hyperparameters. Can run with ipynb.


# Below: Previous README

### Setup

#### Python Installation
To install, first clone this repo, using
```
git clone --recurse-submodules https://github.com/pculbertson/nerf_grasping
```
then follow the instructions [here](https://github.com/stanford-iprl-lab/nerf_shared/)
to install the `nerf_shared` python package and its dependencies.

Note: I made `nerf_shared` a submodule since it's still less-than-stable, and it
might be nice to still edit the package / push commits to the `nerf_shared` repo.

Finally, install this package's dependencies by running
```
pip install -r requirements.txt
```
(If running on google cloud, install using `pip install -r gcloud-requirements.txt`)

#### Data Setup

The current experiment notebook uses some data generated by Adam using Blender.
You can request access to both the training data and a trained model from him.

Once you have access to the data, copy the following files:

1. Copy all of the checkpoint files in `nerf_checkpoints/*` to `torch-ngp/data/logs`.

2. From the `nerf_training_data` folder on Google Drive, copy the directory
`blender_datasets/teddy_bear_dataset/teddy_bear` into
`nerf_grasping/nerf_shared/data/nerf_synthetic/teddy_bear`.

This should be all you need to run the example notebook!

#### Other Setup

An important note: you need to have Blender installed to run the mesh union/intersection
operations required to compute the mesh IoU metric. You can do this per the instructions [here](https://docs.blender.org/manual/en/latest/getting_started/installing/linux.html).

### References

The trifinger robot URDF, and some enviroment setup code is from [https://github.com/pairlab/leibnizgym](https://github.com/pairlab/leibnizgym)
