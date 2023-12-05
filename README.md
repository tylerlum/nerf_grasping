# Grasping with NeRFs

This project focuses on performing grasping and manipulation using
Neural Radiance Fields (NeRFs).

# How to run (2023-12-04)

First, follow the instructions at DexGraspNet to run an experiment. Should have something like the following in DexGraspNet

```
ls ../data/2023-12-04_rubikscube_one_object
augmented_raw_evaled_grasp_config_dicts_opened_hand  augmented_raw_hand_config_dicts_opened_hand  nerfdata
augmented_raw_grasp_config_dicts_opened_hand         evaled_grasp_config_dicts                    raw_evaled_grasp_config_dicts
augmented_raw_hand_config_dicts_closed_hand          hand_config_dicts                            raw_grasp_config_dicts
```

## Train nerfs and create dataset

Everything should be run from the root directory of this repo.

First create a directory to store data:
```
mkdir data
```

Then prepare the DGN data by:
* Creating symlinks to the DexGraspNet data
* Training nerfs on the nerfdata
* Making train/val/test split of `evaled_grasp_config_dicts` (train/val/test across objects)

```
python prepare_dgn_experiment.py --experiment_name 2023-12-04_rubikscube_one_object --train_nerfs --dgn_data_path <PATH TO DEXGRASPNET DATA FOLDER eg. ~/github_repos/DexGraspNet/data>
```

You should now see the roughly the following:
```
ls data/2023-12-04_rubikscube_one_object
augmented_raw_grasp_config_dicts  evaled_grasp_config_dicts_test   grasp_config_dicts  nerfcheckpoints                raw_grasp_config_dicts
depth_image_dataset               evaled_grasp_config_dicts_train  grid_dataset        nerfdata
evaled_grasp_config_dicts         evaled_grasp_config_dicts_val    hand_config_dicts   raw_evaled_grasp_config_dicts
```

The main outputs are:
* `depth_image_dataset/dataset.h5`
* `grid_dataset/dataset.h5`

## Train models

Density grid inputs:
```
python nerf_grasping/learned_metric/Train_DexGraspNet_NeRF_Grasp_Metric.py cnn-2d-1d --task-type PASSED_SIMULATION_AND_PENETRATION_THRESHOLD --nerfdata-config.output-filepath data/2023-12-04_rubikscube_one_object/grid_dataset/dataset.h5 --dataloader.batch-size 128 --wandb.name grid_cnn2d1d --checkpoint-workspace.output_leaf_dir_name grid_cnn2d1d
```

Depth image inputs:
```
python nerf_grasping/learned_metric/Train_DexGraspNet_NeRF_Grasp_Metric.py depth-cnn-2d --task-type PASSED_SIMULATION_AND_PENETRATION_THRESHOLD --nerfdata-config.output-filepath data/2023-12-04_rubikscube_one_object/depth_image_dataset/dataset.h5 --wandb.name depth_cnn2d --checkpoint-workspace.output_leaf_dir_name depth_cnn2d
```

Outputs checkpoints:
```
ls Train_DexGraspNet_NeRF_Grasp_Metric_workspaces/depth_cnn2d

checkpoint_0005.pt  checkpoint_0115.pt  checkpoint_0225.pt  checkpoint_0335.pt  checkpoint_0445.pt  checkpoint_0555.pt  checkpoint_0665.pt
checkpoint_0010.pt  checkpoint_0120.pt  checkpoint_0230.pt  checkpoint_0340.pt  checkpoint_0450.pt  checkpoint_0560.pt  checkpoint_0670.pt
checkpoint_0015.pt  checkpoint_0125.pt  checkpoint_0235.pt  checkpoint_0345.pt  checkpoint_0455.pt  checkpoint_0565.pt  checkpoint_0675.pt
checkpoint_0020.pt  checkpoint_0130.pt  checkpoint_0240.pt  checkpoint_0350.pt  checkpoint_0460.pt  checkpoint_0570.pt  checkpoint_0680.pt
checkpoint_0025.pt  checkpoint_0135.pt  checkpoint_0245.pt  checkpoint_0355.pt  checkpoint_0465.pt  checkpoint_0575.pt  checkpoint_0685.pt
checkpoint_0030.pt  checkpoint_0140.pt  checkpoint_0250.pt  checkpoint_0360.pt  checkpoint_0470.pt  checkpoint_0580.pt  checkpoint_0690.pt
checkpoint_0035.pt  checkpoint_0145.pt  checkpoint_0255.pt  checkpoint_0365.pt  checkpoint_0475.pt  checkpoint_0585.pt  checkpoint_0695.pt
checkpoint_0040.pt  checkpoint_0150.pt  checkpoint_0260.pt  checkpoint_0370.pt  checkpoint_0480.pt  checkpoint_0590.pt  checkpoint_0700.pt
checkpoint_0045.pt  checkpoint_0155.pt  checkpoint_0265.pt  checkpoint_0375.pt  checkpoint_0485.pt  checkpoint_0595.pt  checkpoint_0705.pt
checkpoint_0050.pt  checkpoint_0160.pt  checkpoint_0270.pt  checkpoint_0380.pt  checkpoint_0490.pt  checkpoint_0600.pt  checkpoint_0710.pt
checkpoint_0055.pt  checkpoint_0165.pt  checkpoint_0275.pt  checkpoint_0385.pt  checkpoint_0495.pt  checkpoint_0605.pt  checkpoint_0715.pt
checkpoint_0060.pt  checkpoint_0170.pt  checkpoint_0280.pt  checkpoint_0390.pt  checkpoint_0500.pt  checkpoint_0610.pt  checkpoint_0720.pt
checkpoint_0065.pt  checkpoint_0175.pt  checkpoint_0285.pt  checkpoint_0395.pt  checkpoint_0505.pt  checkpoint_0615.pt  checkpoint_0725.pt
checkpoint_0070.pt  checkpoint_0180.pt  checkpoint_0290.pt  checkpoint_0400.pt  checkpoint_0510.pt  checkpoint_0620.pt  checkpoint_0730.pt
checkpoint_0075.pt  checkpoint_0185.pt  checkpoint_0295.pt  checkpoint_0405.pt  checkpoint_0515.pt  checkpoint_0625.pt  checkpoint_0735.pt
checkpoint_0080.pt  checkpoint_0190.pt  checkpoint_0300.pt  checkpoint_0410.pt  checkpoint_0520.pt  checkpoint_0630.pt  checkpoint_0740.pt
checkpoint_0085.pt  checkpoint_0195.pt  checkpoint_0305.pt  checkpoint_0415.pt  checkpoint_0525.pt  checkpoint_0635.pt  checkpoint_0745.pt
checkpoint_0090.pt  checkpoint_0200.pt  checkpoint_0310.pt  checkpoint_0420.pt  checkpoint_0530.pt  checkpoint_0640.pt  checkpoint_0750.pt
checkpoint_0095.pt  checkpoint_0205.pt  checkpoint_0315.pt  checkpoint_0425.pt  checkpoint_0535.pt  checkpoint_0645.pt  checkpoint_0755.pt
checkpoint_0100.pt  checkpoint_0210.pt  checkpoint_0320.pt  checkpoint_0430.pt  checkpoint_0540.pt  checkpoint_0650.pt  config.yaml
checkpoint_0105.pt  checkpoint_0215.pt  checkpoint_0325.pt  checkpoint_0435.pt  checkpoint_0545.pt  checkpoint_0655.pt  wandb_run_id.txt
checkpoint_0110.pt  checkpoint_0220.pt  checkpoint_0330.pt  checkpoint_0440.pt  checkpoint_0550.pt  checkpoint_0660.pt
```

Can load checkpoint with `--checkpoint-workspace.input_leaf_dir_name depth_cnn2d`

Look at wandb to see results!

## Test Grasp Metric

Test out how well the grasp metric is doing on a file of labeled grasps:
```
python nerf_grasping/test_grasp_metric.py --grasp_config_dict_path <path_to_npy> --max_num_grasps 10 --grasp_metric.classifier_config_path <path to classifier config yaml> --grasp_metric.nerf_checkpoint_path <path to nerf checkpoint config yaml>
```

Prints to screen comparing predicted scores and ground truth.

## Optimizer

Run SGD optimizer:
```
python nerf_grasping/optimizer.py --use-rich optimizer:sgd --optimizer.num-grasps 5 --init_grasp_config_dict_path <path_to_npy> --grasp_metric.classifier_config_path <path to classifier config yaml> --grasp_metric.nerf_checkpoint_path <path to nerf checkpoint config yaml>                                    
```

Saves to npy file with the optimized grasps

## NeRF to Mesh + URDF

After training nerfs, create meshes and urdfs like so:
```
python nerf_grasping/baselines/nerf_to_urdf_all.py --nerfcheckpoints_path data/2023-11-17_01-27-23/nerfcheckpoints --output-dir-path data/2023-11-17_01-27-23/nerf_meshdata_mugs_v10
```

Output:
```
ls data/2023-11-17_01-27-23/nerf_meshdata_mugs_v10
core-mug-1038e4eac0e18dcce02ae6d2a21d494a  core-mug-1305b9266d38eb4d9f818dd0aa1a251   core-mug-162201dfe14b73f0281365259d1cf342
core-mug-10f6e09036350e92b3f21f1137c3c347  core-mug-141f1db25095b16dcfb3760e4293e310  core-mug-17952a204c0a9f526c69dceb67157a66
core-mug-127944b6dabee1c9e20e92c5b8147e4a  core-mug-159e56c18906830278d8f8c02c47cde0
core-mug-128ecbc10df5b05d96eaf1340564a4de  core-mug-15bd6225c209a8e3654b0ce7754570c8

tree data/2023-11-17_01-27-23/nerf_meshdata_mugs_v10/core-mug-1038e4eac0e18dcce02ae6d2a21d494a
data/2023-11-17_01-27-23/nerf_meshdata_mugs_v10/core-mug-1038e4eac0e18dcce02ae6d2a21d494a
└── coacd
    ├── coacd.urdf
    └── decomposed.obj
```

This can be used by DexGraspNet for baselines.

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
