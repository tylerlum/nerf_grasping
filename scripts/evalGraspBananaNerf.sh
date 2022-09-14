#!/usr/bin/bash

# for OBJ in 'BANANA' 'BLEACH_CLEANSER' 'BIG_BANANA'
for f in grasp_data/banana_nerf*.npy
do
    python grasp_test.py --wandb --object BANANA --grasp-data $f model-config:nerf
    if [[ $? != 0 ]]
    then
        break
    fi
done
