#!/usr/bin/bash
# for OBJ in 'BANANA' 'BOX' 'TEDDY_BEAR' 'POWER_DRILL' 'BLEACH_CLEANSER' 'BIG_BANANA'
for OBJ in 'BLEACH_CLEANSER' 'BIG_BANANA'
do
    python grasp_test.py --wandb --object ${OBJ} model-config:nerf
    if [[ $? != 0 ]]
    then
        break
    fi
done
