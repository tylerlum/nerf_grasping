#!/usr/bin/bash
for OBJ in 'BOX' 'TEDDY_BEAR' 'POWER_DRILL' 'BLEACH_CLEANSER'
do
    python generate_grasps.py --wandb --object ${OBJ} --num-grasps 20 model-config:mesh
    if [[ $? != 0 ]]
    then
        break
    fi
done
