#!/usr/bin/bash
# Missing: grasp_data/teddy_bear_50_diced.npy
#  'TEDDY_BEAR' 'POWER_DRILL'
for OBJ in 'BANANA' 'BOX' 'BLEACH_CLEANSER'
do
    echo "running command: grasp_test.py --wandb --object ${OBJ} --level-set 50 --dice-grasp \
        --robot-config.gt-normals model-config:mesh"
    python grasp_test.py --wandb --object ${OBJ} --level-set 50 --dice-grasp \
        --robot-config.gt-normals model-config:mesh
    if [[ $? != 0 ]]
    then
        break
    fi
done
