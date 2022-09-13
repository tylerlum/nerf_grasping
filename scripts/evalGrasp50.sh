#!/usr/bin/bash
for OBJ in 'BANANA' 'BOX' 'TEDDY_BEAR' 'POWER_DRILL' 'BLEACH_CLEANSER'
do
    python grasp_test.py --wandb --object ${OBJ} model-config:mesh --model-config.level-set 50
    if [[ $? != 0 ]]
    then
        break
    fi
done
