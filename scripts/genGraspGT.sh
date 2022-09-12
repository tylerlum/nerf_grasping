#!/usr/bin/bash
for OBJ in 'BANANA' 'BOX' 'TEDDY_BEAR' 'POWER_DRILL' 'BLEACH_CLEANSER' 'BIG_BANANA'
do
    python generate_grasps.py --object ${OBJ} --num-grasps 20 \
        model-config:nerf --model-config.expected-surface --model-config.expected-gradient
    if [[ $? != 0 ]]
    then
        break
    fi
done
