#!/usr/bin/bash
# 'BANANA' 'BOX'
for OBJ in 'TEDDY_BEAR' 'POWER_DRILL' 'BLEACH_CLEANSER' 'BIG_BANANA'
do
    python generate_grasps.py --object ${OBJ} --num-grasps 20 --level-set 50\
        --dice-grasp model-config:mesh
    if [[ $? != 0 ]]
    then
        break
    fi
done
