#!/usr/bin/bash
for OBJ in 'TEDDY_BEAR' 'POWER_DRILL'
do
    python generate_grasps.py --object ${OBJ} --regen --num-grasps 10 \
        model-config:mesh; echo "${OBJ} finished"
    if [[ $? != 0 ]]
    then
        break
    fi
done
