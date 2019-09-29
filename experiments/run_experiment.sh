#!/bin/sh

for i in $(seq 1 5)
do
   screen -dmS "${i}a"
   sleep 1
   screen -S "${i}a" -X stuff "conda activate conda_bo_env\n"
   screen -S "${i}a" -X stuff "python /home/ra598/Raul/Projects/BOPU/experiments/test_GP2.py $i\n"
done

