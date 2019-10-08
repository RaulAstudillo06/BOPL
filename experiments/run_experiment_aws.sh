#!/bin/sh

for i in $(seq 9 9)
do
   screen -dmS "${i}a"
   sleep 5
   screen -S "${i}a" -X stuff "source activate boenv\n"
   screen -S "${i}a" -X stuff "source activate boenv\n"
   screen -S "${i}a" -X stuff "python /home/ubuntu/BOPU/experiments/test_ambulances1.py $i\n"
done

