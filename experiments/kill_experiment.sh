#!/bin/sh

for i in $(seq 26 35)
do
   screen -X -S "${i}a" quit
done

