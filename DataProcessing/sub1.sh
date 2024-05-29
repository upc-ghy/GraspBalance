#!/bin/sh
#JSUB -q normal
#JSUB -n 2
#JSUB -m gpu02
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J task1

source /apps/software/anaconda3/bin/activate GraspBalance

# bash command_generate_tolerance_label.sh

python generate_clean_data.py --dataset_root /hpcfiles/users/guihaiyuan/datasetC --num_workers 15  --camera realsense