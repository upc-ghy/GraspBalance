#!/bin/sh
#JSUB -q normal
#JSUB -n 1
#JSUB -m gpu01
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J task1

source /apps/software/anaconda3/bin/activate Graspbalance

python3.7 setup.py install
