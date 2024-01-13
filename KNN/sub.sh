#!/bin/sh
#JSUB -q normal
#JSUB -n 1
#JSUB -m gpu01
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J task1

python setup.py install
# python3 test.py
