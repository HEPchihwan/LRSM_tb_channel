#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_EGamma_C.py
mkdir result_EGamma_C 
cp *.png result_EGamma_C/
cd result_EGamma_C




