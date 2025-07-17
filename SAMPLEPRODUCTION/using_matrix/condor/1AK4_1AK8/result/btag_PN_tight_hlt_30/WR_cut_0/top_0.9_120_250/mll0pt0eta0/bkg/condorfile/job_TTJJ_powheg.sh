#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_TTJJ_powheg.py
mkdir result_TTJJ_powheg 
cp *.png result_TTJJ_powheg/
cd result_TTJJ_powheg




