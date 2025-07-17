#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_TTLJ_powheg.py
mkdir result_TTLJ_powheg 
cp *.png result_TTLJ_powheg/
cd result_TTLJ_powheg




