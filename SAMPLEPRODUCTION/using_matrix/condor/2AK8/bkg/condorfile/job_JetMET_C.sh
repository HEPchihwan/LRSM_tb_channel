#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_JetMET_C.py
mkdir result_JetMET_C 
cp *.png result_JetMET_C/
cd result_JetMET_C




