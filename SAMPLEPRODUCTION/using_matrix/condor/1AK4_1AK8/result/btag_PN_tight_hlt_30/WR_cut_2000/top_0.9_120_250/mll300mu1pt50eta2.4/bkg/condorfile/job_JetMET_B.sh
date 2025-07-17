#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_JetMET_B.py
mkdir result_JetMET_B 
cp *.png result_JetMET_B/
cd result_JetMET_B




