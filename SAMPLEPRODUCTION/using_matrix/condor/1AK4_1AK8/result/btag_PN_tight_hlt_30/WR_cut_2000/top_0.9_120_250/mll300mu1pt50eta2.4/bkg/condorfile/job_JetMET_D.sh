#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_JetMET_D.py
mkdir result_JetMET_D 
cp *.png result_JetMET_D/
cd result_JetMET_D




