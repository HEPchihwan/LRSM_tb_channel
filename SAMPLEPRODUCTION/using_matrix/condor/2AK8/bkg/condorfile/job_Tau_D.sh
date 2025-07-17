#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_Tau_D.py
mkdir result_Tau_D 
cp *.png result_Tau_D/
cd result_Tau_D




