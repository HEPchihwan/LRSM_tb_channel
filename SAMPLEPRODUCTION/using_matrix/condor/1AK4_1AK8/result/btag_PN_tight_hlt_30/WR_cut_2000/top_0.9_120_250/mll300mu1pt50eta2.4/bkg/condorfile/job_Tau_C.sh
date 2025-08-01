#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_Tau_C.py
mkdir result_Tau_C 
cp *.png result_Tau_C/
cp *.csv result_Tau_C/
cd result_Tau_C




