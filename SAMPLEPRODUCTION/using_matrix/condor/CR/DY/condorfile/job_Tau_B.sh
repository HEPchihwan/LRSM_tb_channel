#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_Tau_B.py
mkdir result_Tau_B 
cp *.png result_Tau_B/
cp *.csv result_Tau_B/
cd result_Tau_B




