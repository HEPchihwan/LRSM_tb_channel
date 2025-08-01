#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_EGamma_B.py
mkdir result_EGamma_B 
cp *.png result_EGamma_B/
cp *.csv result_EGamma_B/
cd result_EGamma_B




