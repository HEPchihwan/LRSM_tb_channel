#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_EGamma_D.py
mkdir result_EGamma_D 
cp *.png result_EGamma_D/
cp *.csv result_EGamma_D/
cd result_EGamma_D




