#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_Jpsito2Mu.py
mkdir result_Jpsito2Mu 
cp *.png result_Jpsito2Mu/
cp *.csv result_Jpsito2Mu/
cd result_Jpsito2Mu




