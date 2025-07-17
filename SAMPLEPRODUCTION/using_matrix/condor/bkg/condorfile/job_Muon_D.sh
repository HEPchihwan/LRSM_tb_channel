#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_Muon_D.py
mkdir result_Muon_D 
cp *.png result_Muon_D/
cd result_Muon_D




