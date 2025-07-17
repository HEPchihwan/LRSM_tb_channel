#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_Muon_C.py
mkdir result_Muon_C 
cp *.png result_Muon_C/
cd result_Muon_C




