#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_SingleMuon_C.py
mkdir result_SingleMuon_C 
cp *.png result_SingleMuon_C/
cd result_SingleMuon_C




