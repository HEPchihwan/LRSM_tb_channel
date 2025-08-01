#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_SingleMuon_B.py
mkdir result_SingleMuon_B 
cp *.png result_SingleMuon_B/
cp *.csv result_SingleMuon_B/
cd result_SingleMuon_B




