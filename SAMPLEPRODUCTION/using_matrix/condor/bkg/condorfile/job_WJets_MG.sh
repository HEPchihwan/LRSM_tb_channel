#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_WJets_MG.py
mkdir result_WJets_MG 
cp *.png result_WJets_MG/
cd result_WJets_MG




