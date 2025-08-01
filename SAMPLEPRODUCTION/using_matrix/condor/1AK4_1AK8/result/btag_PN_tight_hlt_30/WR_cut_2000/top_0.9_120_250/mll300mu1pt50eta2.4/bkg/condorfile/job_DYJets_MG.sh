#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_DYJets_MG.py
mkdir result_DYJets_MG 
cp *.png result_DYJets_MG/
cp *.csv result_DYJets_MG/
cd result_DYJets_MG




