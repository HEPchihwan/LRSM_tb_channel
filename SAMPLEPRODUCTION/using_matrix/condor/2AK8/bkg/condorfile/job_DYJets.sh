#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_DYJets.py
mkdir result_DYJets 
cp *.png result_DYJets/
cd result_DYJets




