#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_WJets.py
mkdir result_WJets 
cp *.png result_WJets/
cd result_WJets




