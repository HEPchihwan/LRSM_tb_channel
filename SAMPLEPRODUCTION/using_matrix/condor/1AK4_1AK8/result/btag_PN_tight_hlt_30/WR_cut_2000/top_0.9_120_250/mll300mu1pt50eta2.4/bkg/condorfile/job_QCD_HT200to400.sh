#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCD_HT200to400.py
mkdir result_QCD_HT200to400 
cp *.png result_QCD_HT200to400/
cp *.csv result_QCD_HT200to400/
cd result_QCD_HT200to400




