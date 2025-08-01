#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCD_HT600to800.py
mkdir result_QCD_HT600to800 
cp *.png result_QCD_HT600to800/
cp *.csv result_QCD_HT600to800/
cd result_QCD_HT600to800




