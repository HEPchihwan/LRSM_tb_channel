#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCD_HT70to100.py
mkdir result_QCD_HT70to100 
cp *.png result_QCD_HT70to100/
cp *.csv result_QCD_HT70to100/
cd result_QCD_HT70to100




