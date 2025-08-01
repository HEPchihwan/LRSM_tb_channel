#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCD_HT100to200.py
mkdir result_QCD_HT100to200 
cp *.png result_QCD_HT100to200/
cp *.csv result_QCD_HT100to200/
cd result_QCD_HT100to200




