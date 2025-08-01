#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCD_HT1200to1500.py
mkdir result_QCD_HT1200to1500 
cp *.png result_QCD_HT1200to1500/
cp *.csv result_QCD_HT1200to1500/
cd result_QCD_HT1200to1500




