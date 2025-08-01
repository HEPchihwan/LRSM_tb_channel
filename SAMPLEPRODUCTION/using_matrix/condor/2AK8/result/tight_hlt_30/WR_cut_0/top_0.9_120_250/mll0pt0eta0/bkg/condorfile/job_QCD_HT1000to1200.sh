#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCD_HT1000to1200.py
mkdir result_QCD_HT1000to1200 
cp *.png result_QCD_HT1000to1200/
cp *.csv result_QCD_HT1000to1200/
cd result_QCD_HT1000to1200




