#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCD_HT800to1000.py
mkdir result_QCD_HT800to1000 
cp *.png result_QCD_HT800to1000/
cp *.csv result_QCD_HT800to1000/
cd result_QCD_HT800to1000




