#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCD_HT2000toInf.py
mkdir result_QCD_HT2000toInf 
cp *.png result_QCD_HT2000toInf/
cd result_QCD_HT2000toInf




