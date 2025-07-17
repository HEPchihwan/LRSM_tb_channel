#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCD_HT400to600.py
mkdir result_QCD_HT400to600 
cp *.png result_QCD_HT400to600/
cd result_QCD_HT400to600




