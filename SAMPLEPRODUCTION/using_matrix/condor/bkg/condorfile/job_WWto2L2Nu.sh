#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_WWto2L2Nu.py
mkdir result_WWto2L2Nu 
cp *.png result_WWto2L2Nu/
cd result_WWto2L2Nu




