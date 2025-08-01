#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_aaa.py
mkdir result_aaa 
cp *.png result_aaa/
cp *.csv result_aaa/
cd result_aaa




