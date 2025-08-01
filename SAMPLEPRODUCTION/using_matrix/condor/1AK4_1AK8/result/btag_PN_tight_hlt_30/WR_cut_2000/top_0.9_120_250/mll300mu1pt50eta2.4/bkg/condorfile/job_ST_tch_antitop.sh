#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_ST_tch_antitop.py
mkdir result_ST_tch_antitop 
cp *.png result_ST_tch_antitop/
cp *.csv result_ST_tch_antitop/
cd result_ST_tch_antitop




