#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_ST_tch_top.py
mkdir result_ST_tch_top 
cp *.png result_ST_tch_top/
cd result_ST_tch_top




