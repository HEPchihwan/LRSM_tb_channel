#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCDB_HT40to100.py
mkdir result_QCDB_HT40to100 
cp *.png result_QCDB_HT40to100/
cd result_QCDB_HT40to100




