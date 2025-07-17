#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCDB_HT1000to1500.py
mkdir result_QCDB_HT1000to1500 
cp *.png result_QCDB_HT1000to1500/
cd result_QCDB_HT1000to1500




