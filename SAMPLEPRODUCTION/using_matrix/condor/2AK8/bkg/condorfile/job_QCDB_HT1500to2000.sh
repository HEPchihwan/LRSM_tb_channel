#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCDB_HT1500to2000.py
mkdir result_QCDB_HT1500to2000 
cp *.png result_QCDB_HT1500to2000/
cd result_QCDB_HT1500to2000




