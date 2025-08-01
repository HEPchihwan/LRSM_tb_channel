#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCDB_HT100to200.py
mkdir result_QCDB_HT100to200 
cp *.png result_QCDB_HT100to200/
cp *.csv result_QCDB_HT100to200/
cd result_QCDB_HT100to200




