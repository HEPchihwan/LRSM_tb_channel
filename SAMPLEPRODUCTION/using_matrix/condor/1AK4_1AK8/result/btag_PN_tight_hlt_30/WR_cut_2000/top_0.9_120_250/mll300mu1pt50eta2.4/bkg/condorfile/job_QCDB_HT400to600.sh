#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCDB_HT400to600.py
mkdir result_QCDB_HT400to600 
cp *.png result_QCDB_HT400to600/
cp *.csv result_QCDB_HT400to600/
cd result_QCDB_HT400to600




