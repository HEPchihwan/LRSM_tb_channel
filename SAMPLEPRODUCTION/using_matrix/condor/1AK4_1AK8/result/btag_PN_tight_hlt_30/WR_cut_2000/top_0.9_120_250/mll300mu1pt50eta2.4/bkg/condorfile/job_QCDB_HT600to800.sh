#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_QCDB_HT600to800.py
mkdir result_QCDB_HT600to800 
cp *.png result_QCDB_HT600to800/
cp *.csv result_QCDB_HT600to800/
cd result_QCDB_HT600to800




