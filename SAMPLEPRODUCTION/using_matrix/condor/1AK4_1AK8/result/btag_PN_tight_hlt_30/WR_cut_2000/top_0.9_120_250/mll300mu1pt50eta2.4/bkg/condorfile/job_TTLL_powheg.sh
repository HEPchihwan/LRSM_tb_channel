#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_TTLL_powheg.py
mkdir result_TTLL_powheg 
cp *.png result_TTLL_powheg/
cp *.csv result_TTLL_powheg/
cd result_TTLL_powheg




