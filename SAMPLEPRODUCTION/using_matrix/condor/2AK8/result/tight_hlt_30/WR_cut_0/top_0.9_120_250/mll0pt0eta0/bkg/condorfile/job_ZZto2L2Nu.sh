#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_ZZto2L2Nu.py
mkdir result_ZZto2L2Nu 
cp *.png result_ZZto2L2Nu/
cp *.csv result_ZZto2L2Nu/
cd result_ZZto2L2Nu




