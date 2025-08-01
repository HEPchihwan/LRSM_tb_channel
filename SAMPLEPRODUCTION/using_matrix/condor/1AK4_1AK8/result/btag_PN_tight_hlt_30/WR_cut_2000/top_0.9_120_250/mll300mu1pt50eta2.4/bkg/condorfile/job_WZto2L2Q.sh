#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_WZto2L2Q.py
mkdir result_WZto2L2Q 
cp *.png result_WZto2L2Q/
cp *.csv result_WZto2L2Q/
cd result_WZto2L2Q




