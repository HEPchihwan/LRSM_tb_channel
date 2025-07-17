#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_WZto3LNu.py
mkdir result_WZto3LNu 
cp *.png result_WZto3LNu/
cd result_WZto3LNu




