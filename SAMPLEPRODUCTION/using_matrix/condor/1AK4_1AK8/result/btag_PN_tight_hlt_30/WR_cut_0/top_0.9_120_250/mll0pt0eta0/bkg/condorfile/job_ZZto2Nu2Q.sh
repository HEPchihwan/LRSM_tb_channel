#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_ZZto2Nu2Q.py
mkdir result_ZZto2Nu2Q 
cp *.png result_ZZto2Nu2Q/
cd result_ZZto2Nu2Q




