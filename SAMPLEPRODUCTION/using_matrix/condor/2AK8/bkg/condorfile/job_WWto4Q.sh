#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_WWto4Q.py
mkdir result_WWto4Q 
cp *.png result_WWto4Q/
cd result_WWto4Q




