#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_WWtoLNu2Q.py
mkdir result_WWtoLNu2Q 
cp *.png result_WWtoLNu2Q/
cp *.csv result_WWtoLNu2Q/
cd result_WWtoLNu2Q




