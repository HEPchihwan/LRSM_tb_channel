#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_WZtoLNu2Q.py
mkdir result_WZtoLNu2Q 
cp *.png result_WZtoLNu2Q/
cp *.csv result_WZtoLNu2Q/
cd result_WZtoLNu2Q




