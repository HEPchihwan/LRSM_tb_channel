#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_HtoZZto4L.py
mkdir result_HtoZZto4L 
cp *.png result_HtoZZto4L/
cp *.csv result_HtoZZto4L/
cd result_HtoZZto4L




