#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_ZZto4L.py
mkdir result_ZZto4L 
cp *.png result_ZZto4L/
cp *.csv result_ZZto4L/
cd result_ZZto4L




