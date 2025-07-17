#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_ST_tW_antitop_Lep.py
mkdir result_ST_tW_antitop_Lep 
cp *.png result_ST_tW_antitop_Lep/
cd result_ST_tW_antitop_Lep




