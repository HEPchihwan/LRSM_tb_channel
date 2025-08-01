#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_ST_tW_antitop_Semilep.py
mkdir result_ST_tW_antitop_Semilep 
cp *.png result_ST_tW_antitop_Semilep/
cp *.csv result_ST_tW_antitop_Semilep/
cd result_ST_tW_antitop_Semilep




