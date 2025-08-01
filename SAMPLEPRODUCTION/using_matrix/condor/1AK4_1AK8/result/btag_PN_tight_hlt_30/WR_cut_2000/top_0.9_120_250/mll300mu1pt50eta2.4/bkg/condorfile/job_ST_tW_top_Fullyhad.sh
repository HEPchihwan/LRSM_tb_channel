#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_ST_tW_top_Fullyhad.py
mkdir result_ST_tW_top_Fullyhad 
cp *.png result_ST_tW_top_Fullyhad/
cp *.csv result_ST_tW_top_Fullyhad/
cd result_ST_tW_top_Fullyhad




