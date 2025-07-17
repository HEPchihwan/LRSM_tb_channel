#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_ST_sch_top_Lep.py
mkdir result_ST_sch_top_Lep 
cp *.png result_ST_sch_top_Lep/
cd result_ST_sch_top_Lep




