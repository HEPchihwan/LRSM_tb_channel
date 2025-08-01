#!/bin/bash
tar -xzf hepenv.tar.gz
source ./bin/activate
echo "PYTHON: $(which python)"
python run_WRtoNMutoMuMuTB-HadTop_MWR-3000_MN-2100_13p6TeV.py
mkdir result_WRtoNMutoMuMuTB-HadTop_MWR-3000_MN-2100_13p6TeV 
cp *.png result_WRtoNMutoMuMuTB-HadTop_MWR-3000_MN-2100_13p6TeV/
cp *.csv result_WRtoNMutoMuMuTB-HadTop_MWR-3000_MN-2100_13p6TeV/
cd result_WRtoNMutoMuMuTB-HadTop_MWR-3000_MN-2100_13p6TeV




