#!/usr/bin/env python3
import os
import glob
import json

# 1) JSON 파일들이 있는 디렉터리
JSON_DIR = "/data6/Users/achihwan/LRSM_tb_channel/SAMPLEPRODUCTION/using_matrix/background/SIGjson"

# 2) MC base directory
#    여기에 모든 signal 디렉터리가 아래 구조처럼 들어 있다고 가정합니다:
#    /gv0/DATA/SKNano/Run3NanoAODv12/2022/MC/<sample_name>/.../0000/*.root
MC_BASE = "/data6/Users/achihwan/LRSM_tb_channel/SAMPLEPRODUCTION/samples"

# 3) 모든 JSON 파일에 대해 처리
for json_file in glob.glob(os.path.join(JSON_DIR, "*.json")):
    with open(json_file, 'r') as fin:
        info = json.load(fin)

    # 4) sample_name 결정: JSON 안에 "name" 필드가 있으면 그걸 쓰고,
    #    없으면 파일명(확장자 제외)으로 대체
    sample_name = info.get("name") or os.path.splitext(os.path.basename(json_file))[0]
    print(f"Processing {json_file}, sample_name = '{sample_name}'")

    # 5) glob 패턴으로 해당 디렉터리 찾기
    #    예: /gv0/.../MC/WRtoNMutoMuMuTB-HadTop_MWR-1000_MN-700_13p6TeV*/**/0000/*.root
    pattern = os.path.join(
        MC_BASE,
        f"{sample_name}*",
        "**",        # subdirs, e.g. timestamp / 0000
        "*.root"
    )
    # recursive=True 로 하위까지 검색
    root_files = sorted(glob.glob(pattern, recursive=True))
    if not root_files:
        print(f"  WARNING: no files found for pattern:\n    {pattern}")
    else:
        print(f"  Found {len(root_files)} files")

    # 6) info["path"] 덮어쓰기
    info["path"] = root_files

    # 7) JSON에 덮어써서 저장 (원본 덮어쓰기)
    with open(json_file, 'w') as fout:
        json.dump(info, fout, indent=2)
        fout.write("\n")

print("All done.")  
