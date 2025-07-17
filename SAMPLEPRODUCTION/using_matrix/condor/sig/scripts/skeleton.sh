#!/bin/bash
# generate_runs.sh

# JSON 파일들이 들어있는 디렉터리
JSON_DIR="/data6/Users/achihwan/LRSM_tb_channel/SAMPLEPRODUCTION/using_matrix/condor/sig/SIGjson"
TEMPLATE="./arun_template.py"



for json in "${JSON_DIR}"/*.json; do
  # JSON 파일명만 (확장자 제외)
  base=$(basename "$json" .json)
  # 출력할 script 이름
  out="run_${base}.py"
  echo "Generating ${out} from ${json}"
  # #### 를 base 로 치환
  sed "s|####|${base}.json|g" "${TEMPLATE}" > "${out}"
  chmod +x "${out}"
done

echo "Done. Generated $(ls run_*.py | wc -l) scripts."
