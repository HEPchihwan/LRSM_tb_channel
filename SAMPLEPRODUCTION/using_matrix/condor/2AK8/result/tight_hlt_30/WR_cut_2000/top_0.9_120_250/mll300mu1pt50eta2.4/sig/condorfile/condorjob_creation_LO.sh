#!/bin/bash
# generate_jobs.sh

# 템플릿 파일 이름
TEMPLATE_JDS="template_LO.jds"

#!/bin/bash
# generate_submits.sh

# JSON 파일들이 들어있는 디렉터리
JSON_DIR="/data6/Users/achihwan/LRSM_tb_channel/SAMPLEPRODUCTION/using_matrix/condor/1AK4_1AK8/result/btag_PN_tight_hlt_30/WR_cut_2000/top_0.9_120_250/mll300mu1pt50eta2.4/sig/SIGjson"

# 출력할 submit 스크립트 폴더
OUT_DIR="/data6/Users/achihwan/LRSM_tb_channel/SAMPLEPRODUCTION/using_matrix/condor/2AK8/result/tight_hlt_30/WR_cut_2000/top_0.9_120_250/mll300mu1pt50eta2.4/sig/condorfile"
mkdir -p "${OUT_DIR}"

count=0
for json in "${JSON_DIR}"/*.json; do
  base=$(basename "$json" .json)
  subfile="${OUT_DIR}/job_${base}.jds"
  echo "Generating ${subfile}"
  # aaa 를 $base 로 치환
  sed "s/aaa/${base}/g" "${TEMPLATE_JDS}" > "${subfile}"
  ((count++))
done

echo "Done. Generated ${count} condor submit scripts in ${OUT_DIR}/"
