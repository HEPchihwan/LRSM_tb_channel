#!/bin/bash
# submit_selected.sh

LISTFILE="asubmit.txt"

if [ ! -f "$LISTFILE" ]; then
  echo "ERROR: $LISTFILE not found!"
  exit 1
fi

while IFS= read -r line; do
  # 빈 줄 또는 # 로 시작하는 줄은 건너뛴다
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

  # 실제 condor_submit
  echo "Submitting $line"
  condor_submit "$line"
done < "$LISTFILE"
