#!/bin/zsh
# Runs the stages that need R-LMD features, once 02_rlmd_features.py has finished.
#   cd analysis && caffeinate -i zsh scripts/run_rest.sh > results/log_run_rest.txt 2>&1
# Every stage caches its finished pieces, so rerunning this skips completed work.
set -e
cd "$(dirname "$0")/.."
PY=../synthesis/.venv/bin/python

while pgrep -f scripts/02_rlmd_features.py > /dev/null; do sleep 60; done
n=$(ls cache/rlmd/*.npz | wc -l)
[[ $n -eq 55 ]] || { echo "R-LMD cache has $n of 55 subjects; rerun 02_rlmd_features.py"; exit 1; }
echo "== R-LMD complete $(date)"

$PY scripts/05_models.py --models rf xgb --jobs 4 2>&1 | grep -v Warning | tee results/log_05_models_part2.txt
$PY scripts/03_eda.py 2>&1 | tee results/log_03_eda.txt
$PY scripts/07_evaluate.py > results/log_07_evaluate_interim.txt 2>&1 || true
echo "== candidates done $(date)"

for protocol in corrected leaky; do
  for task in bin shin3 cog3; do
    echo "== baseline $task $protocol $(date)"
    $PY scripts/04_baselines.py --task $task --protocol $protocol --jobs 8 2>&1 \
      | tee -a results/log_04_baselines.txt
  done
done

$PY scripts/07_evaluate.py 2>&1 | tee results/log_07_evaluate.txt
echo "== all done $(date)"
