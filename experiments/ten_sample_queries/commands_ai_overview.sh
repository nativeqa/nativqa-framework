#!/usr/bin/env bash
# Run from any directory. Existing baseline results remain in results/ten_sample_queries.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON=${NATIVQA_PYTHON:-python3}
if [ -z "${NATIVQA_PYTHON:-}" ] && [ -x ../.venv-nativqa/bin/python ]; then
  PYTHON=../.venv-nativqa/bin/python
fi
mkdir -p results/qa_extract_updated
"$PYTHON" -m unittest tests.test_nativqa -v > results/qa_extract_updated/core_tests.log 2>&1
"$PYTHON" -m pip check
"$PYTHON" -m nativqa --help
if [ -f ../queries/en_synthetic_queries_filtered.tsv ]; then
  "$PYTHON" experiments/ten_sample_queries/prepare_samples.py
fi
"$PYTHON" experiments/ten_sample_queries/run_samples.py \
  --qa_extraction hybrid \
  --ai_overview_max_requests 5 \
  --output_dir results/qa_extract_updated
# The wrapper invokes this CLI and redacts credentials from logs/results:
# ../.venv-nativqa/bin/python -m nativqa \
#   --engine google --search_type text \
#   --input_file experiments/ten_sample_queries/seeds.tsv \
#   --country_code qa --location "Doha, Qatar" \
#   --env ../envs/api_key.txt --output_dir results/qa_extract_updated \
#   --n_iter 1 --qa_extraction hybrid --ai_overview_max_requests 5
"$PYTHON" experiments/ten_sample_queries/replay_updated_qa.py > results/qa_extract_updated/replay_latest_code.log 2>&1
"$PYTHON" experiments/ten_sample_queries/summarize_updated_qa.py
"$PYTHON" -m pip freeze > results/qa_extract_updated/requirements.freeze.txt
