#!/usr/bin/env bash
# Run from any directory. Uses the existing virtual environment and supplied key.
# Bing API responses are cached on disk; rerunning resumes this same ten-seed run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON=${NATIVQA_PYTHON:-python3}
if [ -z "${NATIVQA_PYTHON:-}" ] && [ -x ../.venv-nativqa/bin/python ]; then
  PYTHON=../.venv-nativqa/bin/python
fi

"$PYTHON" -m nativqa --version
"$PYTHON" -m nativqa --help
"$PYTHON" -m pip check
if [ -f ../queries/en_synthetic_queries_filtered.tsv ]; then
  "$PYTHON" experiments/ten_sample_queries/prepare_samples.py
fi
"$PYTHON" experiments/ten_sample_queries/run_samples.py \
  --engine bing \
  --output_dir results/ten_sample_queries_bing \
  --qa_extraction legacy

# The credential-redacting runner executes this framework CLI:
# ../.venv-nativqa/bin/python -m nativqa \
#   --engine bing --search_type text \
#   --input_file experiments/ten_sample_queries/seeds.tsv \
#   --country_code qa --location 'Doha, Qatar' \
#   --env ../envs/api_key.txt \
#   --output_dir results/ten_sample_queries_bing --n_iter 1 \
#   --qa_extraction legacy --ai_overview_max_requests 5
# AI Overview follow-ups are disabled for Bing by legacy extraction.

"$PYTHON" -m unittest tests.test_nativqa -v \
  > results/ten_sample_queries_bing/core_tests.log 2>&1
"$PYTHON" -m pip freeze > results/ten_sample_queries_bing/requirements.freeze.txt
"$PYTHON" experiments/ten_sample_queries_bing/summarize_results.py
cat results/ten_sample_queries_bing/qa_snippets.tsv
