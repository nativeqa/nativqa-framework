#!/usr/bin/env bash
# Reproduce the Google legacy audit from any directory after installing the checkout.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON=${NATIVQA_PYTHON:-python3}
if [ -z "${NATIVQA_PYTHON:-}" ] && [ -x ../.venv-nativqa/bin/python ]; then
  PYTHON=../.venv-nativqa/bin/python
fi
"$PYTHON" -m pip install -e .
"$PYTHON" -m pip check
"$PYTHON" -m nativqa --version
"$PYTHON" -m nativqa --help
mkdir -p results/ten_sample_queries
"$PYTHON" -m unittest tests.test_nativqa -v > results/ten_sample_queries/core_tests.log 2>&1
if [ -f ../queries/en_synthetic_queries_filtered.tsv ]; then
  "$PYTHON" experiments/ten_sample_queries/prepare_samples.py
fi
"$PYTHON" experiments/ten_sample_queries/run_samples.py
# run_samples.py executes the following CLI with the supplied key and redacted logging:
# ../.venv-nativqa/bin/python -m nativqa \
#   --engine google --search_type text \
#   --input_file experiments/ten_sample_queries/seeds.tsv \
#   --country_code qa --location "Doha, Qatar" \
#   --env ../envs/api_key.txt \
#   --output_dir results/ten_sample_queries --n_iter 1
status=0
# Returns 1 if the API succeeds but the stock QA exporter produces no answers.
"$PYTHON" experiments/ten_sample_queries/summarize_results.py || status=$?
"$PYTHON" -m pip freeze > results/ten_sample_queries/requirements.freeze.txt
exit "$status"
