"""Regenerate the updated run from saved responses, prohibiting API calls."""

import os
from pathlib import Path
from unittest.mock import patch

from nativqa.nativqa_framework import run_nativqa


ROOT = Path(__file__).resolve().parents[2]


def main():
    with patch.dict(os.environ, {'API_KEY': 'offline-replay-key'}), patch(
        "nativqa.nativqa_framework.GoogleSearch",
        side_effect=AssertionError("Unexpected API call during cached replay"),
    ) as search:
        run_nativqa(
            engine="google", search_type="text", input_file=str(ROOT / "experiments/ten_sample_queries/seeds.tsv"),
            gl="qa", location="Doha, Qatar", multiple_country=None,
            result_dir=str(ROOT / "results/qa_extract_updated"), env=str(ROOT / "envs/api_key.env"), n_iter=1,
            qa_extraction="hybrid", ai_overview_max_requests=5,
        )
        search.assert_not_called()
    print("Final code replay completed with zero API calls.")


if __name__ == '__main__':
    main()
