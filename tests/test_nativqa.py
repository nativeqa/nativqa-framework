"""Offline integration tests that also run against an installed distribution."""
import csv
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nativqa import __version__
from nativqa.nativqa_framework import run_nativqa


class TestNativQA(unittest.TestCase):
    def test_search_outputs_and_absolute_output_directory(self):
        for search_type, api_engine in (
            ("text", "google"), ("image", "google_images"), ("video", "google_videos")
        ):
            with self.subTest(search_type=search_type), TemporaryDirectory() as temporary:
                root = Path(temporary)
                seed = root / "seed.csv"
                seed.write_text("topic,query\ntravel,What can I visit in Doha?\n", encoding="utf-8")
                env = root / "api.env"
                env.write_text('API_KEY="offline-test-key"\n', encoding="utf-8")
                response = {
                    "search_parameters": {"q": "What can I visit in Doha?"},
                    "related_questions": [{
                        "question": "Which museum can I visit?",
                        "snippet": "Museum of Islamic Art",
                        "link": "https://example.org/museum",
                    }],
                    "images_results": [{"original": "https://example.org/museum.jpg"}],
                    "video_results": [{"link": "https://example.org/museum-video"}],
                }
                with patch.dict(os.environ, {"API_KEY": "offline-test-key"}), patch(
                    "nativqa.nativqa_framework.GoogleSearch"
                ) as search:
                    search.return_value.get_dict.return_value = response
                    run_nativqa(
                        engine="google", search_type=search_type, input_file=str(seed),
                        gl="qa", location="Doha, Qatar", multiple_country=None,
                        result_dir=str(root / "output"), env=str(env), n_iter=1,
                    )
                    search.assert_called_once()
                    self.assertEqual(search.call_args.args[0]["engine"], api_engine)
                    self.assertEqual(search.call_args.args[0]["q"], "What can I visit in Doha?")
                    search.return_value.get_dict.assert_called_once_with()
                result = root / "output" / search_type / "seed"
                self.assertEqual((result / "completed_queries.txt").read_text().strip(),
                                 "What can I visit in Doha?")
                self.assertEqual((result / "iteration_1/output/failed.jsonl").read_text(), "")
                if search_type == "text":
                    with (result / "dataset/seed.tsv").open(encoding="utf-8", newline="") as handle:
                        rows = list(csv.DictReader(handle, delimiter="\t"))
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["answer"], "Museum of Islamic Art")
                else:
                    rows = json.loads((result / "dataset/seed.json").read_text())
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["category"], "travel")
                    self.assertEqual(rows[0]["input_query"], "What can I visit in Doha?")
                    self.assertTrue(rows[0]["data_id"])

    def test_module_help_and_version(self):
        for flag in ("--help", "--version"):
            with self.subTest(flag=flag):
                result = subprocess.run(
                    [sys.executable, "-m", "nativqa", flag],
                    capture_output=True, text=True, check=True,
                )
                expected = __version__ if flag == "--version" else "--input_file"
                self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
