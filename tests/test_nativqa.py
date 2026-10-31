"""Offline integration tests that also run against an installed distribution."""
import csv
import copy
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from nativqa import __version__
from nativqa.nativqa_framework import run_nativqa, scrape
from nativqa.qa_extraction import extract_ai_overview_answer, resolve_ai_overview_references
from nativqa.related_questions import expand_related_questions


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
                        "snippet": "Museum of Islamic Art — متحف الفن الإسلامي",
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
                    self.assertEqual(rows[0]["answer"], "Museum of Islamic Art — متحف الفن الإسلامي")
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

    def test_real_serpapi_client_supports_expansion_and_overview_timeouts(self):
        responses = [
            {"search_parameters": {"q": "seed", "engine": "google"}, "related_questions": [
                {"question": "Initial?", "snippet": "Initial answer", "link": "https://example.org/initial",
                 "next_page_token": "expand"},
            ]},
            {"related_questions": [{"question": "Expanded?", "type": "ai_overview", "page_token": "overview"}]},
            {"ai_overview": overview_answer("Expanded answer")},
        ]
        summary, failures = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {"API_KEY": "offline-test-key"}), \
             patch("serpapi.serp_api_client.requests.get",
                   side_effect=[Mock(text=json.dumps(response)) for response in responses]) as http:
            scrape("google", [["travel", "seed"]], "Doha, Qatar", "qa", summary, failures,
                   qa_extraction="hybrid", ai_overview_max_requests=1, related_questions_max_requests=1)
        self.assertEqual(failures.getvalue(), "")
        self.assertEqual([call.args[1]["engine"] for call in http.call_args_list],
                         ["google", "google_related_questions", "google_ai_overview"])
        self.assertTrue(all(call.kwargs["timeout"] == 30 for call in http.call_args_list))
        response = json.loads(summary.getvalue())
        self.assertEqual(response["related_questions"][1]["nativqa_ai_overview"]["status"], "fetched")


def overview_answer(text="The museum is in Doha."):
    return {
        "text_blocks": [{"type": "paragraph", "snippet": text, "reference_indexes": [7]}],
        "references": [{"index": 7, "title": "Museum", "link": "https://example.org/museum"}],
    }


class TestAIOverviewExtraction(unittest.TestCase):
    def test_nested_text_and_nonsequential_citation_indexes(self):
        overview = overview_answer("متحف في الدوحة")
        overview["references"].append({"index": 99, "link": "https://example.org/uncited"})
        overview["text_blocks"].extend([
            {"type": "list", "reference_indexes": [7], "list": [
                {"title": "Visit", "snippet": "Bring a ticket.", "list": [{"snippet": "Book ahead."}]},
                {"text_blocks": [{"type": "paragraph", "snippet": "Open daily."}]},
            ]},
            {"type": "expandable", "title": "Hours", "text_blocks": [
                {"type": "table", "table": [["Day", "Hours"], ["Monday", "10–18"]],
                 "reference_indexes": [7]},
            ]},
        ])
        result = extract_ai_overview_answer(overview)
        self.assertEqual(result["answer"], "متحف في الدوحة\nVisit: Bring a ticket.\nBook ahead.\nOpen daily.\nHours\nDay | Hours\nMonday | 10–18")
        self.assertEqual(result["source_urls"], ["https://example.org/museum"])
        self.assertEqual(result["answer_blocks"][2]["reference_indexes"], [7])
        self.assertEqual(result["citation_scope"], "block")
        self.assertEqual(result["verification_status"], "unverified")
        self.assertFalse(result["gold_eligible"])

    def test_reference_snippets_are_not_treated_as_answers(self):
        overview = {"references": [{"link": "https://example.org", "snippet": "Reference text only"}]}
        self.assertIsNone(extract_ai_overview_answer(overview))

    def test_null_optional_fields_and_malformed_references(self):
        self.assertIsNone(extract_ai_overview_answer(None))
        overview = overview_answer()
        overview['references'] = None
        self.assertIsNone(extract_ai_overview_answer(overview))
        overview['references'] = [None, {'link': None}, {'link': 'https://example.org/valid'}]
        overview['text_blocks'][0]['reference_indexes'] = None
        result = extract_ai_overview_answer(overview)
        self.assertEqual(result['source_urls'], ['https://example.org/valid'])

    def test_google_html_metadata_is_removed_without_losing_citations(self):
        for comment in ('<!--metadata-->', r'\x3c!--metadata-->', r'\u003c!--metadata--\u003e',
                        '&lt;!--metadata--&gt;'):
            with self.subTest(comment=comment):
                overview = overview_answer('A & B.')
                overview['text_blocks'] += [
                    {'type': 'paragraph', 'snippet': 'A &amp; B.' + comment,
                     'reference_indexes': [9]},
                    {'type': 'list', 'list': [{'snippet': 'Visit' + comment + ' today.'}]},
                    {'type': 'paragraph', 'snippet': comment},
                ]
                overview['references'].append({'index': 9, 'link': 'https://example.org/second'})
                result = extract_ai_overview_answer(overview)
                self.assertEqual(result['answer'], 'A & B.\nVisit today.')
                self.assertEqual(result['answer_blocks'][0]['reference_indexes'], [7,9])
                self.assertEqual(len(result['source_urls']), 2)
                self.assertIn(comment, overview['text_blocks'][1]['snippet'])

    def test_malformed_reference_indexes_and_urls_do_not_abort_extraction(self):
        overview = overview_answer()
        overview['references'] = [
            {'index': [], 'link': 'https://example.org/bad-index'},
            {'index': 7, 'link': 'https://['},
            {'index': 7, 'link': 'https://example.org/valid'},
        ]
        self.assertEqual(extract_ai_overview_answer(overview)['source_urls'], ['https://example.org/valid'])

    def test_malformed_overview_tokens_do_not_block_valid_questions(self):
        response = {'search_parameters': {'q': 'seed'}, 'related_questions': [
            None, {'type': 'ai_overview'},
            {'type': 'ai_overview', 'question': 'Bad?', 'page_token': [], 'serpapi_link': 'https://['},
            {'type': 'ai_overview', 'question': 'Good?', 'page_token': 'valid'},
        ]}
        factory = Mock()
        factory.return_value.get_dict.return_value = {'ai_overview': overview_answer()}
        self.assertEqual(resolve_ai_overview_references(response, factory, 'offline-key'), 1)
        self.assertEqual(response['related_questions'][2]['nativqa_ai_overview']['status'], 'missing_token')
        self.assertEqual(response['related_questions'][3]['nativqa_ai_overview']['status'], 'fetched')

    def test_uncited_or_invalid_sources_do_not_produce_pairs(self):
        for references in ([], [{"index": 7, "link": "javascript:alert(1)"}],
                           [{"index": 9, "link": "https://example.org/wrong-index"}]):
            with self.subTest(references=references):
                overview = overview_answer()
                overview["references"] = references
                self.assertIsNone(extract_ai_overview_answer(overview))

    def test_overview_level_references_are_marked_without_claim_mapping(self):
        overview = overview_answer()
        overview["text_blocks"][0].pop("reference_indexes")
        result = extract_ai_overview_answer(overview)
        self.assertEqual(result["citation_scope"], "overview")
        self.assertEqual(result["answer_blocks"][0]["reference_indexes"], [])

    def test_resolver_limits_requests_reuses_tokens_and_preserves_raw_response(self):
        response = {"search_parameters": {"q": "seed"}, "related_questions": [
            {"type": "ai_overview", "question": "First?", "page_token": "same-token"},
            {"type": "ai_overview", "question": "Again?", "page_token": "same-token"},
            {"type": "ai_overview", "question": "Limited?", "page_token": "other-token"},
        ]}
        factory = Mock()
        factory.return_value.get_dict.return_value = {
            "search_metadata": {"id": "overview-id"}, "ai_overview": overview_answer(),
        }
        count = resolve_ai_overview_references(response, factory, "offline-test-key", max_requests=1)
        self.assertEqual(count, 1)
        factory.assert_called_once_with({"engine": "google_ai_overview", "page_token": "same-token",
                                         "api_key": "offline-test-key"})
        self.assertEqual(factory.return_value.timeout, 30)
        questions = response["related_questions"]
        self.assertEqual(questions[0]["page_token"], "same-token")
        self.assertEqual(questions[0]["nativqa_ai_overview"]["response"]["search_metadata"]["id"], "overview-id")
        self.assertEqual(questions[1]["nativqa_ai_overview"]["status"], "fetched")
        self.assertEqual(questions[2]["nativqa_ai_overview"]["status"], "request_limit")

    def test_resolver_uses_only_overview_tokens_from_links(self):
        response = {"search_parameters": {"q": "seed"}, "related_questions": [
            {"type": "ai_overview", "question": "Good?", "serpapi_ai_overview_link":
             "https://serpapi.com/search.json?engine=google_ai_overview&page_token=good"},
            {"type": "ai_overview", "question": "Wrong engine?", "serpapi_link":
             "https://serpapi.com/search.json?engine=google_related_questions&page_token=wrong"},
            {"type": "ai_overview", "question": "Wrong host?", "serpapi_ai_overview_link":
             "https://example.org/search.json?engine=google_ai_overview&page_token=wrong"},
        ]}
        factory = Mock()
        factory.return_value.get_dict.return_value = {"ai_overview": overview_answer()}
        resolve_ai_overview_references(response, factory, "offline-test-key")
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(factory.call_args.args[0]["page_token"], "good")

    def test_embedded_overview_needs_no_additional_request(self):
        response = {"search_parameters": {"q": "seed"}, "ai_overview": overview_answer()}
        factory = Mock()
        self.assertEqual(resolve_ai_overview_references(response, factory, "key", max_requests=0), 0)
        factory.assert_not_called()
        self.assertEqual(response["ai_overview"]["nativqa_ai_overview"]["status"], "embedded")

    def test_failed_overviews_do_not_block_other_answers_or_leak_keys(self):
        for failure in (RuntimeError("request?api_key=secret-key"), {"error": "expired secret-key"}):
            with self.subTest(failure=failure):
                response = {"search_parameters": {"q": "seed"}, "related_questions": [
                    {"type": "ai_overview", "question": "Failed?", "page_token": "bad"},
                    {"type": "ai_overview", "question": "Good?", "page_token": "good"},
                ]}
                factory = Mock()
                factory.return_value.get_dict.side_effect = [failure, {"ai_overview": overview_answer()}]
                resolve_ai_overview_references(response, factory, "secret-key")
                self.assertEqual(response["related_questions"][0]["nativqa_ai_overview"]["status"], "error")
                self.assertEqual(response["related_questions"][1]["nativqa_ai_overview"]["status"], "fetched")
                self.assertNotIn("secret-key", json.dumps(response))


class TestQAExtractionPipeline(unittest.TestCase):
    def run_pipeline(self, root, response, mode="legacy", engine="google", extra_responses=(), max_requests=5,
                     n_iter=1, related_requests=0, search_type="text"):
        seed = root / "seed.tsv"
        seed.write_text("topic\tquery\ntravel\tseed query\n", encoding="utf-8")
        env = root / "api.env"
        env.write_text('API_KEY="offline-test-key"\n', encoding="utf-8")
        with patch.dict(os.environ, {"API_KEY": "offline-test-key"}), patch(
            "nativqa.nativqa_framework.GoogleSearch"
        ) as search:
            search.return_value.get_dict.side_effect = copy.deepcopy([response, *extra_responses])
            run_nativqa(engine=engine, search_type=search_type, input_file=str(seed), gl="qa",
                        location="Doha, Qatar", multiple_country=None, result_dir=str(root / "output"),
                        env=str(env), n_iter=n_iter, qa_extraction=mode,
                        ai_overview_max_requests=max_requests,
                        related_questions_max_requests=related_requests)
        result = root / "output/text/seed"
        with (result / "dataset/seed.tsv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        return result, rows, search

    def mixed_response(self):
        return {"search_parameters": {"q": "seed query"}, "related_questions": [
            {"question": "Classic question?", "snippet": "Classic answer", "link": "https://example.org/classic"},
            {"question": "Overview question?", "type": "ai_overview", "page_token": "fresh-token"},
        ]}

    def test_legacy_path_remains_available_for_all_text_engines(self):
        for engine in ("google", "bing", "yahoo"):
            with self.subTest(engine=engine), TemporaryDirectory() as temporary:
                result, rows, search = self.run_pipeline(Path(temporary), self.mixed_response(), engine=engine)
                self.assertEqual(search.call_count, 1)
                self.assertNotIn("no_cache", search.call_args.args[0])
                self.assertEqual([(row["question"], row["answer"]) for row in rows],
                                 [("Classic question?", "Classic answer")])
                self.assertFalse((result / "dataset/seed.ai_overview_candidates.jsonl").exists())

    def test_bing_and_yahoo_questions_and_answers_use_their_own_block(self):
        response = {"search_parameters": {"q": "seed query"}, "questions_and_answers": [
            {"question": "Direct question?", "answer": "Direct answer", "link": "https://example.org/direct"},
        ]}
        for engine in ("bing", "yahoo"):
            with self.subTest(engine=engine), TemporaryDirectory() as temporary:
                _, rows, search = self.run_pipeline(Path(temporary), response, engine=engine)
                search.assert_called_once()
                self.assertEqual(rows[0]["answer"], "Direct answer")
                self.assertEqual(rows[0]["question_type"], "questions_and_answers")

    def test_hybrid_and_ai_only_modes_export_provenance_and_resume_without_api_calls(self):
        for mode in ("hybrid", "ai_overview"):
            with self.subTest(mode=mode), TemporaryDirectory() as temporary:
                root = Path(temporary)
                result, rows, search = self.run_pipeline(root, self.mixed_response(), mode=mode,
                    extra_responses=[{"search_metadata": {"id": "ai-id"}, "ai_overview": overview_answer()}])
                self.assertEqual(search.call_count, 2)
                self.assertTrue(search.call_args_list[0].args[0]["no_cache"])
                self.assertEqual(search.call_args_list[1].args[0]["engine"], "google_ai_overview")
                expected_types = {"related_questions", "ai_overview"} if mode == "hybrid" else {"ai_overview"}
                self.assertEqual({row["question_type"] for row in rows}, expected_types)
                candidate_path = result / "dataset/seed.ai_overview_candidates.jsonl"
                candidates = [json.loads(line) for line in candidate_path.read_text().splitlines()]
                self.assertEqual(len(candidates), 1)
                candidate = candidates[0]
                self.assertEqual(candidate["question"], "Overview question?")
                self.assertEqual(candidate["answer"], "The museum is in Doha.")
                self.assertEqual(candidate["ai_overview_search_metadata"]["id"], "ai-id")
                self.assertEqual(candidate["verification_status"], "unverified")
                self.assertFalse(candidate["gold_eligible"])
                self.assertTrue(candidate["collected_at_utc"])
                _, resumed_rows, resumed_search = self.run_pipeline(root, self.mixed_response(), mode=mode)
                resumed_search.assert_not_called()
                self.assertEqual(rows, resumed_rows)

    def test_seed_and_related_overviews_are_paired_with_the_correct_questions(self):
        response = {"search_parameters": {"q": "seed query"}, "ai_overview": overview_answer("Seed answer"),
                    "related_questions": [{"question": "Related question?", "type": "ai_overview",
                                           **overview_answer("Related answer")}]}
        with TemporaryDirectory() as temporary:
            _, rows, search = self.run_pipeline(Path(temporary), response, mode="ai_overview", max_requests=0)
            search.assert_called_once()
            self.assertEqual({row["question"]: row["answer"] for row in rows},
                             {"seed query": "Seed answer", "Related question?": "Related answer"})

    def test_missing_overview_answer_keeps_classic_answer_and_raw_error(self):
        with TemporaryDirectory() as temporary:
            result, rows, _ = self.run_pipeline(Path(temporary), self.mixed_response(), mode="hybrid",
                                               extra_responses=[{"error": "Token expired"}])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["answer"], "Classic answer")
            raw = json.loads((result / "iteration_1/output/original_response.json").read_text())
            self.assertEqual(raw[0]["related_questions"][1]["nativqa_ai_overview"]["error"], "Token expired")

    def test_deduplicated_candidates_match_tsv_rows_across_iterations(self):
        response = {"search_parameters": {"q": "seed query"}, "related_questions": [
            {"type": "ai_overview", "question": "Shared?", **overview_answer("Shared answer")},
        ]}
        second = copy.deepcopy(response)
        second["search_parameters"]["q"] = "Shared?"
        with TemporaryDirectory() as temporary:
            result, rows, search = self.run_pipeline(Path(temporary), response, mode="ai_overview",
                                                     extra_responses=[second], n_iter=2)
            self.assertEqual(search.call_count, 2)
            self.assertEqual(len(rows), 1)
            candidates = [json.loads(line) for line in
                          (result / "dataset/seed.ai_overview_candidates.jsonl").read_text().splitlines()]
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["data_id"], rows[0]["data_id"])

    def test_each_seed_resolves_tokens_before_the_next_search(self):
        calls = []
        def make_search(params, **kwargs):
            calls.append(dict(params))
            if params["engine"] == "google":
                response = {"search_parameters": {"q": params["q"]}, "related_questions": [
                    {"type": "ai_overview", "question": "Related?", "page_token": "fresh"},
                ]}
            else:
                response = {"ai_overview": overview_answer()}
            return Mock(get_dict=Mock(return_value=response))
        summary, failures = io.StringIO(), io.StringIO()
        with patch("nativqa.nativqa_framework.GoogleSearch", side_effect=make_search):
            scrape("google", [["travel", "first"], ["travel", "second"]], "Doha, Qatar", "qa",
                   summary, failures, qa_extraction="hybrid")
        self.assertEqual([call["engine"] for call in calls],
                         ["google", "google_ai_overview", "google", "google_ai_overview"])
        self.assertEqual(failures.getvalue(), "")

    def test_changing_cached_extraction_mode_requires_new_output_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_pipeline(root, self.mixed_response())
            with self.assertRaisesRegex(ValueError, "new --output_dir"):
                self.run_pipeline(root, self.mixed_response(), mode="hybrid")

    def test_cache_rejects_changed_seed_contents(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_pipeline(root, self.mixed_response())
            (root / 'output/text/seed/iteration_1/seed.tsv').write_text('topic\tquery\ntravel\tother seed\n')
            with self.assertRaisesRegex(ValueError, 'new --output_dir'):
                self.run_pipeline(root, self.mixed_response())

    def test_cache_rejects_different_search_engine(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_pipeline(root, self.mixed_response())
            with self.assertRaisesRegex(ValueError, 'new --output_dir'):
                self.run_pipeline(root, self.mixed_response(), engine='bing')

    def test_cache_accepts_equivalent_crlf_and_lf_seed_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_pipeline(root, self.mixed_response())
            cached_seed = root / 'output/text/seed/iteration_1/seed.tsv'
            cached_seed.write_bytes(b'topic\tquery\r\ntravel\tseed query\r\n')
            _, rows, search = self.run_pipeline(root, self.mixed_response())
            search.assert_not_called()
            self.assertEqual(rows[0]['answer'], 'Classic answer')

    def test_non_google_ai_modes_are_rejected_before_any_api_call(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "use legacy"):
                self.run_pipeline(Path(temporary), self.mixed_response(), mode="hybrid", engine="bing")

    def test_expansion_exports_classic_answers_and_resumes_without_api_calls(self):
        initial = {"search_parameters": {"q": "seed query"}, "related_questions": [
            {"question": "Initial?", "snippet": "Initial answer", "link": "https://example.org/initial",
             "next_page_token": "expand-initial"},
        ]}
        expanded = {"search_metadata": {"id": "expansion-id"}, "related_questions": [
            {"question": "Expanded?", "list": ["One", "Two"], "link": "https://example.org/expanded"},
        ]}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, rows, search = self.run_pipeline(root, initial, extra_responses=[expanded], related_requests=2)
            self.assertEqual(search.call_count, 2)
            self.assertEqual(search.call_args_list[1].args[0]["engine"], "google_related_questions")
            self.assertEqual(search.call_args_list[1].args[0]["next_page_token"], "expand-initial")
            self.assertTrue(search.call_args_list[0].args[0]["no_cache"])
            self.assertEqual({row["question"]: row["answer"] for row in rows},
                             {"Initial?": "Initial answer", "Expanded?": "One\nTwo"})
            self.assertTrue(all(row["input_query"] == "seed query" for row in rows))
            raw = json.loads((result / "iteration_1/output/original_response.json").read_text())[0]
            self.assertEqual(raw["nativqa_related_questions"]["requests_made"], 1)
            self.assertEqual(raw["related_questions"][1]["nativqa_related_question"]["parent_question"], "Initial?")
            self.assertNotIn("data_id", raw["nativqa_related_questions"]["requests"][0]["response"]["related_questions"][0])
            _, resumed_rows, resumed_search = self.run_pipeline(root, initial, related_requests=2)
            resumed_search.assert_not_called()
            self.assertEqual(rows, resumed_rows)
            with self.assertRaisesRegex(ValueError, "new --output_dir"):
                self.run_pipeline(root, initial, related_requests=3)

    def test_expanded_overviews_share_one_seed_budget_and_resolve_between_batches(self):
        initial = {"search_parameters": {"q": "seed query"}, "related_questions": [
            {"question": "Initial AI?", "type": "ai_overview", "page_token": "ai-1",
             "next_page_token": "expand-1"},
        ]}
        expanded = {"related_questions": [
            {"question": "Shared AI?", "type": "ai_overview", "page_token": "ai-1"},
            {"question": "New AI?", "type": "ai_overview", "page_token": "ai-2",
             "next_page_token": "expand-2"},
            {"question": "Classic?", "snippet": "Classic answer", "link": "https://example.org/classic"},
        ]}
        final = {"related_questions": [
            {"question": "Limited AI?", "type": "ai_overview", "page_token": "ai-3"},
            {"question": "Embedded AI?", "type": "ai_overview", **overview_answer("Embedded answer")},
        ]}
        for mode in ("hybrid", "ai_overview"):
            with self.subTest(mode=mode), TemporaryDirectory() as temporary:
                result, rows, search = self.run_pipeline(Path(temporary), initial, mode=mode,
                    related_requests=2, max_requests=2, extra_responses=[
                        {"ai_overview": overview_answer("Initial answer")}, expanded,
                        {"ai_overview": overview_answer("New answer")}, final,
                    ])
                self.assertEqual([call.args[0]["engine"] for call in search.call_args_list],
                                 ["google", "google_ai_overview", "google_related_questions",
                                  "google_ai_overview", "google_related_questions"])
                answers = {row["question"]: row["answer"] for row in rows}
                self.assertEqual(answers["Shared AI?"], "Initial answer")
                self.assertEqual(answers["New AI?"], "New answer")
                self.assertEqual(answers["Embedded AI?"], "Embedded answer")
                self.assertNotIn("Limited AI?", answers)
                self.assertEqual("Classic?" in answers, mode == "hybrid")
                candidates = [json.loads(line) for line in
                    (result / "dataset/seed.ai_overview_candidates.jsonl").read_text().splitlines()]
                candidate = next(row for row in candidates if row["question"] == "New AI?")
                self.assertEqual(candidate["related_question_expansion"]["parent_question"], "Initial AI?")

    def test_expansion_failure_preserves_initial_qa(self):
        response = self.mixed_response()
        response["related_questions"][0]["next_page_token"] = "expired"
        with TemporaryDirectory() as temporary:
            result, rows, _ = self.run_pipeline(Path(temporary), response, related_requests=1,
                                                extra_responses=[{"error": "Expired token"}])
            self.assertEqual(rows[0]["answer"], "Classic answer")
            self.assertEqual((result / "iteration_1/output/failed.jsonl").read_text(), "")
            raw = json.loads((result / "iteration_1/output/original_response.json").read_text())[0]
            self.assertEqual(raw["nativqa_related_questions"]["requests"][0]["status"], "error")

    def test_expansion_rejects_unsupported_engines_and_search_types_before_requests(self):
        for engine, search_type in (("bing", "text"), ("yahoo", "text"),
                                    ("google", "image"), ("google", "video")):
            with self.subTest(engine=engine, search_type=search_type), TemporaryDirectory() as temporary:
                root = Path(temporary)
                with patch("nativqa.nativqa_framework.GoogleSearch") as search, \
                     self.assertRaisesRegex(ValueError, "requires Google text search"):
                    self.run_pipeline(root, self.mixed_response(), engine=engine,
                                      search_type=search_type, related_requests=1)
                search.assert_not_called()
                self.assertFalse((root / "output").exists())
        with self.assertRaisesRegex(ValueError, "requires Google text search"):
            scrape("bing", [], "Doha, Qatar", "qa", io.StringIO(), io.StringIO(),
                   related_questions_max_requests=1)

    def test_expansion_cannot_reuse_old_unexpanded_cache(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, _, _ = self.run_pipeline(root, self.mixed_response())
            config_path = result / "qa_extraction_config.json"
            config = json.loads(config_path.read_text())
            config.pop("related_questions_max_requests")
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "new --output_dir"):
                self.run_pipeline(root, self.mixed_response(), related_requests=1)
            _, _, search = self.run_pipeline(root, self.mixed_response())
            search.assert_not_called()

    def test_each_seed_finishes_expansion_before_the_next_seed(self):
        calls = []
        def make_search(params, **kwargs):
            calls.append(dict(params))
            response = ({"search_parameters": {"q": params["q"]}, "related_questions": [
                {"question": "Related?", "next_page_token": "shared-across-seeds"},
            ]} if params["engine"] == "google" else {"related_questions": []})
            return Mock(get_dict=Mock(return_value=response))
        with patch("nativqa.nativqa_framework.GoogleSearch", side_effect=make_search):
            summary, failures = io.StringIO(), io.StringIO()
            scrape("google", [["travel", "first"], ["travel", "second"]], "Doha, Qatar", "qa",
                   summary, failures, related_questions_max_requests=1)
        self.assertEqual([call["engine"] for call in calls],
                         ["google", "google_related_questions", "google", "google_related_questions"])
        self.assertEqual(failures.getvalue(), "")


class TestRelatedQuestionsExpansion(unittest.TestCase):
    def test_disabled_and_absent_tokens_make_no_requests(self):
        response = {"related_questions": [{"question": "Question?", "next_page_token": "token"}]}
        original = copy.deepcopy(response)
        search = Mock()
        self.assertEqual(expand_related_questions(response, search, "key", max_requests=0), 0)
        self.assertEqual(response, original)
        for value in (None, [], [None, {}, {"question": ""}], [{"question": "No token?"}]):
            with self.subTest(value=value):
                self.assertEqual(expand_related_questions({"related_questions": value}, search, "key", 3), 0)
        search.assert_not_called()
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            expand_related_questions(response, search, "key", -1)

    def test_breadth_first_budget_deduplicates_tokens_cycles_and_answers(self):
        initial = {"question": "Initial?", "snippet": "Initial answer", "link": "https://example.org/1",
                   "next_page_token": "first"}
        response = {"related_questions": [initial, {"question": "Other?", "next_page_token": "second"}]}
        new = {"question": "New?", "snippet": "New answer", "link": "https://example.org/2",
               "next_page_token": "third"}
        search = Mock()
        search.return_value.get_dict.side_effect = [
            {"related_questions": [copy.deepcopy(initial), new]},
            {"related_questions": [copy.deepcopy(new),
                dict(new, snippet="Alternative answer", next_page_token="fourth")]},
        ]
        self.assertEqual(expand_related_questions(response, search, "key", 2), 2)
        self.assertEqual([call.args[0]["next_page_token"] for call in search.call_args_list], ["first", "second"])
        self.assertEqual([entry.get("snippet") for entry in response["related_questions"]],
                         ["Initial answer", None, "New answer", "Alternative answer"])
        state = response["nativqa_related_questions"]
        self.assertEqual(state["added_questions"], 2)
        self.assertEqual(state["pending_tokens"], 2)
        self.assertTrue(state["request_limit_reached"])
        self.assertEqual(state["requests"][0]["response"]["related_questions"][1], new)

    def test_link_tokens_are_engine_specific_and_use_fixed_client_parameters(self):
        links = [
            "https://serpapi.com/search.json?engine=google_related_questions&next_page_token=valid%2Btoken",
            "https://attacker.example/search?engine=google_related_questions&next_page_token=wrong",
            "https://serpapi.com/search.json?engine=google_ai_overview&page_token=wrong",
            "http://serpapi.com/search?engine=google_related_questions&next_page_token=wrong",
            "https://[bad-host",
        ]
        response = {"related_questions": [{"question": f"Question {index}?", "serpapi_link": link}
                                          for index, link in enumerate(links)]}
        search = Mock()
        search.return_value.get_dict.return_value = {"related_questions": []}
        expand_related_questions(response, search, "key", 5)
        search.assert_called_once_with({'engine': 'google_related_questions',
                                       'next_page_token': 'valid+token', 'api_key': 'key'})
        self.assertEqual(search.return_value.timeout, 30)

    def test_failures_empty_and_malformed_batches_preserve_other_branches_and_redact_key(self):
        response = {"related_questions": [{"question": f"Question {i}?", "next_page_token": str(i)}
                                          for i in range(5)]}
        search = Mock()
        search.return_value.get_dict.side_effect = [
            {"error": "Expired?api_key=secret-key"}, RuntimeError("Network?api_key=secret-key"),
            {"related_questions": None}, {"related_questions": [None, {}, {"question": 123}]},
            {"search_metadata": {"id": "success"}, "api_key": "secret-key", "related_questions": [
                {"question": "Valid?", "snippet": "Valid answer", "link": "https://example.org/answer"},
            ]},
        ]
        expand_related_questions(response, search, "secret-key", 5)
        state = response["nativqa_related_questions"]
        self.assertEqual([request["status"] for request in state["requests"]],
                         ["error", "error", "empty", "empty", "fetched"])
        self.assertEqual(response["related_questions"][-1]["snippet"], "Valid answer")
        self.assertNotIn("secret-key", json.dumps(response))

    def test_recursive_tokens_are_followed_with_parent_provenance(self):
        response = {"related_questions": [{"question": "Root?", "next_page_token": "root"}]}
        search = Mock()
        search.return_value.get_dict.side_effect = [
            {"related_questions": [{"question": "Child?", "next_page_token": "child"}]},
            {"related_questions": [{"question": "Leaf?", "snippet": "Leaf answer",
                                     "link": "https://example.org/leaf", "next_page_token": "root"}]},
        ]
        self.assertEqual(expand_related_questions(response, search, "key", 5), 2)
        provenance = response["related_questions"][-1]["nativqa_related_question"]
        self.assertEqual(provenance["parent_question"], "Child?")
        self.assertEqual(provenance["depth"], 2)
        self.assertFalse(response["nativqa_related_questions"]["request_limit_reached"])


if __name__ == "__main__":
    unittest.main()
