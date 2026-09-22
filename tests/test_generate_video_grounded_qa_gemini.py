import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "generate_video_grounded_qa_gemini.py"
)
SPEC = importlib.util.spec_from_file_location("generate_video_grounded_qa_gemini", SCRIPT_PATH)
assert SPEC and SPEC.loader
qa_generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa_generator)


class TestGenerateVideoGroundedQA(unittest.TestCase):
    def short_candidate(self):
        return {
            "question_style": "short",
            "question": "What object does the person hold in the video?",
            "answer": "The person holds a red cup.",
            "question_target": "object",
            "grounding_modality": "visual",
            "evidence": [
                {
                    "start_seconds": 2.0,
                    "end_seconds": 4.5,
                    "description": "A red cup is visible in the person's hand.",
                }
            ],
            "reasoning": "The cited interval clearly shows the held object.",
        }

    def long_candidate(self):
        return {
            "question_style": "long",
            "question": (
                "How do the person's actions shown early and later in the video "
                "combine to complete the task?"
            ),
            "answer": (
                "Early in the video, the person places the cup beneath the dispenser "
                "and presses its control. Later, the person removes the filled cup and "
                "sets it on the nearby counter, completing the visible sequence."
            ),
            "question_target": "temporal_sequence",
            "grounding_modality": "visual",
            "evidence": [
                {
                    "start_seconds": 1.0,
                    "end_seconds": 5.0,
                    "description": "The cup is positioned beneath the dispenser.",
                },
                {
                    "start_seconds": 12.0,
                    "end_seconds": 16.0,
                    "description": "The filled cup is moved to the counter.",
                },
            ],
            "reasoning": "The two intervals show the beginning and completion of the sequence.",
        }

    def valid_generation(self):
        return {
            "video_summary": "A person fills a red cup and places it on a counter.",
            "qa_pairs": [self.short_candidate(), self.long_candidate()],
        }

    def test_normalize_generation_accepts_short_and_long_pairs(self):
        result = qa_generator.normalize_generation(self.valid_generation(), 20.0)
        self.assertEqual(
            [pair["question_style"] for pair in result["qa_pairs"]],
            ["short", "long"],
        )

    def test_rejects_yes_no_question(self):
        candidate = self.short_candidate()
        candidate["question"] = "Is the person holding a cup in the video?"
        with self.assertRaisesRegex(ValueError, "open-ended"):
            qa_generator.normalize_candidate(candidate, 20.0)

    def test_accepts_grounded_question_without_literal_video_reference(self):
        candidate = self.short_candidate()
        candidate["question"] = "What object does the person hold?"
        normalized = qa_generator.normalize_candidate(candidate, 20.0)
        self.assertEqual(normalized["question"], candidate["question"])

    def test_rejects_long_pair_with_one_evidence_interval(self):
        candidate = self.long_candidate()
        candidate["evidence"] = candidate["evidence"][:1]
        with self.assertRaisesRegex(ValueError, "at least 2"):
            qa_generator.normalize_candidate(candidate, 20.0)

    def test_long_pair_accepts_distinct_evidence_from_same_moment(self):
        candidate = self.long_candidate()
        candidate["question_target"] = "comparison"
        candidate["evidence"][1]["start_seconds"] = 1.0
        candidate["evidence"][1]["end_seconds"] = 5.0
        normalized = qa_generator.normalize_candidate(candidate, 20.0)
        self.assertEqual(len(normalized["evidence"]), 2)

    def test_rejects_duplicated_long_evidence(self):
        candidate = self.long_candidate()
        candidate["evidence"][1] = dict(candidate["evidence"][0])
        with self.assertRaisesRegex(ValueError, "distinct items"):
            qa_generator.normalize_candidate(candidate, 20.0)

    def test_rejects_evidence_outside_video(self):
        candidate = self.short_candidate()
        candidate["evidence"][0]["end_seconds"] = 25.0
        with self.assertRaisesRegex(ValueError, "outside"):
            qa_generator.normalize_candidate(candidate, 20.0)

    def test_expands_implausibly_short_evidence_interval(self):
        candidate = self.short_candidate()
        candidate["evidence"][0]["start_seconds"] = 0.0
        candidate["evidence"][0]["end_seconds"] = 0.02
        normalized = qa_generator.normalize_candidate(candidate, 20.0)
        self.assertEqual(normalized["evidence"][0]["end_seconds"], 0.25)

    def test_repairs_reversed_evidence_interval(self):
        candidate = self.short_candidate()
        candidate["evidence"][0]["start_seconds"] = 12.0
        candidate["evidence"][0]["end_seconds"] = 4.0
        normalized = qa_generator.normalize_candidate(candidate, 20.0)
        self.assertEqual(normalized["evidence"][0]["start_seconds"], 4.0)
        self.assertEqual(normalized["evidence"][0]["end_seconds"], 12.0)

    def test_temporal_question_requires_separated_evidence(self):
        candidate = self.long_candidate()
        candidate["evidence"] = [
            {
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "description": "One action is visible.",
            },
            {
                "start_seconds": 0.1,
                "end_seconds": 1.1,
                "description": "A second action is visible.",
            },
        ]
        with self.assertRaisesRegex(ValueError, "temporally distinct"):
            qa_generator.normalize_candidate(candidate, 20.0)

    def test_load_records_supports_json_and_jsonl(self):
        row = {"video_name": "one", "video_path": "videos/one.mp4"}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "records.json"
            jsonl_path = root / "records.jsonl"
            json_path.write_text(json.dumps([row]), encoding="utf-8")
            jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(qa_generator.load_records(json_path), [row])
            self.assertEqual(qa_generator.load_records(jsonl_path), [row])

    def test_resolve_video_rejects_path_escape(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root.parent / "outside-video.mp4"
            outside.write_bytes(b"video")
            try:
                with self.assertRaisesRegex(ValueError, "escapes"):
                    qa_generator.resolve_video(
                        {"video_path": "../outside-video.mp4"}, root
                    )
            finally:
                outside.unlink(missing_ok=True)

    def test_prepared_video_reuses_small_supported_file(self):
        with TemporaryDirectory() as temporary:
            video = Path(temporary) / "small.mp4"
            video.write_bytes(b"small-video")
            with qa_generator.prepared_video(video, 5.0, 100, 80) as prepared:
                path, mime_type, transcoded = prepared
                self.assertEqual(path, video)
                self.assertEqual(mime_type, "video/mp4")
                self.assertFalse(transcoded)

    def test_successful_ids_only_include_success_rows(self):
        rows = [
            {
                "schema_version": "video_qa_v1_two_style",
                "processing_status": "success",
                "data_id": "one",
            },
            {
                "schema_version": "video_qa_v1_two_style",
                "processing_status": "error",
                "data_id": "two",
            },
        ]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "output.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            self.assertEqual(qa_generator.load_successful_ids(path), {"one"})

    def test_response_schema_uses_exact_video_duration(self):
        schema = qa_generator.response_schema_for_duration(12.3456)
        properties = schema["properties"]["qa_pairs"]["items"]["properties"][
            "evidence"
        ]["items"]["properties"]
        self.assertEqual(properties["start_seconds"]["maximum"], 12.346)
        self.assertEqual(properties["end_seconds"]["maximum"], 12.346)


if __name__ == "__main__":
    unittest.main()
