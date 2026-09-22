import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "generate_visually_grounded_qa.py"
SPEC = importlib.util.spec_from_file_location("generate_visually_grounded_qa", SCRIPT_PATH)
assert SPEC and SPEC.loader
qa_generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa_generator)


class TestGenerateVisuallyGroundedQA(unittest.TestCase):
    def valid_candidate(self):
        return {
            "question_style": "short",
            "question": "What color is the vehicle shown in the image?",
            "answer": "Red.",
            "explanation": "The visible body of the vehicle is red.",
            "semantic_focus": "Visual Attributes",
            "cognitive_focus": "visual_perception",
            "question_target": "attribute",
        }

    def valid_long_candidate(self):
        return {
            "question_style": "long",
            "question": (
                "How do the vehicle and surrounding road features shown in the image "
                "differ in color and position?"
            ),
            "answer": (
                "The red vehicle occupies the lower center of the image, while pale "
                "lane markings extend around it. Dark pavement fills most of the "
                "background, creating a clear contrast with both the vehicle and lines."
            ),
            "explanation": (
                "The vehicle, lane markings, pavement, colors, and relative positions "
                "are all directly visible."
            ),
            "semantic_focus": "Scene Interpretation and Context",
            "cognitive_focus": "visual_reasoning",
            "question_target": "comparison",
        }

    def test_normalize_candidate_accepts_image_only_pair(self):
        self.assertEqual(
            qa_generator.normalize_candidate(self.valid_candidate()), self.valid_candidate()
        )

    def test_normalize_candidate_rejects_question_without_image_reference(self):
        candidate = self.valid_candidate()
        candidate["question"] = "What color is the vehicle?"
        with self.assertRaisesRegex(ValueError, "explicitly refer"):
            qa_generator.normalize_candidate(candidate)

    def test_normalize_candidate_rejects_prior_knowledge_focus(self):
        candidate = self.valid_candidate()
        candidate["cognitive_focus"] = "knowledge-based"
        with self.assertRaisesRegex(ValueError, "visual_perception"):
            qa_generator.normalize_candidate(candidate)

    def test_normalize_candidate_rejects_yes_no_question(self):
        candidate = self.valid_candidate()
        candidate["question"] = "Is the vehicle shown in the image red?"
        with self.assertRaisesRegex(ValueError, "open-ended"):
            qa_generator.normalize_candidate(candidate)

    def test_normalize_candidate_maps_target_used_as_semantic_focus(self):
        candidate = self.valid_candidate()
        candidate["semantic_focus"] = "attribute"
        normalized = qa_generator.normalize_candidate(candidate)
        self.assertEqual(normalized["semantic_focus"], "Visual Attributes")

    def test_normalize_candidate_accepts_long_visual_reasoning_pair(self):
        normalized = qa_generator.normalize_candidate(self.valid_long_candidate())
        self.assertEqual(normalized["question_style"], "long")
        self.assertEqual(normalized["cognitive_focus"], "visual_reasoning")

    def test_normalize_generation_requires_short_and_long_pairs(self):
        generation = {
            "image_assessment": {
                "content_type": "natural_scene",
                "visual_interest": "high",
                "text_dominant": False,
                "decision": "generate",
                "reason": "The image contains several distinct visible elements.",
            },
            "qa_pairs": [self.valid_candidate(), self.valid_long_candidate()],
        }
        normalized = qa_generator.normalize_generation(generation)
        self.assertEqual([pair["question_style"] for pair in normalized["qa_pairs"]], ["short", "long"])

    def test_normalize_generation_forces_text_dominant_image_to_skip(self):
        generation = {
            "image_assessment": {
                "content_type": "meme",
                "visual_interest": "medium",
                "text_dominant": True,
                "decision": "generate",
                "reason": "Most of the image consists of a caption.",
            },
            "qa_pairs": [self.valid_candidate(), self.valid_long_candidate()],
        }
        normalized = qa_generator.normalize_generation(generation)
        self.assertEqual(normalized["image_assessment"]["decision"], "skip")
        self.assertEqual(normalized["qa_pairs"], [])

    def test_output_row_marks_policy_skip(self):
        row = qa_generator.output_row(
            {"data_id": "one", "image_path": "one.jpg"},
            Path("one.jpg"),
            "test-deployment",
            image_assessment={
                "content_type": "meme",
                "visual_interest": "low",
                "text_dominant": True,
                "decision": "skip",
                "reason": "The image is text-dominant.",
            },
        )
        self.assertEqual(row["processing_status"], "skipped")
        self.assertEqual(row["schema_version"], "visual_qa_v2_two_style")

    def test_apply_rewrite_uses_visible_rewrite_and_updates_labels(self):
        verification = {
            "grounding": "NOT_GROUNDED",
            "issue": "needs_external_knowledge",
            "rewrite_type": "object",
            "rewritten_question": "What object is the person holding in the image?",
            "rewritten_answer": "A cup.",
            "rationale": "A cup is clearly visible in the person's hand.",
            "reused_original_answer": False,
        }
        rewritten = qa_generator.apply_rewrite(self.valid_candidate(), verification)
        self.assertEqual(rewritten["answer"], "A cup.")
        self.assertEqual(rewritten["question_target"], "object")
        self.assertEqual(
            rewritten["semantic_focus"], "Objects, Animals, and Food Recognition"
        )

    def test_grounded_verification_must_have_empty_rewrite_fields(self):
        result = {
            "grounding": "GROUNDED",
            "issue": "ok",
            "rewrite_type": "none",
            "rewritten_question": "unexpected",
            "rewritten_answer": "",
            "rationale": "",
            "reused_original_answer": False,
        }
        with self.assertRaisesRegex(ValueError, "violates"):
            qa_generator.normalize_verification(result)

    def test_load_records_supports_json_and_jsonl(self):
        rows = [{"data_id": "one", "image_path": "one.jpg"}]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "records.json"
            jsonl_path = root / "records.jsonl"
            json_path.write_text(json.dumps(rows), encoding="utf-8")
            jsonl_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            self.assertEqual(qa_generator.load_records(json_path), rows)
            self.assertEqual(qa_generator.load_records(jsonl_path), rows)

    def test_load_records_supports_single_json_record(self):
        row = {"data_id": "one", "image_path": "one.jpg"}
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertEqual(qa_generator.load_records(path), [row])


if __name__ == "__main__":
    unittest.main()
