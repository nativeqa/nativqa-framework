#!/usr/bin/env python3
"""Generate image-dependent open-ended QA pairs with Azure OpenAI."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv
from tqdm import tqdm


SEMANTIC_LABELS = (
    "Location and Place Identification",
    "Scene Interpretation and Context",
    "Architectural Features and Functions",
    "Cultural Significance and Heritage",
    "Traditional Clothing and Attire",
    "Tourism and Cultural Activities",
    "Event and Activity Type",
    "Objects, Animals, and Food Recognition",
    "National Symbols and Identity",
    "Visual Attributes",
    "Recreational Activities and Facilities",
)
QUESTION_TARGETS = (
    "object",
    "attribute",
    "action",
    "count",
    "spatial_relationship",
    "comparison",
    "scene_composition",
)
SEMANTIC_TARGET_FALLBACKS = {
    "object": "Objects, Animals, and Food Recognition",
    "attribute": "Visual Attributes",
    "action": "Scene Interpretation and Context",
    "count": "Objects, Animals, and Food Recognition",
    "spatial_relationship": "Scene Interpretation and Context",
    "comparison": "Visual Attributes",
    "scene_composition": "Scene Interpretation and Context",
}
QUESTION_STYLES = ("short", "long")
CONTENT_TYPES = (
    "natural_scene",
    "visually_rich_graphic",
    "meme",
    "text_dominant",
    "screenshot",
    "advertisement",
    "other",
)
EXCLUDED_CONTENT_TYPES = {"meme", "text_dominant", "screenshot"}
ISSUES = (
    "ok",
    "needs_external_knowledge",
    "not_visible",
    "illegible_text",
    "ambiguous",
    "subjective",
    "incorrect_answer",
)
IMAGE_REFERENCE_RE = re.compile(
    r"\b(image|photo|photograph|picture|shown|pictured|visible|depicted)\b", re.IGNORECASE
)
YES_NO_OPENING_RE = re.compile(
    r"^(is|are|am|was|were|do|does|did|can|could|will|would|shall|should|"
    r"has|have|had|may|might|must)\b",
    re.IGNORECASE,
)


GENERATOR_SYSTEM_PROMPT = """
You create challenging, image-dependent, open-ended Visual Question Answering
data from exactly one attached image. First assess whether the image is visually
interesting enough. Then either skip it or generate exactly TWO complementary
QA pairs: one short and one long.

Image-interest policy:
- SKIP memes, text-dominant graphics, screenshots, quote cards, title cards,
  documents, and images whose main content is written text.
- Also skip images that are too simple, unclear, or visually sparse to support
  two distinct questions.
- An advertisement may be used only when its non-text visual scene is rich and
  the questions do not ask about its wording or branding.
- Do not reward an image merely because it contains readable text. Text is
  incidental evidence only and MUST NOT be the target of either question.
- Generate only when the non-text visual content supports two genuinely
  different, useful questions.

Absolute grounding policy:
- Both questions MUST require viewing this particular image. Each must explicitly
  refer to "the image", "the photo", something "shown", or something "visible".
- Use no outside facts, learned identities, geographic recognition, cultural or
  historical knowledge, typical-use knowledge, stereotypes, or hidden context.
- Do not infer names, locations, nationality, religion, relationships, intent,
  emotion, purpose, cause, time outside the depicted moment, or what happens next.
- Ignore all possible metadata, including the file name, source query, title,
  category, description, URL, and directory. None of it is evidence.
- Allowed targets are: object, attribute, action, count, spatial_relationship,
  comparison, and scene_composition. Never use text recognition/transcription.
- For count, count only clearly visible, separable instances and prefer counts
  no greater than 10. Never count vague units such as groups, clusters,
  corridors, areas, or "major" features.
- Do not ask yes/no, true/false, multiple-choice, identity, location-name, or
  trivia questions. Do not reveal the answer in the wording of the question.
- Each explanation must be under 100 words and cite only visible evidence.

Pair 1 -- short:
- A direct question of no more than 18 words about one clear visual fact.
- Prefer object, attribute, action, or count.
- Use concrete visible nouns and avoid subjective qualifiers such as major,
  prominent, important, impressive, likely, or typical.
- The answer must be one concise sentence of no more than 25 words.
- cognitive_focus must be "visual_perception".

Pair 2 -- long:
- A specific compositional question of at least 16 words. It must require
  combining at least TWO distinct visible elements, regions, relationships, or
  contrasts; do not merely ask for a generic image description.
- Prefer spatial_relationship, comparison, action, or scene_composition.
- The answer must be 25 to 100 words and synthesize multiple visible details in
  two to four coherent sentences.
- cognitive_focus must be "visual_reasoning". This means reasoning across
  visible evidence only, never outside knowledge.

The two pairs must have different questions, answers, targets, and visual
evidence. The long pair must not restate or expand the short pair.

Choose semantic_focus from this exact list:
Location and Place Identification; Scene Interpretation and Context;
Architectural Features and Functions; Cultural Significance and Heritage;
Traditional Clothing and Attire; Tourism and Cultural Activities;
Event and Activity Type; Objects, Animals, and Food Recognition;
National Symbols and Identity; Visual Attributes;
Recreational Activities and Facilities.

Return one JSON object with exactly this structure:
{
  "image_assessment": {
    "content_type": "natural_scene|visually_rich_graphic|meme|text_dominant|screenshot|advertisement|other",
    "visual_interest": "high|medium|low",
    "text_dominant": false,
    "decision": "generate|skip",
    "reason": "brief visible-content reason"
  },
  "qa_pairs": [
    {
      "question_style": "short",
      "question": "single-sentence open-ended question",
      "answer": "one concise sentence",
      "explanation": "visible-evidence justification under 100 words",
      "semantic_focus": "one allowed semantic label",
      "cognitive_focus": "visual_perception",
      "question_target": "object|attribute|action|count"
    },
    {
      "question_style": "long",
      "question": "single-sentence compositional open-ended question",
      "answer": "25-100 word synthesis of visible evidence",
      "explanation": "visible-evidence justification under 100 words",
      "semantic_focus": "one allowed semantic label",
      "cognitive_focus": "visual_reasoning",
      "question_target": "spatial_relationship|comparison|action|scene_composition"
    }
  ]
}
If decision is "skip", qa_pairs MUST be an empty list. If decision is
"generate", qa_pairs MUST contain exactly the short pair and long pair above.
Output valid JSON only, with no markdown or extra text.
""".strip()


GENERATOR_USER_PROMPT = """
Assess the attached image and either skip uninteresting/text-driven content or
create one short and one long image-dependent QA pair. Use image pixels as the
only evidence. No description or metadata has been provided, and no prior or
outside knowledge is allowed. Prefer questions about the non-text visual scene.
""".strip()


VERIFIER_SYSTEM_PROMPT = """
You are a strict VISUAL-GROUNDING verifier for VQA. Exactly one image is attached.

Judge whether the given Question AND OriginalAnswer are correct using ONLY
clearly visible evidence in the image plus the question text. Also require the
question to be image-dependent: it must explicitly refer to the image/photo or
to something shown, pictured, visible, or depicted. Do NOT answer the original
question.

No outside facts, learned identities, geographic recognition, cultural or
historical knowledge, typical-use knowledge, stereotypes, metadata, or hidden
context may be used. Mark NOT_GROUNDED if the question needs any such knowledge,
asks for an unobservable detail, targets text, is ambiguous or subjective, could
be answered without inspecting the image, OR if OriginalAnswer is inaccurate,
unsupported, incomplete, or adds details not clearly visible.

If the question is image-dependent and OriginalAnswer is fully supported by
clear visible evidence: mark GROUNDED.

Do not mark NOT_GROUNDED merely to make a stylistic improvement. A rewrite is
required only for a real grounding, ambiguity, style-contract, or answer error.

If NOT_GROUNDED, you MUST:
1) Rewrite the question into an image-dependent, visually answerable,
   open-ended question using exactly ONE controlled rewrite type:
   object | attribute | action | count | spatial_relationship | comparison |
   scene_composition
2) Provide an answer using ONLY visible evidence in the image.
3) Provide a brief rationale citing only visible evidence.
4) If OriginalAnswer already correctly answers the rewritten question, reuse it
   exactly and set reused_original_answer=true.

Rules for a rewritten question:
- one sentence, explicitly referring to the image/photo or something shown,
  pictured, visible, or depicted
- targets ONLY the chosen rewrite type
- open-ended, not yes/no, true/false, or multiple-choice
- preserves the original intent when possible but removes ungrounded specifics
- never asks the reader to identify, transcribe, or interpret written text
- answer is not disclosed by the question itself
- removes vague or subjective words such as major, prominent, important,
  impressive, likely, or typical
- fully fixes the identified issue so the rewritten pair would pass a fresh,
  independent grounding check without another rewrite
- preserves QuestionStyle: short uses at most 18 question words and at most 25
  answer words; long uses at least 16 question words and a 25-100 word answer
  combining at least two distinct visible details

Language rules:
- rewritten_question and rewritten_answer must be in the language specified in
  Language.
- All control tokens/enums must be English exactly.

OUTPUT FORMAT (valid single-line JSON, no markdown or extra text):
{"grounding":"<GROUNDED|NOT_GROUNDED>","issue":"<ok|needs_external_knowledge|not_visible|illegible_text|ambiguous|subjective|incorrect_answer>","rewrite_type":"<object|attribute|action|count|spatial_relationship|comparison|scene_composition|none>","rewritten_question":"<text_or_empty>","rewritten_answer":"<text_or_empty>","rationale":"<text_or_empty>","reused_original_answer":<true|false>}

Constraints:
- If grounding=="GROUNDED": issue="ok", rewrite_type="none",
  rewritten_question="", rewritten_answer="", rationale="", and
  reused_original_answer=false.
- If grounding=="NOT_GROUNDED": issue!="ok", rewrite_type!="none", and
  rewritten_question, rewritten_answer, and rationale must all be non-empty.
- reused_original_answer can be true ONLY when rewritten_answer is EXACTLY
  identical to OriginalAnswer.
- Do not add fields; escape internal double quotes; do not mention URLs, file
  names, source queries, descriptions, categories, or base64 data.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Skip text-driven images, generate short and long image-only QA pairs, "
            "and verify that neither can be answered without the image."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON or JSONL records")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL file")
    parser.add_argument("--limit", type=int, help="Maximum number of records to process")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--max-verification-rounds",
        type=int,
        default=3,
        help="Maximum verifier calls after generation; rewritten pairs are verified again",
    )
    parser.add_argument(
        "--only-suitable",
        action="store_true",
        help="Use only successful validator records whose llm_annotation is suitable",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard existing output instead of resuming successful records",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(row)
        return records

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if data.get("image_path") or data.get("image_filename"):
            data = [data]
        else:
            data = list(data.values())
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected a JSON array or object of records in {path}")
    return data


def record_id(record: dict[str, Any]) -> str:
    data_id = record.get("data_id")
    if data_id not in (None, ""):
        return str(data_id)
    image_value = record.get("image_filename") or record.get("image_path")
    if image_value:
        return Path(str(image_value)).stem
    raise ValueError("record has neither data_id nor image path")


def resolve_image(record: dict[str, Any], images_dir: Path) -> Path:
    image_value = record.get("image_filename") or record.get("image_path")
    if not isinstance(image_value, str) or not image_value.strip():
        raise ValueError("record has no non-empty image_filename or image_path")
    candidate = images_dir / Path(image_value).name
    if not candidate.is_file():
        raise FileNotFoundError(f"No image named {candidate.name!r} under {images_dir}")
    return candidate


def image_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image extension: {image_path.name}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def azure_settings(env_file: Path) -> tuple[str, dict[str, str], str]:
    if not env_file.is_file():
        raise FileNotFoundError(f"Environment file not found: {env_file}")
    load_dotenv(env_file, override=True)
    required = ("AZURE_API_URL", "AZURE_API_KEY", "AZURE_API_VERSION", "AZURE_ENGINE_NAME")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    base_url = os.environ["AZURE_API_URL"].rstrip("/")
    deployment = os.environ["AZURE_ENGINE_NAME"]
    api_version = os.environ["AZURE_API_VERSION"]
    if "chat/completions" in base_url:
        separator = "&" if "?" in base_url else "?"
        api_url = base_url if "api-version=" in base_url else f"{base_url}{separator}api-version={api_version}"
    elif "/openai/deployments/" in base_url:
        api_url = f"{base_url}/chat/completions?api-version={api_version}"
    else:
        api_url = (
            f"{base_url}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )
    headers = {"api-key": os.environ["AZURE_API_KEY"], "Content-Type": "application/json"}
    return api_url, headers, deployment


def _response_content(body: dict[str, Any]) -> tuple[str, str | None]:
    try:
        choice = body["choices"][0]
        return choice["message"]["content"], choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Azure response is missing choices[0].message.content") from exc


def request_json(
    api_url: str,
    headers: dict[str, str],
    system_prompt: str,
    user_prompt: str,
    image_url: str,
    timeout: float,
    max_retries: int,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                ],
            },
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 1800,
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            if not response.ok:
                detail = response.text[:500].replace("\n", " ")
                raise RuntimeError(f"Azure API returned HTTP {response.status_code}: {detail}")
            content, finish_reason = _response_content(response.json())
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("Model response was not a JSON object")
            return validator(parsed), finish_reason
        except (
            requests.RequestException,
            json.JSONDecodeError,
            ValueError,
            RuntimeError,
        ) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            retry_after = 0.0
            response_obj = getattr(exc, "response", None)
            if response_obj is not None:
                try:
                    retry_after = float(response_obj.headers.get("Retry-After", 0))
                except (TypeError, ValueError):
                    retry_after = 0.0
            time.sleep(max(retry_after, min(30.0, (2**attempt) + random.random())))
    raise RuntimeError(f"Request failed after {max_retries + 1} attempts: {last_error}")


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _word_count(text: str) -> int:
    return len(text.split())


def is_image_dependent_question(question: str) -> bool:
    return bool(IMAGE_REFERENCE_RE.search(question))


def is_open_ended_question(question: str) -> bool:
    stripped = question.strip()
    return (
        stripped.endswith("?")
        and stripped.count("?") == 1
        and "\n" not in stripped
        and not YES_NO_OPENING_RE.match(stripped)
        and "true or false" not in stripped.lower()
    )


def normalize_candidate(candidate: dict[str, Any]) -> dict[str, str]:
    expected = {
        "question_style",
        "question",
        "answer",
        "explanation",
        "semantic_focus",
        "cognitive_focus",
        "question_target",
    }
    if set(candidate) != expected:
        raise ValueError(f"Candidate fields must be exactly {sorted(expected)}")
    normalized = {key: _nonempty_string(candidate.get(key), key) for key in expected}
    style = normalized["question_style"]
    if style not in QUESTION_STYLES:
        raise ValueError(f"Invalid question_style: {style!r}")
    if normalized["semantic_focus"] in SEMANTIC_TARGET_FALLBACKS:
        normalized["semantic_focus"] = SEMANTIC_TARGET_FALLBACKS[
            normalized["semantic_focus"]
        ]
    if normalized["semantic_focus"] not in SEMANTIC_LABELS:
        raise ValueError(f"Invalid semantic_focus: {normalized['semantic_focus']!r}")
    if normalized["question_target"] not in QUESTION_TARGETS:
        raise ValueError(f"Invalid question_target: {normalized['question_target']!r}")
    if not is_image_dependent_question(normalized["question"]):
        raise ValueError("question must explicitly refer to the image or something shown")
    if not is_open_ended_question(normalized["question"]):
        raise ValueError("question must be a single open-ended question")
    if len(normalized["explanation"].split()) >= 100:
        raise ValueError("explanation must be shorter than 100 words")
    question_words = _word_count(normalized["question"])
    answer_words = _word_count(normalized["answer"])
    if style == "short":
        if normalized["cognitive_focus"] != "visual_perception":
            raise ValueError("short cognitive_focus must be 'visual_perception'")
        if normalized["question_target"] not in {"object", "attribute", "action", "count"}:
            raise ValueError("short question has an invalid target")
        if question_words > 18:
            raise ValueError("short question must have at most 18 words")
        if answer_words > 25:
            raise ValueError("short answer must have at most 25 words")
    else:
        if normalized["cognitive_focus"] != "visual_reasoning":
            raise ValueError("long cognitive_focus must be 'visual_reasoning'")
        if normalized["question_target"] not in {
            "spatial_relationship",
            "comparison",
            "action",
            "scene_composition",
        }:
            raise ValueError("long question has an invalid target")
        if question_words < 16:
            raise ValueError("long question must have at least 16 words")
        if not 25 <= answer_words <= 100:
            raise ValueError("long answer must have 25 to 100 words")
    return {key: normalized[key] for key in (
        "question_style",
        "question",
        "answer",
        "explanation",
        "semantic_focus",
        "cognitive_focus",
        "question_target",
    )}


def normalize_generation(result: dict[str, Any]) -> dict[str, Any]:
    if set(result) != {"image_assessment", "qa_pairs"}:
        raise ValueError("Generation fields must be exactly image_assessment and qa_pairs")
    assessment = result.get("image_assessment")
    if not isinstance(assessment, dict) or set(assessment) != {
        "content_type",
        "visual_interest",
        "text_dominant",
        "decision",
        "reason",
    }:
        raise ValueError("image_assessment has invalid fields")
    if assessment.get("content_type") not in CONTENT_TYPES:
        raise ValueError(f"Invalid content_type: {assessment.get('content_type')!r}")
    if assessment.get("visual_interest") not in {"high", "medium", "low"}:
        raise ValueError(f"Invalid visual_interest: {assessment.get('visual_interest')!r}")
    if not isinstance(assessment.get("text_dominant"), bool):
        raise ValueError("text_dominant must be boolean")
    if assessment.get("decision") not in {"generate", "skip"}:
        raise ValueError(f"Invalid decision: {assessment.get('decision')!r}")
    normalized_assessment = dict(assessment)
    normalized_assessment["reason"] = _nonempty_string(assessment.get("reason"), "reason")

    must_skip = (
        normalized_assessment["content_type"] in EXCLUDED_CONTENT_TYPES
        or normalized_assessment["text_dominant"]
        or normalized_assessment["visual_interest"] == "low"
    )
    if must_skip:
        normalized_assessment["decision"] = "skip"

    pairs = result.get("qa_pairs")
    if not isinstance(pairs, list):
        raise ValueError("qa_pairs must be a list")
    if normalized_assessment["decision"] == "skip":
        return {"image_assessment": normalized_assessment, "qa_pairs": []}
    if len(pairs) != 2:
        raise ValueError("generate decision requires exactly two QA pairs")

    normalized_pairs = [normalize_candidate(pair) for pair in pairs]
    by_style = {pair["question_style"]: pair for pair in normalized_pairs}
    if set(by_style) != set(QUESTION_STYLES):
        raise ValueError("qa_pairs must contain one short and one long pair")
    short_pair = by_style["short"]
    long_pair = by_style["long"]
    if short_pair["question"].casefold() == long_pair["question"].casefold():
        raise ValueError("short and long questions must differ")
    if short_pair["answer"].casefold() == long_pair["answer"].casefold():
        raise ValueError("short and long answers must differ")
    if short_pair["question_target"] == long_pair["question_target"]:
        raise ValueError("short and long targets must differ")
    return {
        "image_assessment": normalized_assessment,
        "qa_pairs": [short_pair, long_pair],
    }


def normalize_verification(
    result: dict[str, Any], question_style: str | None = None
) -> dict[str, Any]:
    expected = {
        "grounding",
        "issue",
        "rewrite_type",
        "rewritten_question",
        "rewritten_answer",
        "rationale",
        "reused_original_answer",
    }
    if set(result) != expected:
        raise ValueError(f"Verification fields must be exactly {sorted(expected)}")
    if result.get("grounding") not in {"GROUNDED", "NOT_GROUNDED"}:
        raise ValueError(f"Invalid grounding: {result.get('grounding')!r}")
    if result.get("issue") not in ISSUES:
        raise ValueError(f"Invalid issue: {result.get('issue')!r}")
    if result.get("rewrite_type") not in {*QUESTION_TARGETS, "none"}:
        raise ValueError(f"Invalid rewrite_type: {result.get('rewrite_type')!r}")
    if not isinstance(result.get("reused_original_answer"), bool):
        raise ValueError("reused_original_answer must be boolean")
    for field in ("rewritten_question", "rewritten_answer", "rationale"):
        if not isinstance(result.get(field), str):
            raise ValueError(f"{field} must be a string")

    if result["grounding"] == "GROUNDED":
        if not (
            result["issue"] == "ok"
            and result["rewrite_type"] == "none"
            and result["rewritten_question"] == ""
            and result["rewritten_answer"] == ""
            and result["rationale"] == ""
            and result["reused_original_answer"] is False
        ):
            raise ValueError("GROUNDED verification violates its field constraints")
    else:
        if result["issue"] == "ok" or result["rewrite_type"] == "none":
            raise ValueError("NOT_GROUNDED must include a non-ok issue and rewrite type")
        for field in ("rewritten_question", "rewritten_answer", "rationale"):
            if not result[field].strip():
                raise ValueError(f"NOT_GROUNDED requires non-empty {field}")
        if not is_image_dependent_question(result["rewritten_question"]):
            raise ValueError("rewritten_question must explicitly refer to the image")
        if not is_open_ended_question(result["rewritten_question"]):
            raise ValueError("rewritten_question must be open-ended")
        if question_style == "short":
            if _word_count(result["rewritten_question"]) > 18:
                raise ValueError("rewritten short question is too long")
            if _word_count(result["rewritten_answer"]) > 25:
                raise ValueError("rewritten short answer is too long")
            if result["rewrite_type"] not in {"object", "attribute", "action", "count"}:
                raise ValueError("rewritten short question has an invalid target")
        elif question_style == "long":
            if _word_count(result["rewritten_question"]) < 16:
                raise ValueError("rewritten long question is too short")
            if not 25 <= _word_count(result["rewritten_answer"]) <= 100:
                raise ValueError("rewritten long answer must have 25 to 100 words")
            if result["rewrite_type"] not in {
                "spatial_relationship",
                "comparison",
                "action",
                "scene_composition",
            }:
                raise ValueError("rewritten long question has an invalid target")
    return result


def verifier_user_prompt(question_style: str, question: str, answer: str) -> str:
    return (
        "Language: English\n"
        f"QuestionStyle: {question_style}\n"
        f"Question: {question}\n"
        f"OriginalAnswer: {answer}\n\n"
        "Respond ONLY with the required single-line JSON object."
    )


def semantic_focus_for_rewrite(rewrite_type: str, existing: str) -> str:
    return SEMANTIC_TARGET_FALLBACKS.get(rewrite_type, existing)


def apply_rewrite(candidate: dict[str, str], verification: dict[str, Any]) -> dict[str, str]:
    rewritten = dict(candidate)
    rewrite_type = verification["rewrite_type"]
    question_style = candidate["question_style"]
    rewritten.update(
        {
            "question": verification["rewritten_question"].strip(),
            "answer": verification["rewritten_answer"].strip(),
            "explanation": verification["rationale"].strip(),
            "semantic_focus": semantic_focus_for_rewrite(
                rewrite_type, candidate["semantic_focus"]
            ),
            "cognitive_focus": (
                "visual_perception" if question_style == "short" else "visual_reasoning"
            ),
            "question_target": rewrite_type,
        }
    )
    return normalize_candidate(rewritten)


def generate_and_verify(
    api_url: str,
    headers: dict[str, str],
    image_path: Path,
    timeout: float,
    max_retries: int,
    max_verification_rounds: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, Any]],
    list[list[dict[str, Any]]],
    list[str | None],
]:
    data_url = image_data_url(image_path)
    generation, generation_finish = request_json(
        api_url,
        headers,
        GENERATOR_SYSTEM_PROMPT,
        GENERATOR_USER_PROMPT,
        data_url,
        timeout,
        max_retries,
        normalize_generation,
    )

    finish_reasons: list[str | None] = [generation_finish]
    assessment = generation["image_assessment"]
    if assessment["decision"] == "skip":
        return assessment, [], [], [], finish_reasons

    verified_pairs: list[dict[str, str]] = []
    final_verifications: list[dict[str, Any]] = []
    verification_histories: list[list[dict[str, Any]]] = []
    for generated_candidate in generation["qa_pairs"]:
        candidate = generated_candidate
        history: list[dict[str, Any]] = []
        for _ in range(max_verification_rounds):
            style = candidate["question_style"]
            verification, finish_reason = request_json(
                api_url,
                headers,
                VERIFIER_SYSTEM_PROMPT,
                verifier_user_prompt(style, candidate["question"], candidate["answer"]),
                data_url,
                timeout,
                max_retries,
                lambda result, style=style: normalize_verification(result, style),
            )
            finish_reasons.append(finish_reason)
            history.append(verification)
            if verification["grounding"] == "GROUNDED":
                verified_pairs.append(normalize_candidate(candidate))
                final_verifications.append(verification)
                verification_histories.append(history)
                break
            candidate = apply_rewrite(candidate, verification)
        else:
            raise RuntimeError(
                f"The {candidate['question_style']} QA pair did not receive a final "
                f"GROUNDED verdict within {max_verification_rounds} rounds"
            )

    if verified_pairs[0]["question"].casefold() == verified_pairs[1]["question"].casefold():
        raise RuntimeError("Verified short and long questions are duplicates")
    if verified_pairs[0]["question_target"] == verified_pairs[1]["question_target"]:
        raise RuntimeError("Verified short and long question targets are duplicates")
    return (
        assessment,
        verified_pairs,
        final_verifications,
        verification_histories,
        finish_reasons,
    )


def load_successful_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    successful: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if (
            row.get("schema_version") == "visual_qa_v2_two_style"
            and row.get("processing_status") in {"success", "skipped"}
            and row.get("data_id")
        ):
            successful.add(str(row["data_id"]))
    return successful


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def output_row(
    record: dict[str, Any],
    image_path: Path | None,
    deployment: str,
    image_assessment: dict[str, Any] | None = None,
    qa_pairs: list[dict[str, str]] | None = None,
    final_verifications: list[dict[str, Any]] | None = None,
    verification_histories: list[list[dict[str, Any]]] | None = None,
    finish_reasons: list[str | None] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    rewrites = sum(
        item.get("grounding") == "NOT_GROUNDED"
        for history in (verification_histories or [])
        for item in history
    )
    if error is not None:
        status = "error"
    elif image_assessment and image_assessment.get("decision") == "skip":
        status = "skipped"
    elif qa_pairs and len(qa_pairs) == 2:
        status = "success"
    else:
        status = "error"
    return {
        "schema_version": "visual_qa_v2_two_style",
        "data_id": record_id(record),
        "image_path": record.get("image_path"),
        "image_filename": image_path.name if image_path else record.get("image_filename"),
        "processing_status": status,
        "image_assessment": image_assessment,
        "qa_pairs": qa_pairs or [],
        "grounding_verifications": final_verifications or [],
        "verification_histories": verification_histories or [],
        "rewrite_count": rewrites,
        "processing_error": error,
        "model_deployment": deployment,
        "finish_reasons": finish_reasons or [],
        "generation_policy": (
            "two_style_image_only_no_text_targets_no_metadata_or_outside_knowledge"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.max_verification_rounds < 1:
        raise ValueError("--max-verification-rounds must be >= 1")

    records = load_records(args.input)
    if args.only_suitable:
        records = [
            row
            for row in records
            if row.get("processing_status") == "success"
            and bool(row.get("llm_annotation", {}).get("suitable_for_visual_qa"))
        ]
    selected = records[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]

    images_dir = args.images_dir.resolve()
    output = args.output.resolve()
    if args.overwrite and output.exists():
        output.unlink()
    completed_ids = load_successful_ids(output)
    api_url, headers, deployment = azure_settings(args.env_file)

    success_count = 0
    error_count = 0
    skipped_existing_count = 0
    skipped_by_policy_count = 0
    for record in tqdm(selected, desc="Generating grounded QA", unit="image"):
        data_id = record_id(record)
        if data_id in completed_ids:
            skipped_existing_count += 1
            continue
        image_path: Path | None = None
        try:
            image_path = resolve_image(record, images_dir)
            assessment, qa_pairs, final_verifications, histories, finish_reasons = (
                generate_and_verify(
                api_url,
                headers,
                image_path,
                args.timeout,
                args.max_retries,
                args.max_verification_rounds,
                )
            )
            append_jsonl(
                output,
                output_row(
                    record,
                    image_path,
                    deployment,
                    image_assessment=assessment,
                    qa_pairs=qa_pairs,
                    final_verifications=final_verifications,
                    verification_histories=histories,
                    finish_reasons=finish_reasons,
                ),
            )
            if assessment["decision"] == "skip":
                skipped_by_policy_count += 1
            else:
                success_count += 1
        except Exception as exc:  # Keep batch failures auditable and continue.
            append_jsonl(
                output,
                output_row(record, image_path, deployment, error=str(exc)),
            )
            error_count += 1
            print(f"ERROR {data_id}: {exc}", file=sys.stderr)

    print(
        json.dumps(
            {
                "selected": len(selected),
                "generated": success_count,
                "skipped_by_image_policy": skipped_by_policy_count,
                "errors": error_count,
                "skipped_existing": skipped_existing_count,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
