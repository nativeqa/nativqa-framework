#!/usr/bin/env python3
"""Generate one short-form and one long-form open-ended QA per video.

The model receives only video-derived content and the generation instructions;
source questions, answers, file names, and other dataset metadata are not sent.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from google import genai
from google.genai import types
from google.oauth2 import service_account
from tqdm import tqdm


QUESTION_STYLES = {"short", "long"}
SHORT_TARGETS = {"object", "attribute", "action", "count", "spoken_content"}
LONG_TARGETS = {
    "temporal_sequence",
    "visible_cause_effect",
    "comparison",
    "scene_change",
    "multi_event_summary",
    "audio_visual_relation",
}
GROUNDING_MODALITIES = {"visual", "audio", "audio_visual"}
VIDEO_REFERENCE_RE = re.compile(
    r"\b(video|clip|footage|recording|shown|visible|heard|depicted)\b", re.IGNORECASE
)
YES_NO_OPENING_RE = re.compile(
    r"^\s*(is|are|am|was|were|do|does|did|can|could|will|would|shall|should|"
    r"has|have|had|may|might|must)\b",
    re.IGNORECASE,
)
SUPPORTED_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".mpg": "video/mpg",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
}


SYSTEM_PROMPT = """
You create rigorously video-grounded, open-ended question-answer data from
exactly one attached video. Analyze both the visual stream and audible content,
but use only evidence directly present in the video.

Generate exactly TWO complementary QA pairs in English:

1. Short-form pair
- question_style must be "short".
- Ask one direct question of at most 18 words.
- The answer must be one concise sentence of at most 25 words.
- Use one target from: object, attribute, action, count, spoken_content.
- Cite at least one precise evidence interval.

2. Long-form pair
- question_style must be "long".
- Ask one specific question of at least 16 words.
- The question must require combining evidence from at least two distinct
  moments, events, regions, or audio-visual signals in the video.
- The answer must contain 25 to 100 words in two to four coherent sentences.
- Use one target from: temporal_sequence, visible_cause_effect, comparison,
  scene_change, multi_event_summary, audio_visual_relation.
- Cite at least two distinct evidence intervals.

Rules for both pairs:
- Each question must explicitly refer to the video, clip, footage, recording,
  something shown or visible, or something heard.
- Questions must be open-ended, not yes/no, true/false, or multiple-choice.
- Do not reveal the answer in the question.
- Do not use the file name, metadata, source dataset, or an existing QA pair.
- Do not use outside facts, learned identities, geographic recognition,
  stereotypes, or speculation about hidden intent, emotion, or causality.
- Do not ask for illegible text. Spoken content may be used only when clearly
  audible. Visual details may be used only when clearly visible.
- Evidence descriptions must state what is directly visible or audible and
  must not introduce unsupported facts.
- Evidence times are seconds from the start of the video, with
  0 <= start_seconds <= end_seconds <= the supplied duration.
- Times are ordinary elapsed seconds, not normalized fractions of the video's
  length and not fractions of a minute. Use whole seconds or tenths of a second.
- Every evidence interval must span at least 0.25 seconds.
- The two pairs must have different questions, answers, targets, and evidence.
- The long pair must not merely restate or expand the short pair.

Return only the JSON object required by the response schema.
""".strip()


RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_summary": {
            "type": "string",
            "description": "A brief evidence-only description used for auditing.",
        },
        "qa_pairs": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "question_style": {"type": "string", "enum": ["short", "long"]},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "question_target": {
                        "type": "string",
                        "enum": sorted(SHORT_TARGETS | LONG_TARGETS),
                    },
                    "grounding_modality": {
                        "type": "string",
                        "enum": sorted(GROUNDING_MODALITIES),
                    },
                    "evidence": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_seconds": {"type": "number", "minimum": 0},
                                "end_seconds": {"type": "number", "minimum": 0},
                                "description": {"type": "string"},
                            },
                            "required": ["start_seconds", "end_seconds", "description"],
                            "additionalProperties": False,
                        },
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why the cited video evidence supports the answer.",
                    },
                },
                "required": [
                    "question_style",
                    "question",
                    "answer",
                    "question_target",
                    "grounding_modality",
                    "evidence",
                    "reasoning",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["video_summary", "qa_pairs"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input JSON or JSONL")
    parser.add_argument(
        "--videos-root",
        required=True,
        type=Path,
        help="Root joined with each record's video_path",
    )
    parser.add_argument(
        "--credentials",
        required=True,
        type=Path,
        help="Google service-account JSON used for Vertex AI",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL")
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--location", default="global")
    parser.add_argument("--language", default="English")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0, help="Request timeout seconds")
    parser.add_argument(
        "--max-inline-mb",
        type=float,
        default=14.0,
        help="Transcode inputs larger than this raw-byte size",
    )
    parser.add_argument(
        "--transcode-target-mb",
        type=float,
        default=11.5,
        help="Approximate size target for temporary MP4 files",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard existing output instead of resuming successful records",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop immediately on the first record error"
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records = []
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
        data = [data] if data.get("video_path") else list(data.values())
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected a JSON array or object of records in {path}")
    return data


def record_id(record: dict[str, Any]) -> str:
    for field in ("video_name", "data_id", "sample_id"):
        value = record.get(field)
        if value not in (None, ""):
            return str(value)
    video_path = record.get("video_path")
    if isinstance(video_path, str) and video_path.strip():
        return Path(video_path).stem
    raise ValueError("record has no video_name, data_id, sample_id, or video_path")


def resolve_video(record: dict[str, Any], videos_root: Path) -> Path:
    value = record.get("video_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("record has no non-empty video_path")
    root = videos_root.resolve()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"video_path escapes --videos-root: {value}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Video not found: {candidate}")
    return candidate


def probe_duration(video_path: Path) -> float:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is required but was not found on PATH")
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        duration = float(completed.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not read duration for {video_path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Invalid duration for {video_path}: {duration}")
    return duration


def _run_transcode(
    source: Path,
    destination: Path,
    duration: float,
    target_bytes: int,
) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for oversized or unsupported videos")
    total_bitrate = max(160_000, int((target_bytes * 8 * 0.94) / duration))
    audio_bitrate = 48_000
    video_bitrate = max(100_000, min(650_000, total_bitrate - audio_bitrate))
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "scale='min(640,iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            str(video_bitrate),
            "-maxrate",
            str(int(video_bitrate * 1.25)),
            "-bufsize",
            str(video_bitrate * 2),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            str(audio_bitrate),
            "-ac",
            "1",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )


@contextlib.contextmanager
def prepared_video(
    source: Path,
    duration: float,
    max_inline_bytes: int,
    target_bytes: int,
) -> Iterator[tuple[Path, str, bool]]:
    suffix = source.suffix.lower()
    mime_type = SUPPORTED_VIDEO_MIME_TYPES.get(suffix)
    if mime_type and source.stat().st_size <= max_inline_bytes:
        yield source, mime_type, False
        return

    with tempfile.TemporaryDirectory(prefix="nativqa-video-") as temporary:
        converted = Path(temporary) / "video.mp4"
        _run_transcode(source, converted, duration, target_bytes)
        if converted.stat().st_size > max_inline_bytes:
            converted.unlink()
            _run_transcode(source, converted, duration, int(target_bytes * 0.70))
        if converted.stat().st_size > max_inline_bytes:
            raise RuntimeError(
                f"Transcoded video remains too large: {converted.stat().st_size} bytes"
            )
        yield converted, "video/mp4", True


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _word_count(text: str) -> int:
    return len(text.split())


def is_open_ended_question(question: str) -> bool:
    stripped = question.strip()
    return (
        stripped.endswith("?")
        and stripped.count("?") == 1
        and "\n" not in stripped
        and not YES_NO_OPENING_RE.match(stripped)
        and "true or false" not in stripped.casefold()
        and "which of the following" not in stripped.casefold()
    )


def normalize_candidate(candidate: dict[str, Any], duration: float) -> dict[str, Any]:
    expected = {
        "question_style",
        "question",
        "answer",
        "question_target",
        "grounding_modality",
        "evidence",
        "reasoning",
    }
    if not isinstance(candidate, dict) or set(candidate) != expected:
        raise ValueError(f"QA fields must be exactly {sorted(expected)}")
    style = _nonempty_string(candidate.get("question_style"), "question_style")
    question = _nonempty_string(candidate.get("question"), "question")
    answer = _nonempty_string(candidate.get("answer"), "answer")
    target = _nonempty_string(candidate.get("question_target"), "question_target")
    modality = _nonempty_string(candidate.get("grounding_modality"), "grounding_modality")
    reasoning = _nonempty_string(candidate.get("reasoning"), "reasoning")
    if style not in QUESTION_STYLES:
        raise ValueError(f"Invalid question_style: {style!r}")
    if modality not in GROUNDING_MODALITIES:
        raise ValueError(f"Invalid grounding_modality: {modality!r}")
    if not is_open_ended_question(question):
        raise ValueError("question must be a single open-ended question")

    evidence = candidate.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("evidence must be a list")
    minimum_evidence = 1 if style == "short" else 2
    if len(evidence) < minimum_evidence:
        raise ValueError(f"{style} QA requires at least {minimum_evidence} evidence intervals")
    normalized_evidence = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict) or set(item) != {
            "start_seconds",
            "end_seconds",
            "description",
        }:
            raise ValueError(f"evidence[{index}] has invalid fields")
        try:
            start = float(item["start_seconds"])
            end = float(item["end_seconds"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"evidence[{index}] times must be numbers") from exc
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"evidence[{index}] times must be finite")
        if end < start:
            start, end = end, start
        if start < 0 or end < start or end > duration + 1.0:
            raise ValueError(
                f"evidence[{index}] interval {start}-{end} is outside 0-{duration:.3f}s"
            )
        if end - start < 0.25:
            start = min(start, max(0.0, duration - 0.25))
            end = min(duration, start + 0.25)
        normalized_evidence.append(
            {
                "start_seconds": round(start, 3),
                "end_seconds": round(min(end, duration), 3),
                "description": _nonempty_string(
                    item.get("description"), f"evidence[{index}].description"
                ),
            }
        )

    question_words = _word_count(question)
    answer_words = _word_count(answer)
    if style == "short":
        if target not in SHORT_TARGETS:
            raise ValueError(f"Invalid short question_target: {target!r}")
        if question_words > 18:
            raise ValueError("short question must have at most 18 words")
        if answer_words > 25:
            raise ValueError("short answer must have at most 25 words")
    else:
        if target not in LONG_TARGETS:
            raise ValueError(f"Invalid long question_target: {target!r}")
        if question_words < 16:
            raise ValueError("long question must have at least 16 words")
        if not 25 <= answer_words <= 100:
            raise ValueError("long answer must contain 25 to 100 words")
        distinct_evidence = {
            (
                round(item["start_seconds"], 1),
                round(item["end_seconds"], 1),
                item["description"].casefold(),
            )
            for item in normalized_evidence
        }
        if len(distinct_evidence) < 2:
            raise ValueError("long QA evidence must contain at least two distinct items")
        if target in {
            "temporal_sequence",
            "visible_cause_effect",
            "scene_change",
            "multi_event_summary",
        }:
            centers = sorted(
                (item["start_seconds"] + item["end_seconds"]) / 2
                for item in normalized_evidence
            )
            if centers[-1] - centers[0] < 0.25:
                raise ValueError(
                    f"{target} evidence must cover temporally distinct moments"
                )

    return {
        "question_style": style,
        "question": question,
        "answer": answer,
        "question_target": target,
        "grounding_modality": modality,
        "evidence": normalized_evidence,
        "reasoning": reasoning,
    }


def normalize_generation(result: dict[str, Any], duration: float) -> dict[str, Any]:
    if not isinstance(result, dict) or set(result) != {"video_summary", "qa_pairs"}:
        raise ValueError("Response fields must be exactly video_summary and qa_pairs")
    summary = _nonempty_string(result.get("video_summary"), "video_summary")
    pairs = result.get("qa_pairs")
    if not isinstance(pairs, list) or len(pairs) != 2:
        raise ValueError("qa_pairs must contain exactly two items")
    normalized_pairs = [normalize_candidate(pair, duration) for pair in pairs]
    by_style = {pair["question_style"]: pair for pair in normalized_pairs}
    if set(by_style) != QUESTION_STYLES:
        raise ValueError("qa_pairs must contain exactly one short and one long pair")
    short = by_style["short"]
    long = by_style["long"]
    if short["question"].casefold() == long["question"].casefold():
        raise ValueError("short and long questions must differ")
    if short["answer"].casefold() == long["answer"].casefold():
        raise ValueError("short and long answers must differ")
    if short["question_target"] == long["question_target"]:
        raise ValueError("short and long question targets must differ")
    return {"video_summary": summary, "qa_pairs": [short, long]}


def create_client(credentials_path: Path, location: str, timeout: float) -> genai.Client:
    if not credentials_path.is_file():
        raise FileNotFoundError(f"Credential file not found: {credentials_path}")
    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    project_id = credentials.project_id
    if not project_id:
        raise ValueError("Credential file has no project_id")
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=credentials,
        http_options=types.HttpOptions(
            api_version="v1",
            timeout=int(timeout * 1000),
        ),
    )


def response_schema_for_duration(duration: float) -> dict[str, Any]:
    schema = copy.deepcopy(RESPONSE_JSON_SCHEMA)
    evidence_properties = schema["properties"]["qa_pairs"]["items"]["properties"][
        "evidence"
    ]["items"]["properties"]
    maximum = round(duration, 3)
    evidence_properties["start_seconds"]["maximum"] = maximum
    evidence_properties["end_seconds"]["maximum"] = maximum
    return schema


def _usage_dict(response: types.GenerateContentResponse) -> dict[str, Any]:
    usage = response.usage_metadata
    return usage.model_dump(mode="json", exclude_none=True) if usage else {}


def generate_qa(
    client: genai.Client,
    model: str,
    prepared_path: Path,
    mime_type: str,
    duration: float,
    language: str,
    seed: int,
    max_retries: int,
    validator: Callable[[dict[str, Any], float], dict[str, Any]] = normalize_generation,
) -> tuple[dict[str, Any], dict[str, Any], str | None, str | None]:
    prompt = (
        f"Generate the two QA pairs in {language}. The attached video's exact "
        f"duration is {duration:.3f} seconds. Use the video as the only semantic "
        "evidence and obey the response schema."
    )
    video_bytes = prepared_path.read_bytes()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        seed=seed,
        max_output_tokens=4096,
        response_mime_type="application/json",
        response_json_schema=response_schema_for_duration(duration),
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    )
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=video_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=config,
            )
            if not response.text:
                raise RuntimeError("Gemini returned no text")
            parsed = json.loads(response.text)
            generation = validator(parsed, duration)
            candidate = response.candidates[0] if response.candidates else None
            finish_reason = None
            if candidate and candidate.finish_reason:
                finish_reason = getattr(candidate.finish_reason, "value", str(candidate.finish_reason))
            return generation, _usage_dict(response), finish_reason, response.model_version
        except (Exception,) as exc:  # SDK errors vary by transport and API version.
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(60.0, (2**attempt) + random.random()))
    raise RuntimeError(f"Gemini request failed after {max_retries + 1} attempts: {last_error}")


def load_successful_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    successful = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if (
            row.get("schema_version") == "video_qa_v1_two_style"
            and row.get("processing_status") == "success"
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
    video_path: Path | None,
    model: str,
    location: str,
    duration: float | None = None,
    generation: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    model_version: str | None = None,
    transcoded: bool = False,
    api_input_bytes: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    status = "success" if error is None and generation else "error"
    return {
        "schema_version": "video_qa_v1_two_style",
        "data_id": record_id(record),
        "source_record": record,
        "video_path": record.get("video_path"),
        "video_filename": video_path.name if video_path else None,
        "duration_seconds": duration,
        "processing_status": status,
        "video_summary": generation.get("video_summary") if generation else None,
        "qa_pairs": generation.get("qa_pairs", []) if generation else [],
        "processing_error": error,
        "model": model,
        "model_version": model_version,
        "vertex_location": location,
        "finish_reason": finish_reason,
        "usage_metadata": usage or {},
        "input_transcoded": transcoded,
        "api_input_bytes": api_input_bytes,
        "generation_policy": "two_style_video_only_no_metadata_or_outside_knowledge",
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
    if args.max_inline_mb <= 0 or args.transcode_target_mb <= 0:
        raise ValueError("video size limits must be positive")
    if args.transcode_target_mb >= args.max_inline_mb:
        raise ValueError("--transcode-target-mb must be smaller than --max-inline-mb")

    records = load_records(args.input)
    selected = records[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    output = args.output.resolve()
    if args.overwrite and output.exists():
        output.unlink()
    successful_ids = load_successful_ids(output)
    pending = [row for row in selected if record_id(row) not in successful_ids]
    if not pending:
        print(json.dumps({"status": "nothing_to_do", "output": str(output)}))
        return 0

    videos_root = args.videos_root.resolve()
    max_inline_bytes = int(args.max_inline_mb * 1024 * 1024)
    target_bytes = int(args.transcode_target_mb * 1024 * 1024)
    client = create_client(args.credentials.resolve(), args.location, args.timeout)
    success_count = 0
    error_count = 0
    try:
        for record in tqdm(pending, desc="Generating video QA", unit="video"):
            video_path: Path | None = None
            duration: float | None = None
            try:
                video_path = resolve_video(record, videos_root)
                duration = probe_duration(video_path)
                with prepared_video(
                    video_path,
                    duration,
                    max_inline_bytes,
                    target_bytes,
                ) as (api_video, mime_type, transcoded):
                    generation, usage, finish_reason, model_version = generate_qa(
                        client=client,
                        model=args.model,
                        prepared_path=api_video,
                        mime_type=mime_type,
                        duration=duration,
                        language=args.language,
                        seed=args.seed,
                        max_retries=args.max_retries,
                    )
                    append_jsonl(
                        output,
                        output_row(
                            record,
                            video_path,
                            args.model,
                            args.location,
                            duration=duration,
                            generation=generation,
                            usage=usage,
                            finish_reason=finish_reason,
                            model_version=model_version,
                            transcoded=transcoded,
                            api_input_bytes=api_video.stat().st_size,
                        ),
                    )
                success_count += 1
            except Exception as exc:
                error_count += 1
                append_jsonl(
                    output,
                    output_row(
                        record,
                        video_path,
                        args.model,
                        args.location,
                        duration=duration,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
                if args.fail_fast:
                    raise
    finally:
        client.close()

    print(
        json.dumps(
            {
                "selected": len(selected),
                "already_successful": len(selected) - len(pending),
                "processed_successfully": success_count,
                "errors": error_count,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
