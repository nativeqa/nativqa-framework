#!/usr/bin/env python3
"""Annotate crawled images for visual-QA suitability with Azure OpenAI."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm


IMAGE_CATEGORIES = (
    "Photograph",
    "Illustration",
    "Advertisement",
    "Screenshot/UI Capture",
    "Meme/Text Overlay",
    "Chart/Infographic",
    "Other",
)

SYSTEM_PROMPT = """
You are a meticulous image analyst selecting images for a visually grounded
question-answering (Visual QA) dataset. Judge query relevance and visual
grounding from the supplied image. For location relevance, compare the supplied
TARGET_LOCATION with both visible image evidence and the supplied source
metadata. Do not assume that a location mentioned only in the query is true.

Perform these tasks:
1. Describe the visible image concisely and objectively, and transcribe readable
   text. Never invent unreadable text.
2. Classify it as exactly one of: Photograph, Illustration, Advertisement,
   Screenshot/UI Capture, Meme/Text Overlay, Chart/Infographic, Other.
3. Judge query relevance:
   - highly_relevant: directly depicts the specific subject/entity/event asked about.
   - relevant: clearly depicts a central concept needed by the query.
   - weakly_relevant: only generic, tangential, decorative, or ambiguous support.
   - not_relevant: unrelated or depicts a conflicting subject.
4. Judge visual grounding. visually_grounded is true only when visible content
   would materially help a person answer or understand the query. A generic stock
   image, logo, unrelated thumbnail, or text-only repetition of the query is not
   sufficient.
5. Judge location relevance against TARGET_LOCATION. Use the image together
   with relevant metadata fields such as title, source, link, original, and
   thumbnail URL.

   Evidence policy:
   - Strong evidence includes readable place names, unmistakable landmarks or
     maps, location-specific organizations/events, and credible page/image URLs
     or titles that explicitly identify TARGET_LOCATION.
   - Corroboration across independent fields (for example, visible text plus the
     source title or original URL) increases confidence.
   - The input query, a SerpAPI related-content URL that merely repeats the
     query, the dataset directory in image_path, a hashed filename, or a generic
     website/domain is not independent location evidence. These weak fields
     cannot establish confirmed on their own.
   - If credible evidence explicitly identifies another location, treat that as
     a contradiction even when TARGET_LOCATION appears in the query.

   Labels:
   - confirmed: strong visual evidence or credible, explicit metadata ties the
     depicted subject/image to TARGET_LOCATION, with no stronger contradiction.
   - consistent_but_unverified: the image depicts the specific local subject and
     is compatible with TARGET_LOCATION, but neither the image nor independent
     metadata verifies the location.
   - uncertain: the image is generic or the available visual/metadata evidence
     is insufficient, ambiguous, or materially conflicting.
   - contradicted: credible visual or metadata evidence points to a different
     location than TARGET_LOCATION.
   - not_applicable: no location was requested.

   In location_evidence, name the exact evidence and its source, such as visible
   text, landmark, title, link, original URL, thumbnail URL, or source. State
   explicitly when only weak/query-derived evidence is available.
6. suitable_for_visual_qa is true only if the image is clear/interpretable,
   query_relevance is highly_relevant or relevant, visually_grounded is true,
   location_relevance is confirmed, consistent_but_unverified, or
   not_applicable, and the content is safe/appropriate. An uncertain or
   contradicted location is not suitable for a location-grounded Visual QA item.

Return one JSON object with exactly these fields and value types:
{
  "description": "string",
  "extracted_text": "string",
  "image_category": "one allowed category",
  "clarity": "clear|usable|poor",
  "query_relevance": "highly_relevant|relevant|weakly_relevant|not_relevant",
  "query_relevance_score": 0,
  "query_relevance_reason": "string",
  "visually_grounded": false,
  "visual_grounding_reason": "string",
  "location_relevance": "confirmed|consistent_but_unverified|uncertain|contradicted|not_applicable",
  "location_evidence": "string",
  "safe_content": true,
  "suitable_for_visual_qa": false,
  "status": "suitable|not_suitable",
  "reason": "short overall justification",
  "confidence": 0.0
}

query_relevance_score must be 3 for highly_relevant, 2 for relevant, 1 for
weakly_relevant, and 0 for not_relevant. confidence must be between 0 and 1.
status must agree with suitable_for_visual_qa. Output JSON only.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate crawled images for query, visual-grounding, and location relevance."
    )
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL file")
    parser.add_argument(
        "--location",
        help=(
            "Fallback location when a record has no location key; "
            "defaults to the input JSON filename stem"
        ),
    )
    parser.add_argument("--limit", type=int, help="Maximum number of records to process")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard the existing output instead of resuming successful records",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return data


def load_successful_ids(path: Path) -> set[str]:
    successful: set[str] = set()
    if not path.exists():
        return successful
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if row.get("processing_status") == "success" and row.get("data_id"):
                successful.add(str(row["data_id"]))
    return successful


def resolve_image(record: dict[str, Any], images_dir: Path) -> Path:
    image_path = record.get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("record has no non-empty image_path")
    filename = Path(image_path).name
    candidate = images_dir / filename
    if not candidate.is_file():
        raise FileNotFoundError(f"No image named {filename!r} under {images_dir}")
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


def resolve_location(
    record: dict[str, Any], cli_location: str | None, filename_location: str
) -> tuple[str, str]:
    record_location = record.get("location")
    if isinstance(record_location, str) and record_location.strip():
        return record_location.strip(), "record.location"
    if cli_location and cli_location.strip():
        return cli_location.strip(), "--location fallback"
    return filename_location, "input filename fallback"


def location_metadata(record: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "title",
        "source",
        "link",
        "original",
        "thumbnail",
        "serpapi_related_content_link",
        "image_path",
    )
    return {field: record[field] for field in fields if record.get(field) not in (None, "")}


def user_prompt(record: dict[str, Any], location: str, location_source: str) -> str:
    query = str(record.get("input_query", "")).strip()
    category = str(record.get("category", "")).strip() or "unspecified"
    metadata = json.dumps(location_metadata(record), ensure_ascii=False, indent=2)
    return (
        "Analyze this image for Visual QA dataset inclusion.\n"
        f"Input query: {query}\n"
        f"TARGET_LOCATION: {location or 'none'}\n"
        f"Location value source: {location_source}\n"
        f"Broad topic/category: {category}\n\n"
        "Source metadata for location assessment:\n"
        f"{metadata}\n\n"
        "Base query relevance and visual-grounding judgments on the image itself. "
        "Use the metadata only as additional evidence for location relevance, "
        "following the evidence policy in the system instructions. The category "
        "and input query are context, not proof of image or location relevance."
    )


def request_annotation(
    api_url: str,
    headers: dict[str, str],
    record: dict[str, Any],
    image_path: Path,
    location: str,
    location_source: str,
    timeout: float,
    max_retries: int,
) -> tuple[dict[str, Any], str | None]:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt(record, location, location_source)},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path), "detail": "high"}},
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
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            annotation = json.loads(content)
            if not isinstance(annotation, dict):
                raise ValueError("Model response was not a JSON object")
            finish_reason = body["choices"][0].get("finish_reason")
            return normalize_annotation(annotation), finish_reason
        except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(30.0, (2**attempt) + random.random()))
    raise RuntimeError(f"Annotation failed after {max_retries + 1} attempts: {last_error}")


def normalize_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    relevance_scores = {
        "highly_relevant": 3,
        "relevant": 2,
        "weakly_relevant": 1,
        "not_relevant": 0,
    }
    relevance = annotation.get("query_relevance")
    if relevance not in relevance_scores:
        raise ValueError(f"Invalid query_relevance: {relevance!r}")
    annotation["query_relevance_score"] = relevance_scores[relevance]

    category = annotation.get("image_category")
    if category not in IMAGE_CATEGORIES:
        raise ValueError(f"Invalid image_category: {category!r}")
    if annotation.get("clarity") not in {"clear", "usable", "poor"}:
        raise ValueError(f"Invalid clarity: {annotation.get('clarity')!r}")
    if annotation.get("location_relevance") not in {
        "confirmed",
        "consistent_but_unverified",
        "uncertain",
        "contradicted",
        "not_applicable",
    }:
        raise ValueError(f"Invalid location_relevance: {annotation.get('location_relevance')!r}")

    for key in ("visually_grounded", "safe_content", "suitable_for_visual_qa"):
        if not isinstance(annotation.get(key), bool):
            raise ValueError(f"{key} must be a boolean")

    minimum_suitability_criteria = (
        annotation.get("clarity") in {"clear", "usable"}
        and relevance in {"highly_relevant", "relevant"}
        and annotation["visually_grounded"]
        and annotation.get("location_relevance")
        in {"confirmed", "consistent_but_unverified", "not_applicable"}
        and annotation["safe_content"]
    )
    if annotation["suitable_for_visual_qa"] and not minimum_suitability_criteria:
        annotation["suitable_for_visual_qa"] = False
        existing_reason = str(annotation.get("reason", "")).strip()
        annotation["reason"] = (
            "Not suitable because one or more minimum query, visual-grounding, "
            f"location, clarity, or safety criteria are unmet. {existing_reason}"
        ).strip()
    annotation["status"] = "suitable" if annotation["suitable_for_visual_qa"] else "not_suitable"
    try:
        annotation["confidence"] = max(0.0, min(1.0, float(annotation.get("confidence", 0.0))))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    for key in (
        "description",
        "extracted_text",
        "query_relevance_reason",
        "visual_grounding_reason",
        "location_evidence",
        "reason",
    ):
        if not isinstance(annotation.get(key), str):
            annotation[key] = "" if annotation.get(key) is None else str(annotation[key])
    return annotation


def output_row(
    record: dict[str, Any],
    image_path: Path | None,
    location: str,
    location_source: str,
    deployment: str,
    annotation: dict[str, Any] | None = None,
    error: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "data_id": str(record.get("data_id", Path(str(record.get("image_path", "unknown"))).stem)),
        "input_query": record.get("input_query"),
        "image_path": record.get("image_path"),
        "image_filename": image_path.name if image_path else Path(str(record.get("image_path", ""))).name,
        "location": location,
        "location_source": location_source,
        "category": record.get("category"),
        "source": record.get("source"),
        "source_title": record.get("title"),
        "source_link": record.get("link"),
        "processing_status": "success" if annotation is not None else "error",
        "llm_annotation": annotation,
        "processing_error": error,
        "model_deployment": deployment,
        "finish_reason": finish_reason,
        "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> int:
    args = parse_args()
    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError("--offset must be >= 0 and --limit must be >= 1")

    records = load_records(args.input_json)
    selected = records[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    filename_location = args.input_json.stem
    args.images_dir = args.images_dir.resolve()
    args.output = args.output.resolve()

    if args.overwrite and args.output.exists():
        args.output.unlink()
    completed_ids = load_successful_ids(args.output)
    api_url, headers, deployment = azure_settings(args.env_file)

    success_count = 0
    error_count = 0
    skipped_count = 0
    for record in tqdm(selected, desc="Annotating images", unit="image"):
        location, location_source = resolve_location(record, args.location, filename_location)
        data_id = str(record.get("data_id", Path(str(record.get("image_path", ""))).stem))
        if data_id in completed_ids:
            skipped_count += 1
            continue
        image_path: Path | None = None
        try:
            image_path = resolve_image(record, args.images_dir)
            annotation, finish_reason = request_annotation(
                api_url,
                headers,
                record,
                image_path,
                location,
                location_source,
                args.timeout,
                args.max_retries,
            )
            append_jsonl(
                args.output,
                output_row(
                    record,
                    image_path,
                    location,
                    location_source,
                    deployment,
                    annotation,
                    finish_reason=finish_reason,
                ),
            )
            success_count += 1
        except Exception as exc:  # Keep batch processing and make failures auditable.
            error_count += 1
            append_jsonl(
                args.output,
                output_row(
                    record,
                    image_path,
                    location,
                    location_source,
                    deployment,
                    error=str(exc),
                ),
            )
            print(f"ERROR {data_id}: {exc}", file=sys.stderr)

    print(
        json.dumps(
            {
                "selected": len(selected),
                "annotated": success_count,
                "errors": error_count,
                "skipped_existing": skipped_count,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
