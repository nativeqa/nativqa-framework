#!/usr/bin/env python3
"""Summarize LLM or manual visual-QA annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report visual-QA annotation statistics")
    parser.add_argument("--annotations", required=True, type=Path, help="Validator JSONL output")
    parser.add_argument("--manual-annotations", type=Path, help="Optional manual-review JSONL")
    parser.add_argument("--output-json", type=Path, help="Also save the report as JSON")
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = list(data.values())
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list/object in {path}")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def latest_by_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        data_id = row.get("data_id")
        if data_id is not None:
            latest[str(data_id)] = row
    return latest


def count_values(annotations: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(a.get(key, "missing")) for a in annotations).items()))


def pct(count: int, denominator: int) -> float:
    return round(100.0 * count / denominator, 2) if denominator else 0.0


def build_report(
    llm_rows: list[dict[str, Any]], manual_rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    latest = latest_by_id(llm_rows)
    successful_rows = [row for row in latest.values() if row.get("processing_status") == "success"]
    annotations = [row["llm_annotation"] for row in successful_rows if isinstance(row.get("llm_annotation"), dict)]
    total = len(latest)
    successful = len(annotations)

    suitable = sum(bool(a.get("suitable_for_visual_qa")) for a in annotations)
    grounded = sum(bool(a.get("visually_grounded")) for a in annotations)
    query_relevant = sum(a.get("query_relevance") in {"highly_relevant", "relevant"} for a in annotations)
    highly_relevant = sum(a.get("query_relevance") == "highly_relevant" for a in annotations)
    location_aligned = sum(
        a.get("location_relevance") in {"confirmed", "consistent_but_unverified", "not_applicable"}
        for a in annotations
    )
    location_confirmed = sum(a.get("location_relevance") == "confirmed" for a in annotations)
    strict_eligible = sum(
        a.get("query_relevance") in {"highly_relevant", "relevant"}
        and bool(a.get("visually_grounded"))
        and a.get("location_relevance") in {"confirmed", "not_applicable"}
        and a.get("clarity") in {"clear", "usable"}
        and bool(a.get("safe_content"))
        for a in annotations
    )

    report: dict[str, Any] = {
        "records": {
            "unique_records": total,
            "successful": successful,
            "errors": total - len(successful_rows),
        },
        "key_metrics": {
            "llm_suitable_for_visual_qa": {"count": suitable, "percent": pct(suitable, successful)},
            "visually_grounded": {"count": grounded, "percent": pct(grounded, successful)},
            "query_relevant_or_highly_relevant": {
                "count": query_relevant,
                "percent": pct(query_relevant, successful),
            },
            "query_highly_relevant": {"count": highly_relevant, "percent": pct(highly_relevant, successful)},
            "location_aligned": {"count": location_aligned, "percent": pct(location_aligned, successful)},
            "location_visually_confirmed": {
                "count": location_confirmed,
                "percent": pct(location_confirmed, successful),
            },
            "strict_query_grounding_and_confirmed_location": {
                "count": strict_eligible,
                "percent": pct(strict_eligible, successful),
            },
        },
        "distributions": {
            "status": count_values(annotations, "status"),
            "query_relevance": count_values(annotations, "query_relevance"),
            "location_relevance": count_values(annotations, "location_relevance"),
            "image_category": count_values(annotations, "image_category"),
            "clarity": count_values(annotations, "clarity"),
        },
    }

    if manual_rows is not None:
        decisions = latest_by_id(manual_rows)
        decision_counts = Counter(str(row.get("manual_decision", "pending")) for row in decisions.values())
        accepted_ids = {data_id for data_id, row in decisions.items() if row.get("manual_decision") == "accept"}
        rejected_ids = {data_id for data_id, row in decisions.items() if row.get("manual_decision") == "reject"}
        llm_suitable_ids = {
            str(row["data_id"])
            for row in successful_rows
            if bool(row.get("llm_annotation", {}).get("suitable_for_visual_qa"))
        }
        comparable_ids = (accepted_ids | rejected_ids) & set(latest)
        agreements = sum(
            (data_id in accepted_ids) == (data_id in llm_suitable_ids) for data_id in comparable_ids
        )
        report["manual_review"] = {
            "accepted": decision_counts.get("accept", 0),
            "rejected": decision_counts.get("reject", 0),
            "reviewed": len(accepted_ids | rejected_ids),
            "pending_from_llm_records": max(0, total - len(accepted_ids | rejected_ids)),
            "llm_manual_agreement_count": agreements,
            "llm_manual_agreement_percent": pct(agreements, len(comparable_ids)),
        }
    return report


def print_report(report: dict[str, Any]) -> None:
    records = report["records"]
    print("Visual QA annotation summary")
    print(f"Unique records: {records['unique_records']}")
    print(f"Successful: {records['successful']} | Errors: {records['errors']}")
    print("\nKey metrics")
    for name, metric in report["key_metrics"].items():
        print(f"- {name}: {metric['count']} ({metric['percent']:.2f}%)")
    print("\nDistributions")
    for name, values in report["distributions"].items():
        print(f"- {name}: {json.dumps(values, ensure_ascii=False, sort_keys=True)}")
    if "manual_review" in report:
        print("\nManual review")
        for name, value in report["manual_review"].items():
            print(f"- {name}: {value}")


def main() -> int:
    args = parse_args()
    llm_rows = read_json_or_jsonl(args.annotations)
    manual_rows = read_json_or_jsonl(args.manual_annotations) if args.manual_annotations else None
    report = build_report(llm_rows, manual_rows)
    print_report(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nSaved JSON report: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
