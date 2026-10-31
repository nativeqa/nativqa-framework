"""Validate the ten-query live run and save question/answer excerpts."""

import csv
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
KEY_FILE = ROOT.parent / "envs/api_key.txt"
if not KEY_FILE.is_file():
    KEY_FILE = ROOT / "envs/api_key.env"
OUTPUT = ROOT / "results/ten_sample_queries"
RUN = OUTPUT / "text/seeds"
# Positions selected after reviewing the saved responses for relevance to each seed.
# These remain unverified search snippets, separate from NativQA's QA dataset.
REVIEWED_POSITIONS = [6, 3, 3, 3, 2, 2, 1, 1, 1, 3]


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    seeds = read_tsv(ROOT / "experiments/ten_sample_queries/seeds.tsv")
    pairs = read_tsv(RUN / "dataset/seeds.tsv")
    raw_pairs = read_tsv(RUN / "iteration_1/output/all_related_question_answers.tsv")
    duplicate_rows = read_tsv(RUN / "dataset/seeds.duplicate_qa.tsv")
    responses = json.loads((RUN / "iteration_1/output/original_response.json").read_text())
    failures = [json.loads(line) for line in
                (RUN / "iteration_1/output/failed.jsonl").read_text().splitlines() if line.strip()]
    counts = Counter(row["input_query"] for row in pairs if row["answer"].strip())
    excerpts = []
    for seed in seeds:
        candidates = [row for row in pairs
                      if row["input_query"] == seed["query"] and row["answer"].strip()]
        # Prefer a short, readable answer for the report; preserve full answers in the dataset.
        chosen = min(candidates, key=lambda row: len(row["answer"])) if candidates else {}
        excerpt = " ".join(chosen.get("answer", "").split())
        if len(excerpt) > 240:
            excerpt = excerpt[:237].rsplit(" ", 1)[0] + "..."
        excerpts.append({
            "seed_query": seed["query"], "category": seed["topic"],
            "qa_pairs": counts[seed["query"]],
            "question": chosen.get("question", ""), "answer_snippet": excerpt,
            "source_url": chosen.get("answer_URLs", ""),
        })
    response_queries = [response.get("search_parameters", {}).get("q") for response in responses]
    expected_queries = [seed["query"] for seed in seeds]
    api_statuses = Counter(response.get("search_metadata", {}).get("status", "missing")
                           for response in responses)
    related = [item for response in responses for item in response.get("related_questions", [])]
    organic_snippets = []
    for seed, position in zip(seeds, REVIEWED_POSITIONS):
        response = next(response for response in responses
                        if response.get("search_parameters", {}).get("q") == seed["query"])
        result = next((result for result in response.get("organic_results", [])
                       if result.get("position") == position), {})
        text = " ".join(result.get("snippet", "").split())
        # Keep each excerpt to at most 25 words, preserving the supplied search text.
        words = text.split()
        excerpt = " ".join(words[:25]) + (" ..." if len(words) > 25 else "")
        organic_snippets.append({
            "question": seed["query"], "category": seed["topic"],
            "answer_snippet": excerpt, "source_title": result.get("title", ""),
            "source_url": urljoin("https://www.google.com", result.get("link", "")),
            "organic_position": position, "snippet_kind": "organic_search_candidate",
        })
    checks = {
        "exactly_ten_distinct_seed_queries": len(set(expected_queries)) == len(seeds) == 10,
        "exactly_one_response_per_seed": Counter(response_queries) == Counter(expected_queries),
        "all_serpapi_responses_successful": api_statuses == {"Success": 10},
        "no_api_errors": all("error" not in response for response in responses),
        "no_failed_queries": not failures,
        "every_seed_has_nonempty_qa": all(counts[query] > 0 for query in expected_queries),
        "unique_question_answer_pairs": bool(pairs) and len({(row["question"], row["answer"]) for row in pairs}) == len(pairs),
        "all_qa_have_source_urls": bool(pairs) and all(row["answer_URLs"].startswith("http") for row in pairs),
        "organic_snippet_for_every_seed": all(item["answer_snippet"] for item in organic_snippets),
    }
    key = dotenv_values(KEY_FILE).get("API_KEY")
    checks["credential_absent_from_outputs"] = bool(key) and not any(
        key.encode() in path.read_bytes() for path in OUTPUT.rglob("*") if path.is_file()
    )
    report = {
        "report_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "https://github.com/nativeqa/nativqa-framework",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "python_version": platform.python_version(),
        "configuration": {"engine": "google", "search_type": "text", "country_code": "qa",
                          "location": "Doha, Qatar", "iterations": 1, "random_seed": 42},
        "seed_queries": len(seeds), "serpapi_responses": len(responses),
        "api_statuses": dict(api_statuses), "failed_queries": len(failures),
        "raw_qa_rows": len(raw_pairs), "deduplicated_qa_pairs": len(pairs),
        "discarded_rows": len(duplicate_rows),
        "related_question_items": len(related),
        "related_question_types": dict(Counter(item.get("type", "unspecified") for item in related)),
        "related_questions_with_inline_answers": sum(bool(item.get("snippet") or item.get("list")) for item in related),
        "related_questions_with_source_links": sum(bool(item.get("link")) for item in related),
        "organic_result_items": sum(len(response.get("organic_results", [])) for response in responses),
        "checks": checks, "framework_qa_snippets": excerpts,
        "organic_search_snippets": organic_snippets,
        "outcome": f"Collected {len(pairs)} framework QA pairs from {len(responses)} saved search responses.",
        "interpretation": "Legacy extraction uses inline related-question answers and source links. AI Overview references are not resolved in legacy mode. Organic snippets are separately extracted answer candidates, not framework-generated QA or independently fact-checked answers. No additional API calls were made to produce the snippets.",
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (OUTPUT / "framework_qa_snippets.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(excerpts[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(excerpts)
    with (OUTPUT / "organic_search_snippets.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(organic_snippets[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(organic_snippets)
    (OUTPUT / "qa_snippets.tsv").unlink(missing_ok=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit("At least one live-run validation failed; see run_summary.json.")


if __name__ == "__main__":
    main()
