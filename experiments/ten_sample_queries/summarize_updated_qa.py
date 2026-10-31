"""Validate and summarize the opt-in AI Overview run without additional requests."""

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

from dotenv import dotenv_values
from nativqa.qa_extraction import extract_ai_overview_answer


ROOT = Path(__file__).resolve().parents[2]
KEY_FILE = ROOT.parent / "envs/api_key.txt"
if not KEY_FILE.is_file():
    KEY_FILE = ROOT / "envs/api_key.env"
OUTPUT = ROOT / "results/qa_extract_updated"
RUN = OUTPUT / "text/seeds"


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    seeds = read_tsv(ROOT / "experiments/ten_sample_queries/seeds.tsv")
    rows = read_tsv(RUN / "dataset/seeds.tsv")
    candidates = [json.loads(line) for line in
                  (RUN / "dataset/seeds.ai_overview_candidates.jsonl").read_text().splitlines()]
    responses = json.loads((RUN / "iteration_1/output/original_response.json").read_text())
    failures = (RUN / "iteration_1/output/failed.jsonl").read_text().splitlines()
    statuses = Counter()
    overview_errors = []
    rejected = []
    for response in responses:
        targets = list(response.get("related_questions", [])) + [response.get("ai_overview", {})]
        for target in targets:
            state = target.get("nativqa_ai_overview")
            if state:
                statuses[state["status"]] += 1
                if state.get("error"):
                    overview_errors.append({"seed_query": response["search_parameters"]["q"],
                                            "question": target.get("question"), "error": state["error"]})
                if state['status'] in ('fetched', 'embedded'):
                    overview = state.get('response', {}).get('ai_overview', target.get('ai_overview', target))
                    if extract_ai_overview_answer(overview) is None:
                        rejected.append({'seed_query': response['search_parameters']['q'],
                                         'question': target.get('question', response['search_parameters']['q']),
                                         'reason': 'missing_references' if not overview.get('references')
                                                   else 'no_usable_answer_or_citation_mapping'})
    per_seed = Counter(row["input_query"] for row in rows)
    snippets = []
    for seed in seeds:
        options = [row for row in rows if row["input_query"] == seed["query"]]
        chosen = next((row for row in options if row["question"] == seed["query"]),
                      options[0] if options else {})
        answer = " ".join(chosen.get("answer", "").split())
        snippets.append({"seed_query": seed["query"], "qa_pairs": len(options),
                         "question": chosen.get("question", ""),
                         "answer_snippet": " ".join(answer.split()[:25]),
                         "question_type": chosen.get("question_type", ""),
                         "source_urls": chosen.get("answer_URLs", "")})
    key = dotenv_values(KEY_FILE)["API_KEY"]
    checks = {
        "ten_successful_seed_responses": len(responses) == 10 and all(
            response.get("search_metadata", {}).get("status") == "Success" for response in responses),
        "expected_seeds_only": Counter(response["search_parameters"]["q"] for response in responses)
                               == Counter(seed["query"] for seed in seeds),
        "no_failed_seed_queries": not failures,
        "qa_pair_for_every_seed": all(per_seed[seed["query"]] for seed in seeds),
        "generated_rows_have_provenance": {(row["data_id"], row["answer"]) for row in rows
                                           if row["question_type"] == "ai_overview"}
                                          == {(row["data_id"], row["answer"]) for row in candidates},
        "all_candidates_unverified": bool(candidates) and all(
            row["verification_status"] == "unverified" and row["gold_eligible"] is False
            and row["source_urls"] and row["answer_blocks"] and row["collected_at_utc"]
            for row in candidates),
        "no_html_comment_metadata_in_answers": all(
            '<!--' not in row['answer'] and '\\x3c!--' not in row['answer']
            and '\\u003c!--' not in row['answer'] for row in candidates),
        "no_credential_in_outputs": all(key.encode() not in path.read_bytes()
                                        for path in OUTPUT.rglob("*") if path.is_file()),
    }
    report = {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed_queries": len(seeds), "seed_responses": len(responses), "qa_pairs": len(rows),
        "question_types": dict(Counter(row["question_type"] for row in rows)),
        "ai_overview_candidates": len(candidates), "overview_resolution_statuses": dict(statuses),
        "overview_errors": overview_errors, "rejected_overviews": rejected,
        "qa_pairs_per_seed": dict(per_seed),
        "checks": checks, "snippets": snippets,
        "interpretation": "AI Overview answers are collected candidates requiring independent source review before benchmark gold use.",
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (OUTPUT / "qa_snippets.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(snippets[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(snippets)
    print(json.dumps({key: value for key, value in report.items() if key != "snippets"}, indent=2))
    if not all(checks.values()):
        raise SystemExit("At least one live validation check failed.")


if __name__ == "__main__":
    main()
