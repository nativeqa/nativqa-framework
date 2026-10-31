"""Validate the saved Bing run and export reviewed search snippet candidates.

This script reads existing responses only; it makes no additional API requests.
Search snippets remain separate from the framework's extracted QA dataset.
"""

import base64
import csv
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote, quote_plus, urldefrag, urlparse

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
KEY_FILE = ROOT.parent / "envs/api_key.txt"
if not KEY_FILE.is_file():
    KEY_FILE = ROOT / "envs/api_key.env"
OUTPUT = ROOT / 'results/ten_sample_queries_bing'
RUN = OUTPUT / 'text/seeds'
SEEDS = ROOT / 'experiments/ten_sample_queries/seeds.tsv'

# Reviewed excerpts from this run, in seed order. Limit each quotation to 25 words.
# For later runs, fail if a reviewed passage is absent instead of reusing stale text.
SELECTIONS = [
    {'kind': 'organic_results', 'position': 1,
     'text': 'The short answer: More than 30 years in the wild.'},
    {'kind': 'answer_box',
     'text': 'omega-3 fish oil (EPA and DHA) added to food for systemic anti-inflammatory support, colloidal oatmeal baths'},
    {'kind': 'answer_box',
     'text': 'Brown sugar is not meaningfully healthier than white sugar.'},
    {'kind': 'answer_box',
     'text': 'Nutritionally, a single serving (approx. 100g) provides 309 calories, 5.11g of protein, 33.1g of carbohydrates, and 17.4g of fat.'},
    {'kind': 'organic_results', 'position': 1,
     'text': 'Cardamom pairs beautifully with chocolate, coffee, citrus, and vanilla in sweet',
     'limitation': 'Describes cardamom pairings; does not directly answer which foods to serve with cardamom coffee.'},
    {'kind': 'answer_box',
     'text': 'fruits, vegetables, protein, grains, and dairy'},
    {'kind': 'answer_box',
     'text': 'BDT 1,50,000 to BDT 3,00,000'},
    {'kind': 'answer_box',
     'text': 'Use a clear subject line, proper greeting with titles, concise body text, professional closing, and complete signature'},
    {'kind': 'organic_results', 'position': 3,
     'text': 'This list contains 7 cities of Bangladesh that start with the letter C.',
     'limitation': 'The snippet does not enumerate the city names.'},
    {'kind': 'answer_box',
     'text': 'Is eating fish truly off-limits? Should you really avoid a refreshing swim?',
     'limitation': 'The snippet raises possible myths but does not explain or substantiate them.'},
]


def read_tsv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def source_url(raw_url):
    """Decode a Bing destination already in the response, without fetching it."""
    parsed = urlparse(raw_url)
    if parsed.hostname in ('bing.com', 'www.bing.com') and parsed.path == '/ck/a':
        target = parse_qs(parsed.query).get('u', [''])[0]
        if target.startswith('a1'):
            encoded = target[2:]
            try:
                decoded = base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)).decode('utf-8')
            except (ValueError, UnicodeError):
                decoded = ''
            if urlparse(decoded).scheme in ('http', 'https'):
                raw_url = decoded
    return urldefrag(raw_url)[0]


def main():
    seeds = read_tsv(SEEDS)
    pairs = read_tsv(RUN / 'dataset/seeds.tsv')
    raw_rows = read_tsv(RUN / 'iteration_1/output/all_related_question_answers.tsv')
    discarded = read_tsv(RUN / 'dataset/seeds.duplicate_qa.tsv')
    responses = json.loads((RUN / 'iteration_1/output/original_response.json').read_text())
    failures = [json.loads(line) for line in
                (RUN / 'iteration_1/output/failed.jsonl').read_text().splitlines() if line.strip()]
    response_queries = [r.get('search_parameters', {}).get('q') for r in responses]
    expected_queries = [seed['query'] for seed in seeds]
    api_statuses = Counter(r.get('search_metadata', {}).get('status', 'missing') for r in responses)
    counts = Counter(pair['input_query'] for pair in pairs if pair['answer'].strip())
    snippets = []
    if len(seeds) != len(SELECTIONS):
        raise SystemExit('The seed count differs from the reviewed selections.')
    for seed, selection in zip(seeds, SELECTIONS):
        response = next(r for r in responses if r['search_parameters']['q'] == seed['query'])
        if selection['kind'] == 'answer_box':
            item = response.get('answer_box', {})
            source = next(iter(item.get('sources', [])), item)
        else:
            item = next(result for result in response.get('organic_results', [])
                        if result.get('position') == selection['position'])
            source = item
        text = ' '.join((item.get('snippet') or '').split())
        excerpt = selection['text']
        if excerpt not in text or len(excerpt.split()) > 25:
            raise SystemExit(f'Reviewed excerpt absent or too long: {seed["query"]}')
        raw_url = source.get('link', '')
        snippets.append({
            'question': seed['query'], 'category': seed['topic'],
            'answer_snippet': excerpt, 'snippet_kind': selection['kind'],
            'source_title': source.get('title', ''), 'source_url': source_url(raw_url),
            'raw_source_url': raw_url, 'organic_position': selection.get('position', ''),
            'framework_qa_pairs': counts[seed['query']],
            'verification_status': 'unverified_search_excerpt',
            'limitation': selection.get('limitation', ''),
        })
    key = dotenv_values(KEY_FILE).get('API_KEY')
    secrets = {key, quote(key, safe=''), quote_plus(key)} if key else set()
    checks = {
        'exactly_ten_distinct_seed_queries': len(seeds) == len(set(expected_queries)) == 10,
        'exactly_one_response_per_seed': Counter(response_queries) == Counter(expected_queries),
        'all_serpapi_responses_successful': api_statuses == {'Success': 10},
        'all_responses_use_bing': all(r.get('search_parameters', {}).get('engine') == 'bing' for r in responses),
        'no_api_errors': all('error' not in r for r in responses),
        'no_failed_queries': not failures,
        'snippet_for_every_seed': len(snippets) == 10 and all(row['answer_snippet'] for row in snippets),
        'all_snippets_have_source_urls': all(row['source_url'].startswith('https://') for row in snippets),
        'credential_absent_from_outputs': bool(key) and not any(
            secret.encode() in path.read_bytes()
            for path in OUTPUT.rglob('*') if path.is_file() for secret in secrets),
    }
    metadata = [response.get('search_metadata', {}) for response in responses]
    report = {
        'report_generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'configuration': {'engine': 'bing', 'search_type': 'text', 'country_code_cli': 'qa',
                          'location': 'Doha, Qatar', 'iterations': 1, 'qa_extraction': 'legacy',
                          'sample_random_seed': 42, 'credential_file': '../envs/api_key.txt'},
        'sample_selection': json.loads((SEEDS.parent / 'selection.json').read_text()),
        'python_version': platform.python_version(),
        'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'repository_has_existing_changes': bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        'seed_queries': len(seeds), 'serpapi_responses': len(responses),
        'api_statuses': dict(api_statuses), 'failed_queries': len(failures),
        'raw_export_rows': len(raw_rows), 'framework_qa_pairs': len(pairs),
        'discarded_rows': len(discarded),
        'unavailable_placeholder_rows': sum(row['question'] == 'Not Available' for row in raw_rows),
        'answer_boxes': sum(bool(response.get('answer_box')) for response in responses),
        'organic_results': sum(len(response.get('organic_results', [])) for response in responses),
        'related_question_items': sum(len(response.get('related_questions', [])) for response in responses),
        'questions_and_answers_items': sum(len(response.get('questions_and_answers', [])) for response in responses),
        'api_searches': [{'id': item.get('id'), 'created_at': item.get('created_at'),
                          'total_time_taken': item.get('total_time_taken')} for item in metadata],
        'checks': checks, 'snippets': snippets,
        'outcome': f'Collected {len(pairs)} framework QA pairs from {len(responses)} saved Bing responses.',
        'interpretation': 'Legacy extraction uses related_questions and questions_and_answers, '
                          'and ignores answer_box and organic_results. Not Available placeholder rows '
                          'are discarded from the final dataset. The separately exported '
                          'snippets are unverified search excerpts, not framework QA pairs or benchmark gold answers.',
        'additional_api_calls_for_summary': 0,
    }
    (OUTPUT / 'run_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    with (OUTPUT / 'qa_snippets.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(snippets[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(snippets)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('sample_selection', 'snippets', 'api_searches')}, indent=2))
    if not all(checks.values()):
        raise SystemExit('Bing run validation failed; see run_summary.json.')


if __name__ == '__main__':
    main()
