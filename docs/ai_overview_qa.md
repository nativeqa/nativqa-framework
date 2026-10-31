# AI Overview QA collection

NativQA can collect Google's AI Overview text as **unverified answer candidates**.
The default remains legacy extraction. AI Overview citations help locate evidence;
they do not establish that an answer is correct or suitable as benchmark gold.

## Extraction modes

| Mode | Engines | Collected answers |
| --- | --- | --- |
| `legacy` (default) | Google, Bing, Yahoo | Existing related-question snippets/lists and `questions_and_answers` |
| `hybrid` | Google text search | Legacy answers plus AI Overview candidates |
| `ai_overview` | Google text search | AI Overview candidates only |

The Python API accepts the same optional `qa_extraction` argument; existing calls
to `run_nativqa` retain their legacy behavior. Image and video search are unchanged.
Requesting an AI mode for another engine or search type produces an explicit error.

```bash
python3 -m nativqa \
  --engine google \
  --search_type text \
  --input_file data/test_query.csv \
  --country_code qa \
  --location "Doha, Qatar" \
  --env envs/api_key.env \
  --output_dir results/ai_overview_trial \
  --n_iter 1 \
  --qa_extraction hybrid \
  --ai_overview_max_requests 5
```

For a comparable legacy collection, use `--qa_extraction legacy` and a separate
output directory. Extraction/search settings and the seed fingerprint are stored
with each run. Changing the seeds, engine, locale, mode, or request limit requires
a new output directory to avoid mixing cached responses.
Seed fingerprints normalize CRLF/LF line endings before hashing UTF-8 text.

## How references are resolved

An AI Overview may contain `text_blocks` and `references` directly, or only a
`page_token` and a SerpApi link. Deferred tokens require a follow-up request with
`engine=google_ai_overview`. SerpApi documents that these tokens expire within one
minute, so NativQA resolves them inside each seed search, before the next seed.
See the [SerpApi AI Overview API](https://serpapi.com/google-ai-overview-api) and
[related-question schema](https://serpapi.com/related-questions).

AI modes request fresh Google responses (`no_cache=true`) because cached searches
can contain expired tokens. They use up to `--ai_overview_max_requests` additional
requests **per seed**, prioritizing related questions and then the seed overview.
At the default limit, ten seeds require at most ten Google searches plus fifty
overview requests for one iteration. Later iterations can expand the query set.
Use `--n_iter 1` for a bounded trial. A limit of `0` processes embedded overviews
only. Legacy mode makes no overview requests.

Google text searches can also opt into [Related Questions expansion](related_questions.md)
with `--related_questions_max_requests`. Expanded questions share the same per-seed
AI Overview budget with the initial questions and seed overview. Initial overview
tokens are resolved before expansion; each new batch is resolved before fetching
the next batch. Repeated overview tokens share a cache across these batches.

Repeated tokens within a seed share one request. Empty, expired, failed, or
budget-limited overviews retain their status in the raw output and do not suppress
classic answers or other overviews. Expired tokens are not automatically retried.
An old token-only collection must be searched again to obtain fresh tokens.

`extract_ai_overview_answer` is a pure function in `nativqa/qa_extraction.py`.
It flattens paragraphs, headings, nested lists, expandable sections, and tables
while preserving block-to-reference indexes. It never manufactures an answer from
reference titles/snippets. Literal or escaped HTML comments containing Google
rendering metadata are removed from extracted text, and adjacent duplicate
paragraphs are merged with their citation indexes. The raw response is retained.
An export requires answer text and at least one usable
HTTP(S) reference; citation indexes must map to the corresponding references when
present. References without block indexes are labeled `citation_scope=overview`,
which does not imply that every claim has a citation.

A seed overview is paired with the seed query; a related overview is paired with
that related question. The seed overview is never reused to answer every related
question. Response-supplied URLs are parsed only for known SerpApi tokens; requests
use the fixed SerpApi client endpoint.

## Outputs and provenance

The existing seven TSV columns remain unchanged. Generated rows use
`question_type=ai_overview`; multiple `answer_URLs` are newline-separated inside a
quoted TSV field. Legacy mode retains the existing output schema and answers.

For an input named `test_query.csv`, AI modes also write:

- `iteration_<n>/output/ai_overview_candidates.jsonl`: candidate records for the iteration.
- `dataset/test_query.ai_overview_candidates.jsonl`: candidates deduplicated by question and answer across iterations.
- `iteration_<n>/output/original_response.json`: original search fields plus each overview's resolution status and follow-up response under `nativqa_ai_overview`.
- `qa_extraction_config.json`: extraction/search settings and the input SHA-256 used by the run.

Candidate records include the answer, question, seed query, category, source URLs,
reference objects, ordered answer blocks with citation indexes, location/search
parameters, search IDs/metadata, and collection time. Every candidate has:

```json
{
  "answer_origin": "google_ai_overview",
  "verification_status": "unverified",
  "gold_eligible": false
}
```

These fields describe collection status. NativQA does not automatically verify or
promote generated answers. A domain reliability label alone is insufficient: it
does not check whether the cited page supports the generated claim. Keep the JSONL
provenance when using the TSV; its `question_type` also distinguishes generated
rows from legacy rows. Image/video exports are unaffected.

## Recommended benchmark use

Use overviews to discover questions, candidate answers, and evidence sources.
For the benchmark's scored reference answers, retrieve and archive cited passages,
check every substantive claim, and have qualified annotators write or confirm the
answer from that evidence. Include review of language, local context, ambiguity,
and time-sensitive facts. Apply evidence checks to legacy snippets too.

Google describes AI Overviews as generated summaries and warns that AI responses
can contain mistakes ([Google Search Help](https://support.google.com/websearch/answer/14901683?hl=en)).
Research on other generative search engines also found unsupported statements
and inaccurate citations; this supports checking evidence rather than treating
the presence of citations as validation. It does not provide an error rate for
today's Google AI Overviews ([Liu, Zhang, and Liang, 2023](https://arxiv.org/abs/2304.09848)).

For an LLM benchmark, accepting another generative system's output directly as
gold may reward agreement with its wording or mistakes. This is a methodological
risk, not a measured effect in this framework. Keep generated candidates separate
from independently reviewed gold data and report answer provenance. Freeze search
dates, language, location, evidence snapshots, and dataset versions so future
evaluations do not silently change with live search. If evaluating search
faithfulness itself, overview answers can instead be the system outputs being
evaluated against independently annotated source evidence.

## Offline tests

```bash
python3 -m unittest tests.test_nativqa -v
```

Tests cover legacy Google/Bing/Yahoo extraction, deferred and embedded overviews,
nested answers and citations, correct question association, request limits,
error isolation, credential redaction, and resuming cached runs without API calls.
