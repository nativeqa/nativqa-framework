# Google Related Questions expansion

NativQA already extracts Related Questions and their snippet/list answers from
normal search responses. The optional `--related_questions_max_requests` setting
also retrieves additional People also ask questions using SerpApi's
[`google_related_questions` API](https://serpapi.com/google-related-questions-api).

## Supported engines and modes

| Search configuration | Related-question expansion |
| --- | --- |
| Google text, `legacy`, `hybrid`, or `ai_overview` | Supported |
| Bing or Yahoo text | Unsupported; use limit `0` |
| Image or video searches | Unsupported; use limit `0` |

The default is `0`, which disables all expansion requests. Unsupported engines
and search types reject a positive limit before loading credentials, creating
output files, or making API requests. Negative limits are also rejected.

Keep `--engine google` for seed searches. The dedicated follow-up endpoint needs
a `next_page_token` from Google's `related_questions` block; it cannot accept a
seed query directly or expand Bing/Yahoo results. A search without tokens produces
no expansion requests. Expansion does not guarantee an answer for every question.

## Usage

```bash
python3 -m nativqa \
  --engine google --search_type text \
  --input_file tests/data/test_query.csv \
  --country_code qa --location "Doha, Qatar" \
  --env envs/api_key.env \
  --output_dir results/google_related_questions \
  --related_questions_max_requests 3 \
  --qa_extraction hybrid --ai_overview_max_requests 5 \
  --n_iter 1
```

The Python API accepts the same optional `related_questions_max_requests=3`
keyword argument on `run_nativqa`. The experiment runner at
`experiments/ten_sample_queries/run_samples.py` also forwards this CLI option.
Use a new output directory when enabling expansion or changing its limit;
completed runs resume from their saved responses without additional API calls.
Older unexpanded caches remain usable with limit `0`.

## Requests and answer extraction

The collector requests fresh Google seed responses, then follows expansion
tokens breadth first. Each distinct token is requested at most once per seed.
Cycles and duplicate tokens do not cause repeated calls. Both successful and
failed attempts count toward the limit; each request has a 30-second timeout.
Errors, empty responses, and missing tokens preserve the initial questions and
answers. Failed tokens are recorded without an automatic retry.

Returned questions with distinct answer content are appended to the seed's
`related_questions` list. Repeated question/answer content is merged while
different answers to the same question are retained. Their original seed query
and category remain attached to the exported rows.

- `legacy` exports classic snippets/lists with source links, including expanded
  questions. It makes no AI Overview follow-up calls.
- `hybrid` exports classic answers and AI Overview candidates.
- `ai_overview` exports AI Overview candidates only.

AI Overview tokens from the initial response are resolved before expansion.
Fresh tokens from each expansion batch are resolved before requesting the next
batch. The `--ai_overview_max_requests` budget and token cache are shared across
all batches for that seed. Embedded overview text can still be extracted when
the overview request budget is exhausted. See [AI Overview collection](ai_overview_qa.md)
for candidate provenance and verification status.

For one iteration, `N` seeds, expansion limit `R`, and overview limit `A`, the
maximum request count is `N × (1 + R + A)` in an AI mode, or `N × (1 + R)` in
legacy mode. For ten seeds, `R=3` and `A=5` mean at most 90 requests, including
the ten seed searches. Increasing `--n_iter` can introduce more seed queries.

## Saved provenance

Expanded classic answers use the existing seven-column TSV schema and
`question_type=related_questions`. AI answers use `question_type=ai_overview`.
Answer boxes and organic search snippets are not substituted for related answers.

Saved `summary.jsonl` and `original_response.json` records contain:

- `related_questions`: initial questions plus distinct expanded entries.
- `nativqa_related_questions`: configured limit, attempts, added-question count,
  pending-token count, and whether the limit stopped traversal.
- `nativqa_related_questions.requests`: parent question, token, depth, status,
  timestamp, and credential-redacted raw response or error for each attempt.
- `nativqa_related_question` on each expanded entry: parent question, depth,
  request index, and follow-up search metadata.

The recorded follow-up responses preserve their fields before framework IDs or
AI resolution states are added. Expanded AI candidates also carry their
`related_question_expansion` provenance in the candidate JSONL export.
`qa_extraction_config.json` records the expansion limit to prevent mixing caches
from different settings.
