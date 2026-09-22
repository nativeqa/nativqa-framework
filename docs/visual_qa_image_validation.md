# Visual QA image validation

This workflow matches each JSON record to an image by the basename of
`image_path`, asks the configured Azure OpenAI vision deployment to annotate it,
reports aggregate statistics, and supports manual accept/reject review.

## Install

Use Python 3.10+ from a repository checkout. The example data paths below
refer to your own downloaded images and search records.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The env file must define `AZURE_API_URL`, `AZURE_API_KEY`,
`AZURE_API_VERSION`, and `AZURE_ENGINE_NAME` for your Azure deployment.

## Run GPT validation

Small five-image sample:

```bash
python scripts/validate_visual_qa_images.py \
  --input-json data/sample_image_data/California.json \
  --images-dir data/sample_image_data/images \
  --env-file envs/azure-vision.env \
  --output data/sample_image_data/California_visual_qa_annotations.sample.jsonl \
  --location California \
  --limit 5
```

Full dataset (remove `--limit` and use a full-output filename):

```bash
python scripts/validate_visual_qa_images.py \
  --input-json data/sample_image_data/California.json \
  --images-dir data/sample_image_data/images \
  --env-file envs/azure-vision.env \
  --output data/sample_image_data/California_visual_qa_annotations.jsonl \
  --location California
```

The validator resumes an existing JSONL file and skips successful `data_id`
records. Use `--overwrite` to start over. Failed records remain auditable and are
retried on a later run because only successful records are skipped.

The annotation explicitly separates:

- query relevance;
- whether the image materially grounds the answer;
- whether the requested location is confirmed, merely consistent, uncertain, or
  contradicted using the image plus source metadata such as `title`, `link`,
  `original`, and `source`;
- the combined Visual QA suitability decision.

The target location is selected per record: a non-empty JSON `location` value
takes precedence, followed by `--location`, followed by the input filename stem.
Query-echoing SerpAPI URLs, dataset directory names, and hashed filenames are not
treated as independent proof of location.

For a location-grounded item, `uncertain` and `contradicted` locations cannot be
marked suitable. Suitability requires `confirmed`, `consistent_but_unverified`,
or `not_applicable` location relevance in addition to the query, grounding,
clarity, and safety criteria.

## Generate image-only open-ended QA

Generate one short and one long QA pair for each visually interesting image:

```bash
python scripts/generate_visually_grounded_qa.py \
  --input data/sample_image_data/California_visual_qa_annotations.sample.jsonl \
  --images-dir data/sample_image_data/images \
  --env-file envs/azure-vision.env \
  --output data/sample_image_data/California_two_style_grounded_qa.jsonl \
  --only-suitable
```

For a small smoke run, add `--limit 1 --overwrite`. The script also accepts the
original JSON image-search records; omit `--only-suitable` in that case.

The generator deliberately does not send the source query, location, category,
title, description, URL, filename, or other metadata to the model. It sends only
the image and instructions. Memes, screenshots, text-dominant graphics, and
low-interest images are written with `processing_status: "skipped"` and an
`image_assessment` reason; written text is never a question target.

Eligible images receive exactly two entries in `qa_pairs`:

- `short`: at most 18 question words, a direct visual-perception target, and an
  answer of at most 25 words;
- `long`: at least 16 question words, a compositional visual-reasoning target
  combining two or more visible details, and a 25-100 word answer.

Both pairs forbid outside knowledge and must use different targets and visual
evidence. Each pair is sent with the same image to an independent verifier that
checks both question grounding and answer correctness. An ungrounded pair is
rewritten and verified again. A generated record is successful only after both
pairs receive `GROUNDED`; otherwise the JSONL row records an auditable error.
Existing version-2 success and policy-skipped IDs are resumed unless
`--overwrite` is used.

## Report statistics

```bash
python scripts/summarize_visual_qa_annotations.py \
  --annotations data/sample_image_data/California_visual_qa_annotations.sample.jsonl \
  --output-json data/sample_image_data/California_visual_qa_stats.sample.json
```

After manual review, include agreement statistics:

```bash
python scripts/summarize_visual_qa_annotations.py \
  --annotations data/sample_image_data/California_visual_qa_annotations.sample.jsonl \
  --manual-annotations data/sample_image_data/California_manual_annotations.sample.jsonl \
  --output-json data/sample_image_data/California_combined_stats.sample.json
```

## Launch manual review

```bash
python scripts/review_visual_qa_annotations.py \
  --annotations data/sample_image_data/California_visual_qa_annotations.sample.jsonl \
  --images-dir data/sample_image_data/images \
  --output data/sample_image_data/California_manual_annotations.sample.jsonl \
  --port 7860
```

Open <http://127.0.0.1:7860>. Accepting or rejecting an item saves immediately
and advances to the next item. Decisions can be revised or cleared. The app shows
accepted, rejected, pending, completion, and LLM/manual agreement statistics.
