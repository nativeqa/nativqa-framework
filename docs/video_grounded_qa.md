# Video-grounded open-ended QA

`scripts/generate_video_grounded_qa_gemini.py` sends each local video to
Gemini 3.5 Flash through Vertex AI and generates exactly two English open-ended
QA pairs: one short-form pair and one long-form pair.

The model receives the video and generation prompt only. Existing ActivityNetQA
questions, answers, video names, and other source metadata are retained in the
output for provenance but are not included in the model request.

## Install

Use Python 3.10+ from a repository checkout.
Install the Python dependencies and ensure `ffmpeg` and `ffprobe` are on `PATH`:

```bash
python3 -m pip install -r requirements-video.txt
ffmpeg -version
ffprobe -version
```

The credential must be a Google service-account JSON with permission to invoke
Gemini on Vertex AI. Its `project_id` is used automatically. The default Vertex
location is `global` and the default model is `gemini-3.5-flash`.

The examples assume you have prepared local videos and an input JSONL manifest;
the dataset files are not included in this repository.

## Smoke test

Process one video and replace an existing smoke output:

```bash
python3 scripts/generate_video_grounded_qa_gemini.py \
  --input data/activitynetqa_100/samples.jsonl \
  --videos-root data/activitynetqa_100 \
  --credentials envs/google-service-account.json \
  --output data/activitynetqa_100/generated_video_qa_gemini.smoke.jsonl \
  --offset 13 \
  --limit 1 \
  --overwrite \
  --fail-fast
```

## Full 100-video run

```bash
python3 scripts/generate_video_grounded_qa_gemini.py \
  --input data/activitynetqa_100/samples.jsonl \
  --videos-root data/activitynetqa_100 \
  --credentials envs/google-service-account.json \
  --output data/activitynetqa_100/generated_video_qa_gemini.jsonl
```

The JSONL output is resumable. Existing successful `data_id` values are skipped;
error rows remain auditable and are retried on the next run. Use `--overwrite`
to start again.

Gemini accepts several video formats directly. Unsupported `.mkv` files and
inputs larger than 14 MiB are automatically transcoded to a temporary MP4. The
temporary video is deleted after each request and the original dataset video is
never modified.

Each successful row contains:

- the original source record and local video path;
- exact video duration and preprocessing information;
- one short and one long QA pair;
- timestamped evidence intervals;
- model version, finish reason, and token-usage metadata.
