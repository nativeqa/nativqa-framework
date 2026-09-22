# NativQA Framework: Multilingual QA Dataset Collection for LLM Evaluation

[![PyPI version](https://img.shields.io/pypi/v/nativqa-framework)](https://pypi.org/project/nativqa-framework/)
[![Total PyPI downloads](https://img.shields.io/pepy/dt/nativqa-framework)](https://www.pepy.tech/projects/nativqa-framework)
[![Python versions](https://img.shields.io/pypi/pyversions/nativqa-framework)](https://pypi.org/project/nativqa-framework/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Website](https://img.shields.io/badge/website-nativeqa.github.io-blue)](https://nativeqa.github.io/)

*Download counts are provided by Pepy and include PyPI and known mirrors; they do not represent unique users.*

**NativQA Framework** is an open-source toolkit for collecting multilingual, culturally aligned, search-grounded question answering datasets. It helps researchers and practitioners build **natural query**, **multilingual QA**, **region-aware evaluation**, and **visual question answering (VQA)** resources from real search results across different countries, languages, and domains.

NativQA is designed for teams working on **LLM evaluation**, **benchmark construction**, **fine-tuning data collection**, **culturally grounded QA**, and **location-specific search analysis**. Starting from seed queries, it expands through related searches and search result structures to produce datasets that reflect everyday information needs in local contexts.

More details are available at [https://nativeqa.github.io/](https://nativeqa.github.io/).

![NativQA framework, demonstrating the data collection and annotation process.](https://raw.githubusercontent.com/nativeqa/nativqa-framework/main/nativqa_framework.png)

Here is a quick overview video:
> [![NativQA Framework Overview](https://markdown-videos-api.jorgenkh.no/youtube/gTgpeYqWm9s)](https://youtu.be/gTgpeYqWm9s)

## Why NativQA?

- Build **multilingual QA datasets** from real search behavior instead of synthetic prompts alone.
- Collect **culturally aligned natural queries** tied to a country, city, or language context.
- Generate datasets for **text QA**, **image search**, and **video search** workflows.
- Expand seed queries iteratively through related searches and related questions.
- Support **LLM benchmarking**, **fine-tuning**, **VQA dataset construction**, and **domain reliability validation**.
- Work with lightweight CSV or TSV inputs that are easy to create, version, and audit.

## Use Cases

- Evaluate LLMs on **region-specific everyday knowledge** rather than generic global trivia.
- Create **culturally grounded benchmarks** for Arabic, English, and other languages.
- Build **image-based datasets** for visual question answering and multimodal evaluation.
- Compare how search results differ by **location**, **country code**, and **multiple-country constraints**.
- Bootstrap **low-resource language** datasets with template-driven or user-collected seed queries.

## Search Support

| Search type | Engines | Output dataset |
| --- | --- | --- |
| `text` | Google, Bing, Yahoo | TSV |
| `image` | Google, Bing | JSON |
| `video` | Google | JSON |

## Quick Start

The pip package requires Python 3.9 or newer and a SerpAPI key for search requests.

### Option 1: Install from source

1. Clone the repository:
   ```bash
   git clone https://github.com/nativeqa/nativqa-framework/
   cd nativqa-framework
   ```
2. Install the requirements:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
3. Add your SerpAPI key to `envs/api_key.env`:
   ```bash
   mkdir -p envs
   printf 'API_KEY="your-serpapi-key"\n' > envs/api_key.env
   ```
4. Run the sample text-search pipeline:
   ```bash
   python3 -m nativqa \
     --engine google \
     --search_type text \
     --input_file data/test_query.csv \
     --country_code qa \
     --location "Doha, Qatar" \
     --env envs/api_key.env \
     --n_iter 3
   ```

### Option 2: Install from PyPI

This installs the core search package and the `nativqa` command. Check the installed
version with `nativqa --version` (available from version 0.1.3).

1. Install the package:
   ```bash
   python3 -m pip install nativqa-framework
   ```
2. Create an env file for your SerpAPI key:
   ```bash
   mkdir -p envs
   printf 'API_KEY="your-serpapi-key"\n' > envs/api_key.env
   ```
3. Create a small seed query file:
   ```bash
   cat > test_query.csv <<'EOF'
   topic,query
   animal,What is New York City State animal?
   food,What food is New York City famous for?
   travel,What is the best time to visit Central Park?
   EOF
   ```
4. Run NativQA:
   ```bash
   nativqa \
     --engine google \
     --search_type text \
     --input_file test_query.csv \
     --country_code us \
     --location "New York, United States" \
     --env envs/api_key.env \
     --n_iter 1
   ```

## Input Format

NativQA accepts seed queries in CSV or TSV format with two columns:

```csv
topic,query
animal,What is New York City State animal?
food,What food is New York City famous for?
travel,What is the best time to visit Central Park?
```

The `topic` column is used as a high-level category. The `query` column is the natural-language seed query sent to the search engine.

## Core CLI Parameters

- `--engine`: Search engine to use. Supported values: `google`, `bing`, `yahoo`
- `--search_type`: Search mode. Supported values: `text`, `image`, `video`
- `--input_file`: Path to the seed query CSV or TSV file
- `--country_code`: Country code for location-aware search
- `--location`: Search origin location string, for example `"Doha, Qatar"`
- `--multiple_countries`: Optional country restriction string such as `countryQA|countryBD`
- `--env`: Path to the env file containing `API_KEY`
- `--output_dir`: Optional output directory. Defaults to `./results/`
- `--n_iter`: Number of iterative search rounds

## Common Examples

### Text search

```bash
python3 -m nativqa \
  --engine google \
  --search_type text \
  --input_file data/test_query.csv \
  --country_code qa \
  --location "Doha, Qatar" \
  --env envs/api_key.env \
  --n_iter 3
```

### Image search

```bash
python3 -m nativqa \
  --engine google \
  --search_type image \
  --input_file data/test_query.csv \
  --country_code us \
  --location "New York, United States" \
  --env envs/api_key.env \
  --n_iter 1
```

### Video search

```bash
python3 -m nativqa \
  --engine google \
  --search_type video \
  --input_file data/test_query.csv \
  --country_code us \
  --location "New York, United States" \
  --env envs/api_key.env \
  --n_iter 1
```

## Output Structure

For an input file named `test_query.csv`, NativQA creates a result directory like:

```text
results/<search_type>/test_query/
```

Inside that directory:

- `dataset/`
  - Final merged dataset
  - `text` runs produce `.tsv`
  - `image` and `video` runs produce `.json`
- `iteration_<n>/output/`
  - Per-iteration raw and processed outputs such as `summary.jsonl`, `original_response.json`, and related-search files
- `completed_queries.txt`
  - Queries already processed across iterations

## Included Utilities

The repository includes helper scripts for common dataset-building tasks. These
scripts, templates, domain annotations, and demo apps are not installed by
`pip install nativqa-framework`; clone the repository to use them. Some utilities
require additional dependencies and newer Python versions; see their documentation.

- [scripts/template2seeds.py](https://github.com/nativeqa/nativqa-framework/blob/main/scripts/template2seeds.py): generate seed queries from a template file
- [scripts/download_images.py](https://github.com/nativeqa/nativqa-framework/blob/main/scripts/download_images.py): download images from an image-search dataset
- [scripts/filter_near_duplicates_flann.py](https://github.com/nativeqa/nativqa-framework/blob/main/scripts/filter_near_duplicates_flann.py): filter near-duplicate images
- [scripts/check_domain_reliability.py](https://github.com/nativeqa/nativqa-framework/blob/main/scripts/check_domain_reliability.py): retain answers from reliable domains
- [scripts/GPT_4o_labeling.py](https://github.com/nativeqa/nativqa-framework/blob/main/scripts/GPT_4o_labeling.py): annotate datasets with LLM-based labels
- [scripts/validate_visual_qa_images.py](https://github.com/nativeqa/nativqa-framework/blob/main/scripts/validate_visual_qa_images.py): validate image relevance and visual grounding with Azure OpenAI
- [scripts/generate_visually_grounded_qa.py](https://github.com/nativeqa/nativqa-framework/blob/main/scripts/generate_visually_grounded_qa.py): skip text-driven images, then generate and verify short/long image-only QA pairs
- [scripts/generate_video_grounded_qa_gemini.py](https://github.com/nativeqa/nativqa-framework/blob/main/scripts/generate_video_grounded_qa_gemini.py): generate short/long video-grounded QA pairs with Gemini 3.5 Flash
- [scripts/summarize_visual_qa_annotations.py](https://github.com/nativeqa/nativqa-framework/blob/main/scripts/summarize_visual_qa_annotations.py): report LLM and manual Visual QA annotation statistics
- [scripts/review_visual_qa_annotations.py](https://github.com/nativeqa/nativqa-framework/blob/main/scripts/review_visual_qa_annotations.py): manually accept or reject images in a Gradio app

For setup and examples, see the [visual QA guide](https://github.com/nativeqa/nativqa-framework/blob/main/docs/visual_qa_image_validation.md) and [video QA guide](https://github.com/nativeqa/nativqa-framework/blob/main/docs/video_grounded_qa.md).

## Query Collection

### Template-based seed generation

```bash
python3 scripts/template2seeds.py \
  --template_file templates/arabic_template.csv \
  --output_file templates/test.csv \
  --location "قطر"
```

### Seed query collection app

The repository also includes a lightweight Flask-based seed collection app in [seed_query_collector/app.py](https://github.com/nativeqa/nativqa-framework/blob/main/seed_query_collector/app.py) for gathering user-written queries.

## QA Validation

### Domain Reliability Check (DRC)

Manually verified domain labels are stored in [domain/annotated_domains.csv](https://github.com/nativeqa/nativqa-framework/blob/main/domain/annotated_domains.csv).

To filter answers by source reliability:

```bash
python3 scripts/check_domain_reliability.py \
  --input_file <dataset_directory>/input_filename.tsv \
  --output_file <output_directory>/output_filename.tsv
```

### LLM-based annotation

```bash
python3 scripts/GPT_4o_labeling.py \
  --input_file results/text/arabic_qa/dataset.json \
  --env_path envs/gpt4-api-key.env \
  --output_dir results/text/arabic_qa/GPT4o_labeling/ \
  --location "Qatar"
```



## License

NativQA Framework is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

You are free to share and adapt the framework for non-commercial purposes, provided that you give appropriate credit, indicate changes, and distribute contributions under the same license.

## Citation

Please cite our papers when referring to this framework:

```bibtex
@article{Alam2025nativqa,
  title={NativQA Framework: A Framework for Collecting Multilingual Culturally-Aligned Natural Queries},
  author={Alam, Firoj and Hasan, Md Arid and Laskar, Sahinur Rahman and Kutlu, Mucahid and Chowdhury, Shammur Absar},
  journal={arXiv},
  year={2025}
}

@inproceedings{hasan-etal-2025-nativqa,
    title = "{N}ativ{QA}: Multilingual Culturally-Aligned Natural Query for {LLM}s",
    author = "Hasan, Md. Arid  and
      Hasanain, Maram  and
      Ahmad, Fatema  and
      Laskar, Sahinur Rahman  and
      Upadhyay, Sunaya  and
      Sukhadia, Vrunda N  and
      Kutlu, Mucahid  and
      Chowdhury, Shammur Absar  and
      Alam, Firoj",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.770/",
    doi = "10.18653/v1/2025.findings-acl.770",
    pages = "14886--14909",
    ISBN = "979-8-89176-256-5",
    abstract = "Natural Question Answering (QA) datasets play a crucial role in evaluating the capabilities of large language models (LLMs), ensuring their effectiveness in real-world applications. Despite the numerous QA datasets that have been developed and some work done in parallel, there is a notable lack of a framework and large-scale region-specific datasets queried by native users in their own languages. This gap hinders effective benchmarking and the development of fine-tuned models for regional and cultural specificities. In this study, we propose a scalable, language-independent framework, NativQA, to seamlessly construct culturally and regionally aligned QA datasets in native languages for LLM evaluation and tuning. We demonstrate the efficacy of the proposed framework by designing a multilingual natural QA dataset, MultiNativQA, consisting of approximately {\textasciitilde}64K manually annotated QA pairs in seven languages, ranging from high- to extremely low-resource, based on queries from native speakers from 9 regions covering 18 topics. We benchmark both open- and closed-source LLMs using the MultiNativQA dataset. The dataset and related experimental scripts are publicly available for the community at: https://huggingface.co/datasets/QCRI/MultiNativQA and https://gitlab.com/nativqa/multinativqa."
}
```
