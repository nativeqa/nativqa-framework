# Changelog

All notable changes to this project are documented in this file.

This project follows a Keep a Changelog style and uses Semantic Versioning (`MAJOR.MINOR.PATCH`) for releases.

## [Unreleased]

### Added

- Opt-in Google AI Overview QA extraction, including immediate resolution of
  deferred references, nested answer text, citation mappings, collection metadata,
  and separate unverified candidate JSONL exports. Legacy extraction remains the default.
- `--qa_extraction` and `--ai_overview_max_requests` CLI options, with offline
  regression tests for Google, Bing, Yahoo, failures, request limits, and cached runs.

- Publish the documented visual validation, image/video QA generation, annotation
  summary, and manual review scripts with setup guides and offline generator tests.

- Lightweight static demo page in `demo/` for:
  - paste seed queries
  - generate JSONL preview
  - download JSONL file
- Demo deployment instructions in `demo/README.md`.
- `CONTRIBUTING.md` with setup, test, and merge request guidance.

### Changed

- Improved root `README.md` structure and discoverability:
  - clearer project positioning and quick links
  - table of contents
  - refined quick start and output descriptions
  - demo, roadmap, and contributing sections

### Fixed

- Read legacy `questions_and_answers` from its own response block.
- Exclude TSV headers from the logged text QA row counts.
- Remove Google HTML comment metadata from extracted overview answers while
  preserving raw responses and citation mappings.
- Protect cached collections against changed seed inputs, engines, and locales;
  make experiment entry points independent of the caller's working directory.
- Pin Python 3.9 CI jobs to Ubuntu 24.04 to avoid unavailable runtimes on Ubuntu 26.04.

## [0.1.3] - 2026-09-22

### Added

- `nativqa --version` and `python -m nativqa --version`.
- Offline integration tests for text, image, and video searches.
- Package checks for wheel and source installations in GitHub Actions.

### Changed

- Require Python 3.9+ to match the existing logging API and modern build backend.
- Require setuptools 77.0.3+ for SPDX license metadata.
- Use compatible runtime dependency ranges instead of exact pins.
- Refresh PyPI description, keywords, project URLs, and README download badge.
- Clarify that repository utilities are not part of the pip installation.

### Fixed

- Preserve absolute output directory paths.
- Write CSV/TSV files without extra blank rows on Windows and read multilingual data as UTF-8.
- Close completed-query files before subsequent processing.
- Replace the core test's incomplete API call with mocked, key-free searches.

## [0.1.2] - 2026-03-19

- Added `pyproject.toml` packaging metadata, dynamic package version, and license manifest.
- Expanded the installation and usage documentation for the PyPI release.

## [0.1.0]

### Added

- Initial `nativqa` package and CLI entrypoint.
- Query collection workflow using seed queries and search engines.
- Domain reliability checking script.
- LLM-based annotation helper script.
- Basic unit/integration test scaffold in `tests/`.
