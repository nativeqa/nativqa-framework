# Contributing to NativQA Framework

Thank you for contributing to NativQA Framework.

## Ways to Contribute

- Report bugs and unexpected behavior.
- Propose new features or improvements.
- Improve code, tests, documentation, and examples.
- Help with multilingual seed/query templates and validation resources.

## Development Setup

1. Fork and clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:
   - `pip install -r requirements.txt`
   - `pip install -e .`

## Running the Project

Example command:

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

## Running Tests

Run the offline core package tests (no API key required):

```bash
python3 -m unittest tests.test_nativqa
```

Run all repository tests, including optional utility tests:

```bash
python3 -m unittest discover -s tests
```

Notes:

- Some tests execute scraping logic and may require internet access.
- Tests that hit APIs require a valid key (for example in `tests/envs/api_key.env`).
- If a test depends on external services, document that in your merge request.

## Contribution Workflow

1. Create a branch from `main`.
2. Make focused changes (one logical change per branch).
3. Add or update tests when behavior changes.
4. Update docs (`README.md`, this file, or related docs) when needed.
5. Open a merge request with a clear description.

## Merge Request Checklist

- [ ] The change is scoped and focused.
- [ ] Tests are added/updated for behavior changes.
- [ ] Existing tests pass locally (or limitations are documented).
- [ ] Documentation is updated where relevant.
- [ ] No secrets, keys, or sensitive data are committed.

## Coding Guidelines

- Follow the existing style and structure in the repository.
- Prefer small, composable functions over large monolithic logic.
- Keep public CLI behavior backward-compatible when possible.
- If breaking changes are required, clearly document them.

## Reporting Bugs

Please include:

- Environment details (OS, Python version).
- Exact command used.
- Input file sample (minimal reproducible).
- Expected vs actual behavior.
- Stack trace or error logs.

## Release Notes and Changelog

User-visible changes should be reflected in `CHANGELOG.md` under `Unreleased`.

See [PACKAGING.md](PACKAGING.md) for distribution validation and release preparation.
