# Building and releasing the pip package

The pip distribution contains the `nativqa` package and console command. Repository
scripts, templates, datasets, and demo apps are maintained separately and are not
installed by pip. Core runtime dependencies are declared in `pyproject.toml`;
`requirements.txt` is for repository workflows and may include additional tools.

## Validate a release

1. Update `nativqa/__init__.py` and document the release in `CHANGELOG.md`.
2. In a fresh virtual environment, install build tools:
   ```bash
   python -m pip install --upgrade pip build twine
   ```
3. Build into a new, empty output directory so old releases cannot be uploaded accidentally:
   ```bash
   python -m build --outdir dist/0.1.3
   python -m twine check --strict dist/0.1.3/*
   ```
4. Install the wheel into a separate clean virtual environment. From outside the
   repository, run `nativqa --help`, `nativqa --version`,
   `python -m nativqa --help`, `python -m nativqa --version`, and `python -m pip check`.
5. Run the offline core tests against the installed package:
   ```bash
   python -I -m unittest discover -s /absolute/path/to/checkout/tests -p test_nativqa.py
   ```
   Run this command outside the checkout so subprocesses also use the installed package.
6. Repeat the installation and checks using the source `.tar.gz` distribution.
7. Wait for the package workflow to pass across its Python and OS matrix before publishing.

The GitHub Actions workflow builds both distributions, checks README/metadata,
checks current and minimum runtime dependencies, and tests installations outside
the source checkout. It uploads build artifacts but does not publish to PyPI.
Tests mock SerpAPI; no API credentials or billable requests are needed.

## Publish

Publish only after reviewing the exact artifacts and setting the release date in
the changelog. Use a new version for each release; the README, keywords, and project
URLs on PyPI are distributed with the package metadata. A GitHub README edit alone
does not update the published PyPI page.

For an automated publishing workflow, configure PyPI Trusted Publishing for this
repository and a protected release environment first:
https://docs.pypi.org/trusted-publishers/

The version history before 0.1.2 is incomplete in this checkout. Do not invent
0.1.1 release notes; reconstruct them from release artifacts or maintainer records.
