"""Create ten reproducible NativQA seeds from the supplied four-column TSV."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "queries/en_synthetic_queries_filtered.tsv"
DESTINATION = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output_dir', type=Path, default=DESTINATION)
    args = parser.parse_args(argv)
    source = args.source.resolve()
    if not source.is_file():
        parser.error('Source TSV not found. Supply --source or use the committed seeds.tsv directly.')
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {'category', 'query'}.issubset(reader.fieldnames or []):
            parser.error('Source TSV must include category and query columns.')
        rows = list(reader)
    # Deduplicate before sampling so the run always contains ten distinct queries.
    unique = {}
    for line_number, row in enumerate(rows, start=2):
        query = (row["query"] or '').strip()
        if query:
            unique.setdefault(query, (line_number, row))
    if len(unique) < 10:
        parser.error('Source TSV must contain at least ten distinct nonempty queries.')
    selected = sorted(random.Random(42).sample(list(unique.values()), 10))
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "seeds.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator='\n')
        writer.writerow(["topic", "query"])
        writer.writerows((row["category"], row["query"].strip()) for _, row in selected)
    manifest = {
        "source": str(source.relative_to(ROOT.parent)) if source.is_relative_to(ROOT.parent) else str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": len(rows),
        "unique_queries": len(unique),
        "random_seed": 42,
        "sample_size": 10,
        "samples": [dict(source_line=line, **row) for line, row in selected],
    }
    (destination / "selection.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(selected)} seeds from {len(rows)} rows: {destination / 'seeds.tsv'}")


if __name__ == "__main__":
    main()
