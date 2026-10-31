"""Run the stock NativQA CLI, redacting the credential from logs and results."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, quote_plus

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/ten_sample_queries"
KEY_FILE = ROOT.parent / "envs/api_key.txt"
if not KEY_FILE.is_file():
    KEY_FILE = ROOT / 'envs/api_key.env'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('google', 'bing', 'yahoo'), default='google')
    parser.add_argument('--qa_extraction', choices=('legacy', 'hybrid', 'ai_overview'), default='legacy')
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--ai_overview_max_requests', type=int, default=5)
    parser.add_argument('--env', type=Path, default=KEY_FILE)
    parser.add_argument('--input_file', type=Path, default=ROOT / 'experiments/ten_sample_queries/seeds.tsv')
    args = parser.parse_args(argv)
    if args.engine != 'google' and args.qa_extraction != 'legacy':
        parser.error('Bing and Yahoo require --qa_extraction legacy')
    if args.ai_overview_max_requests < 0:
        parser.error('--ai_overview_max_requests must be nonnegative')
    default_output = OUTPUT if args.engine == 'google' else OUTPUT.with_name(f'{OUTPUT.name}_{args.engine}')
    output = (ROOT / (args.output_dir or default_output)).resolve()
    if not output.is_relative_to(ROOT / 'results'):
        parser.error('--output_dir must be inside this checkout\'s results directory')
    key_file = (ROOT / args.env).resolve()
    input_file = (ROOT / args.input_file).resolve()
    if not input_file.is_file():
        parser.error('--input_file must name an existing CSV or TSV file')
    key = dotenv_values(key_file).get("API_KEY")
    if not key:
        parser.error('--env must name a dotenv file defining API_KEY')
    secrets = {key, quote(key, safe=""), quote_plus(key)}

    def redact(text):
        for secret in secrets:
            text = text.replace(secret, "[REDACTED]")
        return text

    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "nativqa",
        "--engine", args.engine, "--search_type", "text",
        "--input_file", str(input_file),
        "--country_code", "qa", "--location", "Doha, Qatar",
        "--env", str(key_file),
        "--output_dir", str(output), "--n_iter", "1",
        "--qa_extraction", args.qa_extraction,
        "--ai_overview_max_requests", str(args.ai_overview_max_requests),
    ]
    # Honor the requested file even if the shell already has an API_KEY set.
    environment = dict(os.environ, API_KEY=key, PYTHONUNBUFFERED="1")
    print(shlex.join(command), flush=True)
    with (output / "framework.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        )
        try:
            for line in process.stdout:
                line = redact(line)
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for path in output.rglob("*"):
                if path.is_file():
                    try:
                        original = path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        continue
                    sanitized = redact(original)
                    if original != sanitized:
                        path.write_text(sanitized, encoding="utf-8")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
