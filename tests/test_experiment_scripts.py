"""Offline checks for the reproducible experiment entry points and README CLI."""

import contextlib
import csv
import importlib.util
import io
import json
import os
import re
import shlex
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from nativqa.nativqa_framework import main as framework_main


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'experiments/ten_sample_queries' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExperimentScripts(unittest.TestCase):
    def test_seed_selection_is_reproducible_and_retains_source_rows(self):
        module = load_script('prepare_samples')
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.tsv'
            with source.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(['category', 'query'])
                writer.writerows(('travel', f'query {index}') for index in range(15))
                writer.writerow(['travel', 'query 0'])
            with contextlib.redirect_stdout(io.StringIO()):
                module.main(['--source', str(source), '--output_dir', str(root / 'first')])
                module.main(['--source', str(source), '--output_dir', str(root / 'second')])
            self.assertEqual((root / 'first/seeds.tsv').read_bytes(), (root / 'second/seeds.tsv').read_bytes())
            manifest = json.loads((root / 'first/selection.json').read_text())
            self.assertEqual(manifest['unique_queries'], 15)
            self.assertEqual(len(manifest['samples']), 10)
            for row in manifest['samples']:
                self.assertEqual(row['query'], f"query {row['source_line'] - 2}")

    def test_seed_preparation_rejects_too_few_queries_without_output(self):
        module = load_script('prepare_samples')
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.tsv'
            source.write_text('category\tquery\ntravel\tone\n', encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                module.main(['--source', str(source), '--output_dir', str(root / 'out')])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse((root / 'out').exists())

    def test_runner_uses_supplied_key_and_redacts_logs_and_saved_output(self):
        module = load_script('run_samples')
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            seed = root / 'seed.tsv'
            seed.write_text('topic\tquery\ntravel\tDoha\n', encoding='utf-8')
            env = root / 'key.env'
            env.write_text('API_KEY="test-private-token"\n', encoding='utf-8')
            output = root / 'results/run'
            process = Mock(stdout=iter(['request failed?api_key=test-private-token\n']))
            process.wait.return_value = 0
            process.poll.return_value = 0
            def start(command, **kwargs):
                (output / 'response.json').write_text('{"value":"test-private-token"}', encoding='utf-8')
                (output / 'binary.bin').write_bytes(b'\xff')
                self.assertEqual(kwargs['env']['API_KEY'], 'test-private-token')
                self.assertEqual(command[command.index('--engine') + 1], 'bing')
                self.assertEqual(command[command.index('--env') + 1], str(env))
                return process
            with patch.object(module, 'ROOT', root), patch.object(module.subprocess, 'Popen', side_effect=start), \
                 contextlib.redirect_stdout(io.StringIO()) as stdout, self.assertRaises(SystemExit) as raised:
                module.main(['--engine', 'bing', '--env', str(env), '--input_file', str(seed),
                             '--output_dir', 'results/run'])
            self.assertEqual(raised.exception.code, 0)
            for text in (stdout.getvalue(), (output / 'framework.log').read_text(),
                         (output / 'response.json').read_text()):
                self.assertNotIn('test-private-token', text)
                self.assertIn('[REDACTED]', text)

    def test_runner_rejects_invalid_engine_mode_before_launch(self):
        module = load_script('run_samples')
        with patch.object(module.subprocess, 'Popen') as process, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                module.main(['--engine', 'bing', '--qa_extraction', 'hybrid'])
            self.assertEqual(raised.exception.code, 2)
            process.assert_not_called()

    def test_importing_replay_does_not_run_the_pipeline(self):
        with patch('nativqa.nativqa_framework.run_nativqa') as run:
            load_script('replay_updated_qa')
            run.assert_not_called()

    def test_readme_framework_commands_parse_and_use_supported_modes(self):
        commands = []
        for block in re.findall(r'```bash\n(.*?)\n\s*```', (ROOT / 'README.md').read_text(), re.DOTALL):
            for line in block.replace('\\\n', ' ').splitlines():
                tokens = shlex.split(line, comments=True)
                if tokens[:3] == ['python3', '-m', 'nativqa'] and not {'--help', '--version'} & set(tokens):
                    commands.append(tokens[3:])
        self.assertGreaterEqual(len(commands), 7)
        for command in commands:
            with self.subTest(command=command), patch.object(sys, 'argv', ['nativqa', *command]), \
                 patch('nativqa.nativqa_framework.run_nativqa') as run:
                framework_main()
                self.assertEqual(run.call_count, 1)
                if run.call_args.args[0] != 'google' or run.call_args.args[1] != 'text':
                    self.assertEqual(run.call_args.kwargs['qa_extraction'], 'legacy')


if __name__ == '__main__':
    unittest.main()
