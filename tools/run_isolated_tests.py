"""Run unittest discovery with fail-closed storage/inference guards and JSON evidence."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default='test_*.py')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    for name in ('APERTURE_DYNAMIC_SELF', 'APERTURE_RELATIONSHIP', 'APERTURE_SELF_CONSOLIDATION'):
        os.environ[name] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    # The managed Windows host's default temporary directory may be unwritable.
    # Select a new scratch parent inside this authorized workspace, not more privileges.
    scratch = ROOT / '.reliability-tmp'
    scratch.mkdir(exist_ok=True)
    tempfile.tempdir = str(scratch)
    sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'app')]
    import isolation_support as safety
    safety.install_guards()  # Before any application imports.
    started = time.perf_counter()
    with safety.TestTemporaryDirectory(prefix='aperture-day1-tests-') as directory, ExitStack() as stack:
        # All default lazy stores resolve to newly created synthetic storage.
        # Constructor defaults remain guard sentinels for explicit isolation tests.
        for name, constructor, filename in (
            ('memory', 'MemoryStore', 'memory.db'),
            ('experience', 'ExperienceStore', 'experience.db'),
            ('goals', 'GoalStore', 'goals.db'),
        ):
            module = __import__(name)  # Audited: imports do not initialize stores.
            stack.enter_context(patch.object(module, '_STORE', getattr(module, constructor)(Path(directory) / filename)))
        import experience
        stack.enter_context(patch.object(experience, '_CURRENT_EPISODE_ID', None))
        loader = unittest.TestLoader()
        suite = loader.discover(str(ROOT / 'tests'), pattern=args.pattern)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report = {
            'command': [sys.executable, '-B', *sys.argv],
            'pattern': args.pattern, 'synthetic_only': True,
            'isolation': 'registered temporary SQLite roots; all three lazy stores redirected; production mkdir/SQLite and unmocked Ollama blocked',
            'bytecode_disabled': sys.dont_write_bytecode,
            'environment': {k: os.environ.get(k) for k in ('PYTHONDONTWRITEBYTECODE',)},
            'duration_seconds': time.perf_counter() - started,
            'tests_run': result.testsRun,
            'passed': result.testsRun - len({t.id() for t, _ in result.failures + result.errors}) - len(result.skipped) - len(result.expectedFailures) - len(result.unexpectedSuccesses),
            'failures': [{'test': t.id(), 'traceback': tb} for t, tb in result.failures],
            'errors': [{'test': t.id(), 'traceback': tb} for t, tb in result.errors],
            'skipped': [{'test': t.id(), 'reason': reason} for t, reason in result.skipped],
            'expected_failures': len(result.expectedFailures),
            'unexpected_successes': len(result.unexpectedSuccesses),
            'guard_violations': list(safety._VIOLATIONS),
            'exit_code': 0 if result.wasSuccessful() and not safety._VIOLATIONS else 1,
        }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('tests_run', 'passed', 'duration_seconds', 'exit_code', 'guard_violations')}))
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
