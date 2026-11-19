"""Measure three production retrieval paths with frozen synthetic fixtures."""
from __future__ import annotations

import argparse
from contextlib import closing, nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'evaluations' / 'retrieval'
SCRATCH = ROOT / '.reliability-tmp'
ENDPOINT = 'http://127.0.0.1:11434'
TIMEOUT_SECONDS = 45
TOOL_RECORD = re.compile(r'^\[(m[0-9]{11})\] \[([^\]]+)\]', re.MULTILINE)
ACTOR_RECORD = re.compile(r'^- id=(m[0-9]{11}) \[([^\]]+)\]', re.MULTILINE)


def canonical_hash(data):
    text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def seed_store(memory, store, fixture):
    by_key = {}
    inactive = set()
    for index, item in enumerate(fixture['records']):
        timestamp = f'2026-01-01T00:{index // 60:02d}:{index % 60:02d}+00:00'
        kwargs = {key: item[key] for key in ('content', 'category', 'importance', 'subject')}
        with patch.object(memory.uuid, 'uuid4', return_value=SimpleNamespace(hex=item['id'])), \
                patch.object(memory, '_now', return_value=timestamp):
            if item.get('supersedes'):
                prior = by_key[item['supersedes']]
                record = store.supersede(prior, source='synthetic_retrieval', **kwargs)
                inactive.add(prior)
            else:
                record = store.remember(source='synthetic_retrieval', **kwargs)
            if record['id'] != item['id']:
                raise AssertionError('Synthetic fixture ID was not preserved.')
            if item.get('forgotten'):
                if not store.forget(record['id']):
                    raise AssertionError('Could not forget the synthetic record.')
                inactive.add(record['id'])
        by_key[item['key']] = record['id']
    return inactive


def measure_case(memory, case, path, subjects, inactive, model_selector=None):
    row = dict(case)
    row.update(candidate_ids=None, selected_ids=None, selector_error=None)

    def select(**request):
        candidates = [record['id'] for record in request['candidates']]
        row['candidate_ids'] = candidates
        if path == 'actor_fallback':
            raise RuntimeError('Synthetic selector-unavailable probe.')
        if model_selector is None:
            selected = [key for key in case['gold_ids'] if key in candidates][:request['limit']]
        else:
            try:
                selected = model_selector(**request)
            except Exception as error:
                row['selector_error'] = type(error).__name__
                raise
        row['selected_ids'] = selected
        return selected

    if path == 'lexical_tool':
        result = memory.search_memory(case['query'], limit=5)
        if result.startswith('MEMORY_ERROR:'):
            raise RuntimeError('Synthetic lexical tool search failed.')
        records = TOOL_RECORD.findall(result)
    else:
        result = memory.build_actor_memory_context(
            case['query'], recent_dialogue=case['recent_dialogue'], selector=select,
        )
        records = ACTOR_RECORD.findall(result)
    returned = [key for key, _ in records]
    if path == 'actor_selector' and row['selector_error'] is None and row['selected_ids'] != returned:
        row['selector_error'] = 'SelectionRejected'
    row.update(
        returned_ids=returned,
        returned_subjects=dict(records),
        inactive_ids_returned=[key for key in returned if key in inactive],
        ownership_tag_violations=[key for key, subject in records if subjects.get(key) != subject],
        other_subject_ids=[key for key, subject in records
                           if case.get('expected_subject') and case['expected_subject'] != subject],
    )
    return row


def rate(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                rate=numerator / denominator if denominator else None)


def summarize(rows):
    measured = [row for row in rows if row['selector_error'] is None]
    answerable = [row for row in measured if row['gold_ids']]
    unanswered = [row for row in measured if not row['gold_ids']]
    with_candidates = [row for row in rows if row['candidate_ids'] is not None and row['gold_ids']]
    corrections = [row for row in measured if row['kind'] == 'correction']

    def gold_fraction(row, available):
        gold = set(row['gold_ids'])
        return len(gold.intersection(available)) / len(gold)

    return dict(
        cases=len(rows),
        completed_measurements=len(measured),
        selector_errors=len(rows) - len(measured),
        recall_at_5=rate(sum(gold_fraction(row, row['returned_ids'][:5]) for row in answerable),
                         len(answerable)),
        mrr_at_5=rate(sum(next((1 / rank for rank, key in enumerate(row['returned_ids'][:5], 1)
                               if key in row['gold_ids']), 0) for row in answerable), len(answerable)),
        no_answer_false_positive_rate=rate(sum(bool(row['returned_ids']) for row in unanswered),
                                           len(unanswered)),
        candidate_gold_coverage=rate(sum(gold_fraction(row, row['candidate_ids'])
                                         for row in with_candidates), len(with_candidates)),
        current_version_accuracy=rate(sum(set(row['gold_ids']).issubset(row['returned_ids'][:5])
                                          and not row['inactive_ids_returned'] for row in corrections),
                                      len(corrections)),
        inactive_returns=sum(len(row['inactive_ids_returned']) for row in rows),
        ownership_tag_violations=sum(len(row['ownership_tag_violations']) for row in rows),
        other_subject_returns=sum(len(row['other_subject_ids']) for row in rows),
    )


def installed_digest(client, model):
    name = model if ':' in model.rsplit('/', 1)[-1] else f'{model}:latest'
    for installed in client.list().models:
        if installed.model == name and installed.digest:
            return installed.digest
    raise ValueError('The requested model has no installed digest on local Ollama.')


def evaluate(output, *, live=False, model=None, group='all'):
    output = Path(output).resolve()
    if output.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('Evaluation output cannot be under production data.')
    if output.exists():
        raise FileExistsError(f'Choose a new evaluation report path: {output}')
    fixture_paths = [FIXTURES / name for name in ('baseline.json', 'fresh.json')]
    baseline, fresh = fixtures = [json.loads(path.read_text(encoding='utf-8')) for path in fixture_paths]
    groups = {
        'historical': [case for case in baseline['cases'] if case['split'] == 'held_out'],
        'fresh': fresh['cases'],
    }
    if group != 'all':
        groups = {group: groups[group]}
    sys.path[:0] = [path for path in (str(ROOT / 'tests'), str(ROOT / 'app')) if path not in sys.path]
    import isolation_support as safety
    import ollama

    if live and safety._INSTALLED:
        raise RuntimeError('Run --live in a standalone evaluator process.')
    sdk_chat = ollama.Client.chat
    safety.install_guards()
    SCRATCH.mkdir(exist_ok=True)

    import memory

    report = {
        'schema_version': 1,
        'synthetic_only': True,
        'python_version': sys.version,
        'fixture_sha256': {path.relative_to(ROOT).as_posix(): canonical_hash(fixture)
                           for path, fixture in zip(fixture_paths, fixtures)},
        'fixture_hash_format': 'UTF-8 JSON with original key order, two-space indentation, and final LF',
        'source_sha256': {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('app/memory.py', 'app/llm.py', 'app/config.py', 'app/context_budget.py',
                         'app/inference.py', 'requirements.txt', 'tests/isolation_support.py',
                         'tools/retrieval_eval.py')
        },
        'actor_selector_mode': 'live' if live else 'gold_id_mock',
        'model': None,
        'model_calls': 0,
        'groups': {},
    }
    subjects = {item['id']: item['subject'] for item in baseline['records']}
    max_calls = sum(map(len, groups.values()))

    client_context = nullcontext()
    if live:
        # The SDK otherwise adds its environment API key even with HTTPX trust_env disabled.
        with patch.dict(os.environ):
            os.environ.pop('OLLAMA_API_KEY', None)
            client = ollama.Client(host=ENDPOINT, timeout=TIMEOUT_SECONDS,
                                   follow_redirects=False, trust_env=False)
        client_context = closing(client)
    with client_context as client, safety.TestTemporaryDirectory(dir=SCRATCH, prefix='aperture-retrieval-') as folder:
        store = memory.MemoryStore(Path(folder) / 'memory.db')
        inactive = seed_store(memory, store, baseline)
        model_patch = nullcontext()
        model_selector = None
        if live:
            import llm
            from config import DEFAULT_MODEL
            from inference import _validate_response

            model = model or DEFAULT_MODEL
            report['model'] = {
                'name': model, 'digest': installed_digest(client, model), 'endpoint': ENDPOINT,
                'timeout_seconds': TIMEOUT_SECONDS, 'retries': 0, 'max_calls': max_calls,
            }

            def chat(*, role, **request):
                if role != 'retrieval' or request.pop('model') != model or report['model_calls'] >= max_calls:
                    raise RuntimeError('Unexpected inference outside the frozen retrieval run.')
                report['model_calls'] += 1
                report['model']['request_settings'] = {
                    key: request[key] for key in ('think', 'format', 'options')
                }
                return _validate_response(sdk_chat(client, model=model, **request), role)

            model_patch = patch.multiple(llm, ollama_chat=chat, RETRIEVAL_MODEL=model)
            model_selector = llm._select_memory_ids

        with patch.object(memory, '_STORE', store), model_patch:
            for split, cases in groups.items():
                report['groups'][split] = {}
                for path in ('lexical_tool', 'actor_fallback', 'actor_selector'):
                    rows = [measure_case(memory, case, path, subjects, inactive, model_selector)
                            for case in cases]
                    report['groups'][split][path] = dict(metrics=summarize(rows), results=rows)
        if live and installed_digest(client, model) != report['model']['digest']:
            raise RuntimeError('The local model digest changed during the evaluation.')

    report['guard_violations'] = list(safety._VIOLATIONS)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SCRATCH / 'retrieval-report.json')
    parser.add_argument('--group', choices=('all', 'historical', 'fresh'), default='all')
    parser.add_argument('--live', action='store_true', help='Use an already installed model on local Ollama.')
    parser.add_argument('--model', help='Installed model for --live (default: source default).')
    args = parser.parse_args()
    if args.model and not args.live:
        parser.error('--model requires --live')
    report = evaluate(args.output, live=args.live, model=args.model, group=args.group)
    print(json.dumps({
        'report': str(args.output),
        'actor_selector_mode': report['actor_selector_mode'],
        'model_calls': report['model_calls'],
        'groups': {split: {path: item['metrics'] for path, item in paths.items()}
                   for split, paths in report['groups'].items()},
    }, indent=2))
    return int(bool(report['guard_violations']) or any(
        paths['actor_selector']['metrics']['selector_errors'] for paths in report['groups'].values()
    ))


if __name__ == '__main__':
    raise SystemExit(main())
