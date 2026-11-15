"""Synthetic reflection-memory evaluation. Live Ollama use requires --live.

The dry run validates a frozen fixture without importing the application. Live
cases use fresh temporary stores and never dispatch actor or workspace tools.
Decision matches are structural measurements; memory meaning needs human review.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import tempfile
import time
from urllib.parse import urlsplit
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'evaluations' / 'memory_acceptance' / 'cases.json'
OPTIONAL_INFERENCE_SOURCE = ROOT / 'app' / 'inference.py'
CLASSES = (
    'speaker_ownership', 'user_inference', 'unperformed_activity_proxy',
    'temporal_update', 'semantic_fidelity',
)
DECISIONS = {'create', 'reject', 'supersede', 'pending'}


class EvaluationBudgetError(RuntimeError):
    """A configured inference budget was exhausted before another call."""


def _remaining_timeout(budget: dict, per_call_seconds: int) -> float:
    remaining = budget['max_seconds'] - (time.perf_counter() - budget['started'])
    if remaining <= 0:
        raise EvaluationBudgetError('Evaluation wall-time budget exhausted.')
    return min(per_call_seconds, remaining)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str | None]:
    required = (ROOT / 'app/reflection.py', ROOT / 'app/memory.py',
                ROOT / 'app/experience.py', ROOT / 'app/goals.py',
                ROOT / 'app/config.py', Path(__file__).resolve())
    sources = {path.relative_to(ROOT).as_posix(): _digest(path)
               for path in required}
    sources['app/inference.py'] = (_digest(OPTIONAL_INFERENCE_SOURCE)
                                   if OPTIONAL_INFERENCE_SOURCE.is_file() else None)
    return sources


def _write_report(output: Path, report: dict) -> None:
    output = output.resolve()
    if output.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('Evaluation output cannot be under production data.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    os.replace(temporary, output)


def _render_dialogue(messages: list[dict]) -> str:
    return '\n\n'.join(
        ('ARDA' if message['role'] == 'user' else 'APERTURE') + ': '
        + json.dumps(message['content'], ensure_ascii=False)
        for message in messages[-6:]
    )


def load_fixture(path: Path = FIXTURE) -> dict:
    """Reject malformed or unbalanced gold data before any model call."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('schema_version') != 1 \
            or data.get('synthetic_only') is not True \
            or not isinstance(data.get('cases'), list):
        raise ValueError('Invalid synthetic memory evaluation fixture.')
    cases = data['cases']
    if len(cases) != 30:
        raise ValueError('Memory evaluation requires exactly 30 cases.')
    ids = set()
    cells = Counter()
    decisions = Counter()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get('id'), str) \
                or not case['id'] or case['id'] in ids:
            raise ValueError('Each case needs a distinct nonempty ID.')
        ids.add(case['id'])
        split = case.get('split')
        failure_class = case.get('failure_class')
        if split not in {'dev', 'holdout'} or failure_class not in CLASSES:
            raise ValueError(f"Invalid split/class in {case['id']}.")
        cells[(split, failure_class)] += 1
        messages = case.get('messages')
        if not isinstance(messages, list) or not messages \
                or any(not isinstance(message, dict)
                       or set(message) != {'role', 'content'}
                       or message['role'] not in {'user', 'assistant'}
                       or not isinstance(message['content'], str)
                       or not message['content'].strip()
                       for message in messages) \
                or messages[-1]['role'] != 'user' \
                or len(_render_dialogue(messages)) < 80:
            raise ValueError(f"Invalid reflection dialogue in {case['id']}.")
        seed = case.get('seed_memory')
        seed_observation = case.get('seed_observation')
        if (seed is None) != (seed_observation is None) \
                or (seed is not None and (
                    not isinstance(seed_observation, str)
                    or not seed_observation.strip())):
            raise ValueError(f"Seed memory needs its observed user source in {case['id']}.")
        if seed is not None and (not isinstance(seed, dict)
                or set(seed) != {'content', 'category', 'subject'}
                or seed['subject'] != 'user'
                or any(not isinstance(seed[key], str) or not seed[key].strip()
                       for key in seed)):
            raise ValueError(f"Invalid synthetic seed in {case['id']}.")
        expected = case.get('expected')
        if not isinstance(expected, dict) or expected.get('decision') not in DECISIONS:
            raise ValueError(f"Invalid expected decision in {case['id']}.")
        if expected['decision'] == 'supersede' and seed is None:
            raise ValueError(f"Supersession needs a seed memory in {case['id']}.")
        decisions[(split, expected['decision'])] += 1
        if not isinstance(case.get('review_criterion'), str) \
                or not case['review_criterion'].strip():
            raise ValueError(f"Missing human criterion in {case['id']}.")
    for failure_class in CLASSES:
        if cells[('dev', failure_class)] != 2 \
                or cells[('holdout', failure_class)] != 4:
            raise ValueError(f'Unbalanced fixture class: {failure_class}.')
    for split in ('dev', 'holdout'):
        if decisions[(split, 'reject')] == 0 \
                or decisions[(split, 'create')] + decisions[(split, 'supersede')] == 0:
            raise ValueError(f'{split} requires positive saves and true negatives.')
    return data


@contextmanager
def _isolated_stores(directory: Path):
    """Bind every application store to one synthetic root and guard SQLite."""
    directory = Path(directory).resolve()
    if directory.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('Synthetic stores cannot be under production data.')
    directory.mkdir(parents=True, exist_ok=True)
    original_connect = sqlite3.connect
    original_mkdir = Path.mkdir

    def guarded_connect(database, *args, **kwargs):
        value = os.fsdecode(os.fspath(database))
        if value != ':memory:':
            if value.startswith('file:') or kwargs.get('uri') \
                    or not Path(value).resolve().is_relative_to(directory):
                raise AssertionError('Evaluation SQLite access outside its temporary store.')
        return original_connect(database, *args, **kwargs)

    def guarded_mkdir(path, *args, **kwargs):
        if path.resolve().is_relative_to((ROOT / 'data').resolve()):
            raise AssertionError('Evaluation attempted production data initialization.')
        return original_mkdir(path, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(sqlite3, 'connect', guarded_connect))
        stack.enter_context(patch.object(sqlite3.dbapi2, 'connect', guarded_connect))
        stack.enter_context(patch.object(Path, 'mkdir', guarded_mkdir))
        sys.path.insert(0, str(ROOT / 'app'))
        stack.callback(sys.path.remove, str(ROOT / 'app'))
        memory = importlib.import_module('memory')
        experience = importlib.import_module('experience')
        goals = importlib.import_module('goals')
        reflection = importlib.import_module('reflection')
        memories = memory.MemoryStore(directory / 'memory.db')
        events = experience.ExperienceStore(directory / 'experience.db')
        goal_store = goals.GoalStore(directory / 'goals.db')
        stack.enter_context(patch.object(memory, '_STORE', memories))
        stack.enter_context(patch.object(experience, '_STORE', events))
        stack.enter_context(patch.object(goals, '_STORE', goal_store))
        stack.enter_context(patch.object(experience, '_CURRENT_EPISODE_ID', None))
        stack.enter_context(patch.object(reflection, '_casual_turns_since_reflection', 0))
        yield memory, experience, reflection, memories, events


def _response_json(response):
    if hasattr(response, 'model_dump'):
        return response.model_dump(mode='json')
    message = getattr(response, 'message', None)
    return {'message': {'content': getattr(message, 'content', None)},
            'done_reason': getattr(response, 'done_reason', None)}


def _public_memory(record: dict) -> dict:
    return {key: record.get(key) for key in (
        'id', 'content', 'category', 'importance', 'source', 'subject',
        'evidence_event_ids', 'active', 'valid_from', 'valid_until',
        'supersedes_memory_id',
    )}


def run_case(case: dict, directory: Path, *, model_call,
             budget: dict, on_progress=None) -> dict:
    """Analyze once, then apply that result through the real reflection path."""
    result = {
        'id': case['id'], 'split': case['split'],
        'failure_class': case['failure_class'],
        'expected': case['expected'], 'review_criterion': case['review_criterion'],
        'semantic_verdict': 'unreviewed', 'model_calls': [], 'error': None,
        'observed_decision': None,
    }
    started = time.perf_counter()
    with _isolated_stores(directory) as (_, experience, reflection, memories, _):
        seed_id = None
        seed = case.get('seed_memory')
        if seed is not None:
            experience.start_episode(metadata={'synthetic_memory_evaluation': True,
                                               'phase': 'seed'})
            turn = experience.start_turn(case['seed_observation'])
            source = next(event for event in experience.get_turn_events(turn)
                          if event['event_type'] == 'user_message')
            result['seed_observation_event'] = {key: source.get(key) for key in (
                'id', 'episode_id', 'turn_id', 'event_type', 'event_class',
                'actor', 'content')}
            seed_id = memories.remember(**seed, source='synthetic_fixture',
                                        evidence_event_ids=[source['id']])['id']
            experience.end_episode(reason='synthetic_seed_complete')
        current_episode = experience.start_episode(
            metadata={'synthetic_memory_evaluation': True, 'case_id': case['id']})
        last_turn = None
        for message in case['messages']:
            if message['role'] == 'user':
                last_turn = experience.start_turn(message['content'])
            else:
                if last_turn is None:
                    experience.record_event(
                        event_type='assistant_message', event_class='observed',
                        actor='aperture', content=message['content'])
                else:
                    experience.record_assistant_message(last_turn, message['content'])
        before = {record['id']: record for record in memories.recent_records(limit=100)}
        dialogue = reflection._recent_dialogue(case['messages'], limit=6)
        existing = reflection.build_memory_validation_context(limit=30)
        original_model_call = model_call

        def counted_call(*args, **kwargs):
            if budget['calls'] >= budget['max_calls'] \
                    or time.perf_counter() - budget['started'] >= budget['max_seconds']:
                raise EvaluationBudgetError('Evaluation inference budget exhausted.')
            budget['calls'] += 1  # Count attempted calls, including failures.
            call = {'request': {'args': list(args), **kwargs}, 'response': None,
                    'error': None, 'latency_seconds': None}
            result['model_calls'].append(call)
            if on_progress is not None:
                on_progress(result)
            call_started = time.perf_counter()
            try:
                response = original_model_call(*args, **kwargs)
                call['response'] = _response_json(response)
                return response
            except Exception as exc:
                call['error'] = type(exc).__name__
                raise
            finally:
                call['latency_seconds'] = round(time.perf_counter() - call_started, 4)
                if on_progress is not None:
                    on_progress(result)

        try:
            with patch.object(reflection, 'ollama_chat', counted_call):
                analysis_started = time.perf_counter()
                analysis = reflection.analyze_reflection_debug(
                    dialogue, existing, messages=case['messages'])
                result['analysis_seconds'] = round(
                    time.perf_counter() - analysis_started, 4)
            result['analysis'] = analysis
            with patch.object(reflection, 'REFLECTION_INTERVAL', 1), \
                    patch.object(reflection, 'analyze_reflection_debug', return_value=analysis):
                result['reflection_result'] = reflection.maybe_reflect(
                    case['messages'], used_action_tool=False,
                    memory_operation_used=False, turn_id=last_turn)
        except Exception as exc:
            result['error'] = type(exc).__name__
        after = {record['id']: record for record in memories.recent_records(limit=100)}
        result['before_memories'] = [_public_memory(record) for record in before.values()]
        result['after_memories'] = [_public_memory(record) for record in after.values()]
        result['pending'] = [{key: proposal.get(key) for key in (
            'id', 'content', 'category', 'importance', 'evidence_event_ids',
            'source_episode_id', 'supersedes_memory_id', 'status')}
            for proposal in memories.list_pending_user_memories(limit=100)]
        observed_events = [event for event in
                           experience.get_recent_observed_message_events(limit=100)
                           if event.get('episode_id') == current_episode]
        result['observed_events'] = [{key: event.get(key) for key in (
            'id', 'episode_id', 'turn_id', 'event_type', 'event_class',
            'actor', 'content')} for event in observed_events]
        current_user_events = {
            event['id']: event for event in observed_events
            if event.get('episode_id') == current_episode
            and event.get('event_type') == 'user_message'
            and event.get('actor') == 'arda'
        }
        new_records = [record for memory_id, record in after.items()
                       if memory_id not in before]
        new_active_user = [record for record in new_records
                           if record['subject'] == 'user' and record['active']]
        seed_after = after.get(seed_id) if seed_id is not None else None
        changed_existing = any(after.get(memory_id) != record
                               for memory_id, record in before.items())
        if result['pending']:
            observed = 'pending'
        elif new_active_user and seed_after is not None and not seed_after['active']:
            observed = 'supersede'
        elif new_active_user:
            observed = 'create'
        elif changed_existing or new_records:
            observed = 'other_write'
        else:
            observed = 'reject'
        result['observed_decision'] = observed
        source_records = new_active_user + result['pending']
        result['provenance_valid'] = all(
            bool(record['evidence_event_ids'])
            and all(event_id in current_user_events
                    and current_user_events[event_id].get('event_class') == 'observed'
                    for event_id in record['evidence_event_ids'])
            for record in source_records)
        result['lineage_valid'] = (observed != 'supersede' or (
            seed_after is not None and not seed_after['active']
            and len(new_active_user) == 1
            and new_active_user[0]['supersedes_memory_id'] == seed_id
            and new_active_user[0]['valid_from'] == seed_after['valid_until']))
        result['unexpected_owner_write'] = any(record['subject'] != 'user'
                                               for record in new_records)
        result['decision_match'] = (result['error'] is None
                                    and observed == case['expected']['decision']
                                    and result['provenance_valid']
                                    and result['lineage_valid']
                                    and not result['unexpected_owner_write'])
        experience.end_episode(reason='synthetic_evaluation_complete')
    result['total_seconds'] = round(time.perf_counter() - started, 4)
    return result


def summarize(results: list[dict]) -> dict:
    """Separate positive capture from negative rejection, never grade wording."""
    summary = {}
    for split in ('dev', 'holdout'):
        subset = [row for row in results if row['split'] == split]
        expected = Counter(row['expected']['decision'] for row in subset)
        observed = Counter(row['observed_decision'] for row in subset)
        matched = Counter(row['expected']['decision'] for row in subset
                          if row.get('decision_match'))
        timings = [row['total_seconds'] for row in subset
                   if row.get('total_seconds') is not None]
        summary[split] = {
            'evaluated': len(subset), 'expected': dict(expected),
            'observed': dict(observed), 'decision_matches': dict(matched),
            'create_capture': (matched['create'] / expected['create']
                               if expected['create'] else None),
            'reject_specificity': (matched['reject'] / expected['reject']
                                   if expected['reject'] else None),
            'pending_capture': (matched['pending'] / expected['pending']
                                if expected['pending'] else None),
            'supersession_capture': (matched['supersede'] / expected['supersede']
                                     if expected['supersede'] else None),
            'false_active_writes': sum(
                row['expected']['decision'] in {'reject', 'pending'}
                and row['observed_decision'] in {'create', 'supersede', 'other_write'}
                for row in subset),
            'errors': sum(bool(row.get('error')) for row in subset),
            'provenance_failures': sum(row.get('provenance_valid') is False
                                       for row in subset),
            'median_total_seconds': (statistics.median(timings) if timings else None),
            'semantic_reviewed': 0,
        }
    return summary


def _localhost_host(raw: str) -> str:
    parsed = urlsplit(raw)
    if parsed.scheme not in {'http', 'https'} or parsed.hostname not in {
            '127.0.0.1', 'localhost', '::1'} \
            or parsed.username or parsed.password or parsed.path not in {'', '/'} \
            or parsed.query or parsed.fragment:
        raise ValueError('Live evaluation requires a local Ollama endpoint.')
    return raw


def evaluate(output: Path, *, fixture: Path = FIXTURE, live: bool = False,
             split: str = 'all', max_cases: int = 30, max_calls: int = 80,
             timeout_seconds: int = 120, max_seconds: int = 1800,
             model_call=None, temporary_factory=None,
             overwrite: bool = False) -> dict:
    if split not in {'all', 'dev', 'holdout'}:
        raise ValueError('Split must be all, dev, or holdout.')
    if not 1 <= max_cases <= 30 or not 1 <= max_calls <= 120 \
            or not 1 <= timeout_seconds <= 300 or not 1 <= max_seconds <= 7200:
        raise ValueError('Bounds: cases 1..30, calls 1..120, timeout 1..300s, total 1..7200s.')
    fixture = Path(fixture).resolve()
    data = load_fixture(fixture)
    selected = [case for case in data['cases']
                if split == 'all' or case['split'] == split][:max_cases]
    output = Path(output).resolve()
    if output.is_relative_to((ROOT / 'data').resolve()) or output == fixture:
        raise ValueError('Evaluation output cannot overwrite fixture or production data.')
    if output.exists() and not overwrite:
        raise FileExistsError('Evaluation output already exists; use --overwrite deliberately.')
    sources = _source_hashes()
    report = {
        'schema_version': 1, 'synthetic_only': True, 'live': live,
        'selected_split': split,
        'fixture_sha256': _digest(fixture), 'source_sha256': sources,
        'model': None, 'model_metadata': None, 'model_settings': None,
        'scope': 'reflection extraction, validation and one memory write pass; no actor tools',
        'semantic_status': 'unreviewed; raw outputs require human review',
        'limits': {'max_cases': max_cases, 'max_calls': max_calls,
                   'timeout_seconds_per_call': timeout_seconds,
                   'max_seconds_pre_call_budget': max_seconds},
        'timeout_semantics': ('The wall-time budget is checked before each model call. '
                              'The Ollama HTTP timeout applies per network operation; '
                              'a slow response may exceed the stated wall time.'),
        'cases': [], 'metrics': None, 'errors': [],
    }
    if not live:
        report['prepared_cases'] = [{key: case[key] for key in (
            'id', 'split', 'failure_class', 'expected', 'review_criterion')}
            for case in selected]
        _write_report(output, report)
        return report

    scratch = ROOT / '.reliability-tmp'
    scratch.mkdir(exist_ok=True)
    budget = {'calls': 0, 'max_calls': max_calls,
              'started': time.perf_counter(), 'max_seconds': max_seconds}
    factory = temporary_factory or (lambda: tempfile.TemporaryDirectory(
        prefix='aperture-memory-eval-', dir=scratch))
    # Even the model identity is obtained only after the store guards are active.
    with factory() as bootstrap_directory:
        with _isolated_stores(Path(bootstrap_directory)) as (_, _, reflection, _, _):
            report['model'] = reflection.MODEL
            report['model_settings'] = {
                'memory_options': dict(reflection.MEMORY_OPTIONS),
                'timeout_seconds_per_call': timeout_seconds,
            }
    try:
        if model_call is None:
            import ollama
            host = _localhost_host(os.environ.get('APERTURE_OLLAMA_HOST',
                                                  os.environ.get('OLLAMA_HOST',
                                                                 'http://127.0.0.1:11434')))
            report['model_settings']['endpoint'] = host
            client = ollama.Client(
                host=host, timeout=_remaining_timeout(budget, timeout_seconds))
            def bounded_chat(*args, **kwargs):
                return ollama.Client(
                    host=host, timeout=_remaining_timeout(budget, timeout_seconds)
                ).chat(*args, **kwargs)
            model_call = bounded_chat
            try:
                models = client.list().model_dump(mode='json').get('models', [])
                report['model_metadata'] = next((entry for entry in models
                    if entry.get('model') == report['model']
                    or entry.get('name') == report['model']), None)
            except Exception as exc:
                report['model_metadata_error'] = type(exc).__name__
        report['model_calls_attempted'] = 0
        _write_report(output, report)
        for case in selected:
            if budget['calls'] >= max_calls \
                    or time.perf_counter() - budget['started'] >= max_seconds:
                report['errors'].append('EvaluationBudgetError')
                break
            entry = {'id': case['id'], 'status': 'running',
                     'expected': case['expected'], 'model_calls': []}
            report['cases'].append(entry)
            _write_report(output, report)

            def progress(partial):
                entry.update(partial)
                entry['status'] = 'running'
                report['model_calls_attempted'] = budget['calls']
                _write_report(output, report)

            with factory() as directory:
                entry.update(run_case(case, Path(directory), model_call=model_call,
                                      budget=budget, on_progress=progress))
            entry['status'] = 'complete'
            if entry.get('error'):
                report['errors'].append({'case_id': case['id'],
                                         'type': entry['error']})
            report['model_calls_attempted'] = budget['calls']
            report['metrics'] = summarize([row for row in report['cases']
                                           if row.get('status') == 'complete'])
            _write_report(output, report)
        report['complete'] = len(report['cases']) == len(selected) \
            and all(row.get('status') == 'complete' and not row.get('error')
                    for row in report['cases'])
        _write_report(output, report)
        return report
    except Exception:
        _write_report(output, report)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--split', choices=('all', 'dev', 'holdout'), default='all')
    parser.add_argument('--max-cases', type=int, default=30)
    parser.add_argument('--max-calls', type=int, default=80)
    parser.add_argument('--timeout-seconds', type=int, default=120)
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / '.reliability-tmp').resolve()):
        parser.error('CLI output must be under .reliability-tmp.')
    report = evaluate(args.output, live=args.live, split=args.split,
                      max_cases=args.max_cases,
                      max_calls=args.max_calls,
                      timeout_seconds=args.timeout_seconds,
                      max_seconds=args.max_seconds, overwrite=args.overwrite)
    print(json.dumps({'live': report['live'],
                      'prepared': len(report.get('prepared_cases', [])),
                      'completed': sum(row.get('status') == 'complete'
                                       for row in report['cases']),
                      'model_calls_attempted': report.get('model_calls_attempted', 0),
                      'metrics': report['metrics']}, ensure_ascii=False))
    return 0 if not args.live or report.get('complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
