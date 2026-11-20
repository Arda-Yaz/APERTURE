"""Measure the synchronous CLI turn on isolated synthetic conversations."""
import argparse
from contextlib import ExitStack, closing, nullcontext, redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

from memory_acceptance_eval import _isolated_stores, _write_report
from retrieval_eval import ENDPOINT, TIMEOUT_SECONDS, canonical_hash, installed_digest, seed_store


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / '.reliability-tmp'
CORPUS = ROOT / 'evaluations/retrieval/baseline.json'
PREFERENCES = (
    'For software documentation I prefer short examples and readable tables, except when commands require extra context.',
    'I usually choose an afternoon train for long trips because I can avoid the morning crowds.',
    'I prefer leaving English function names intact in Turkish technical explanations so I can search for them easily.',
    'When reviewing drafts, I usually prefer high-level feedback before detailed line edits.',
)
FILE_QUESTIONS = (
    'In one sentence, what is the usual format implied by the filename notes.md?',
    'What does the extension in an example filename report.csv commonly stand for? Give a brief general explanation.',
    'What is the usual purpose of the extension in diagram.svg? No need to inspect a file.',
    'Explain the filename backup.zip as a generic example, without looking at any file.',
)
CONDITIONS = {'ordinary': PREFERENCES, 'reflection_due': PREFERENCES, 'file_conversation': FILE_QUESTIONS}
TOKEN_LIMITS = {'actor': 2048, 'retrieval': 512, 'reflection': 1024, 'task_controller': 16}
PER_TURN_CALL_LIMIT = 8
LIVE_SECONDS = 480
RUNTIME_BUDGET_SECONDS = 720


def percentiles(values):
    ordered = sorted(values)
    return {'p50': statistics.median(ordered), 'p95': ordered[math.ceil(.95 * len(ordered)) - 1]}


def summarize(rows):
    metrics = {'samples': len(rows), 'active_records': rows[0]['active_records'],
               'turn_seconds': percentiles([row['seconds'] for row in rows]),
               'failed_turns': sum(bool(row['error'] or any(call['error'] for call in row['calls'])) for row in rows),
               'unverified_replies': sum(row['outcome'] == 'unverified' for row in rows), 'inference': {}, 'spans': {}}
    actors = [[call for call in row['calls'] if call['role'] == 'actor'] for row in rows]
    metrics['actor_outcomes'] = {
        'answered_turns': sum(row['outcome'] == 'answer' for row in rows),
        'stopped_turns': sum(row['error'] == 'ActorStopped' for row in rows),
        'length_stops': sum(call.get('done_reason') in {'length', 'max_tokens'} for calls in actors for call in calls),
        'retries': sum(max(0, len(calls) - 1) for calls in actors),
    }
    for role in TOKEN_LIMITS:
        calls = [[call['seconds'] for call in row['calls'] if call['role'] == role] for row in rows]
        active = [sum(times) for times in calls if times]
        metrics['inference'][role] = {'calls': sum(map(len, calls)), 'turns_with_calls': len(active),
                                      'seconds_per_active_turn': percentiles(active) if active else None}
    for span in ('retrieval_context', 'reflection'):
        values = [row['spans'][span] for row in rows if span in row['spans']]
        metrics['spans'][span] = {'samples': len(values), 'seconds': percentiles(values) if values else None}
    return metrics


def measure_turn(llm, controller, reflection, experience, condition, prompt, records, invoke):
    from inference import InferenceError

    row = {'condition': condition, 'prompt': prompt, 'active_records': records,
           'calls': [], 'spans': {}, 'error': None, 'outcome': None}

    def timed_call(*, role='actor', **request):
        if len(row['calls']) >= PER_TURN_CALL_LIMIT:
            row['error'] = 'TurnCallBudgetExceeded'
            raise RuntimeError('Synthetic per-turn call budget was exceeded.')
        started = perf_counter()
        call = {'role': role, 'error': None}
        row['calls'].append(call)
        try:
            response = invoke(role, request)
            call['done_reason'] = getattr(response, 'done_reason', None)
            if role == 'actor' and getattr(response.message, 'tool_calls', None):
                call['unexpected_tool_call'] = True
                raise InferenceError('malformed_response', role='actor')
            return response
        except Exception as exc:
            call['error'] = getattr(exc, 'code', type(exc).__name__)
            raise
        finally:
            call['seconds'] = perf_counter() - started

    def timed_span(name, function):
        def wrapped(*args, **kwargs):
            started = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                row['spans'][name] = row['spans'].get(name, 0) + perf_counter() - started
        return wrapped

    counter = reflection.REFLECTION_INTERVAL - 1 if condition == 'reflection_due' else 0
    with ExitStack() as stack:
        stack.enter_context(patch.multiple(
            llm, ollama_chat=timed_call, get_tools_for_goal=lambda _: [], TOOL_MAP={}, ENABLE_DYNAMIC_SELF=False,
            ENABLE_RELATIONSHIP=False, ENABLE_SELF_CONSOLIDATION=False,
            build_actor_memory_context=timed_span('retrieval_context', llm.build_actor_memory_context),
            maybe_reflect=timed_span('reflection', reflection.maybe_reflect),
        ))
        stack.enter_context(patch.multiple(
            reflection, _casual_turns_since_reflection=counter, REQUIRE_REFLECTION_USER_MEMORY_APPROVAL=True,
            ollama_chat=lambda **request: timed_call(role='reflection', **request),
        ))
        stack.enter_context(patch.object(controller, 'ollama_chat',
                                         lambda **request: timed_call(role='task_controller', **request)))
        stack.enter_context(redirect_stdout(io.StringIO()))
        experience.start_episode()
        started = perf_counter()
        try:
            answer = llm.chat([{'role': 'user', 'content': prompt}], require_cli_response=True)
            if answer == llm.UNVERIFIED_ACTION:
                row['outcome'] = 'unverified'
            elif answer.startswith('Actor failed:'):
                row['error'] = 'ActorFailed'
            elif answer.startswith('Actor stopped:'):
                row['error'] = 'ActorStopped'
            elif '[WARNING]' in answer:
                row['error'] = 'PostprocessingWarning'
            else:
                row['outcome'] = 'answer'
        except Exception as exc:
            row['error'] = type(exc).__name__
        finally:
            row['seconds'] = perf_counter() - started
            experience.end_episode()
    return row


def budget_options(options, ceiling):
    options = dict(options or {})
    return dict(options, num_predict=min(ceiling, options.get('num_predict', ceiling)))


def evaluate(output, *, live, samples, budget='fixed'):
    output = output.resolve()
    if output.is_relative_to((ROOT / 'data').resolve()) or output.exists():
        raise ValueError('Choose a new report path outside production data.')
    sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'app')]
    import isolation_support as safety
    import ollama
    sdk_chat = ollama.Client.chat
    safety.install_guards()
    import llm
    import task_controller
    from config import DEFAULT_MODEL, resolve_inference_settings
    from inference import _validate_response

    fixture = json.loads(CORPUS.read_text(encoding='utf-8'))
    runtime = {role: resolve_inference_settings(role).sanitized() for role in TOKEN_LIMITS}
    ceilings = {role: runtime[role]['num_predict'] for role in TOKEN_LIMITS} if budget == 'runtime' else TOKEN_LIMITS
    deadline = RUNTIME_BUDGET_SECONDS if budget == 'runtime' else LIVE_SECONDS
    if live and budget == 'runtime' and any(settings['model'] != DEFAULT_MODEL for settings in runtime.values()):
        raise ValueError('The runtime-budget comparison requires the installed default model for all measured roles.')
    report = {'synthetic_only': True, 'live': live, 'model': None, 'live_calls': 0,
              'budget': budget, 'runtime_roles': runtime,
              'planned_turns': samples * len(CONDITIONS) * 2, 'complete': False, 'results': [],
              'fixture_sha256': canonical_hash({'records': fixture['records'], 'turns': CONDITIONS}),
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                  'app/llm.py', 'app/memory.py', 'app/reflection.py', 'app/task_controller.py', 'app/inference.py', 'app/config.py',
                  'tools/turn_latency_eval.py', 'tools/memory_acceptance_eval.py', 'tools/retrieval_eval.py',
              )}}
    client_context = nullcontext()
    if live:
        with patch.dict(os.environ):
            os.environ.pop('OLLAMA_API_KEY', None)
            client_context = closing(ollama.Client(host=ENDPOINT, timeout=TIMEOUT_SECONDS,
                                                   follow_redirects=False, trust_env=False))
    SCRATCH.mkdir(exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    started = perf_counter()
    with client_context as client:
        if live:
            report['model'] = {'name': DEFAULT_MODEL, 'digest': installed_digest(client, DEFAULT_MODEL),
                               'endpoint': ENDPOINT, 'timeout_seconds': TIMEOUT_SECONDS, 'retries': 0,
                               'role_token_ceilings': ceilings, 'effective_num_predict': {},
                               'turn_call_limit': PER_TURN_CALL_LIMIT,
                               'between_calls_deadline_seconds': deadline}

        def invoke(role, request):
            if not live:
                content = {'actor': f'{llm.CONVERSATION_PREFIX} Understood.', 'retrieval': '{"memory_ids": []}',
                           'reflection': '{}', 'task_controller': '{"classification":"conversation"}'}[role]
                return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))
            if perf_counter() - started >= deadline:
                raise RuntimeError('Synthetic experiment deadline reached.')
            request['options'] = budget_options(request.get('options'), ceilings[role])
            effective = report['model']['effective_num_predict'].setdefault(role, [])
            if request['options']['num_predict'] not in effective:
                effective.append(request['options']['num_predict'])
            request['model'] = DEFAULT_MODEL
            report['live_calls'] += 1
            return _validate_response(sdk_chat(client, **request), role)

        for index in range(samples):
            for filled in (False, True):
                for condition, prompts in CONDITIONS.items():
                    if live and perf_counter() - started >= deadline:
                        break
                    with safety.TestTemporaryDirectory(dir=SCRATCH, prefix='aperture-turn-') as directory, \
                            _isolated_stores(Path(directory)) as (memory, experience, reflection, store, _):
                        inactive = seed_store(memory, store, fixture) if filled else set()
                        records = len(fixture['records']) - len(inactive) if filled else 0
                        row = measure_turn(llm, task_controller, reflection, experience, condition, prompts[index],
                                           records, invoke)
                    row['sample'] = index + 1
                    report['results'].append(row)
                    _write_report(output, report)
        if live and installed_digest(client, DEFAULT_MODEL) != report['model']['digest']:
            raise RuntimeError('Local model digest changed during the experiment.')
    groups = {}
    for row in report['results']:
        groups.setdefault(f"{row['active_records']}_{row['condition']}", []).append(row)
    report['groups'] = {group: summarize(rows) for group, rows in groups.items()}
    report['complete'] = len(report['results']) == report['planned_turns']
    report['guard_violations'] = list(safety._VIOLATIONS)
    _write_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--samples', type=int, choices=range(1, len(PREFERENCES) + 1), default=len(PREFERENCES))
    parser.add_argument('--budget', choices=('fixed', 'runtime'), default='fixed')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.output, live=args.live, samples=args.samples, budget=args.budget)
    print(json.dumps({'complete': report['complete'], 'live_calls': report['live_calls'], 'groups': report['groups']}))
    return int(not report['complete'] or bool(report['guard_violations'])
               or any(group['failed_turns'] for group in report['groups'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
