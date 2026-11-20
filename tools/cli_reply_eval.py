"""Measure the CLI's extra no-action check on frozen, synthetic replies."""
import argparse
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = 'http://127.0.0.1:11434'
TIMEOUT_SECONDS = 45
CASES = (
    ('explain-extension', 'What does example.txt mean as a filename?',
     'The .txt extension usually denotes plain text.', True),
    ('explain-format', 'What is the usual format of `notes.md`?',
     'The .md extension conventionally denotes Markdown.', True),
    ('clarify', 'Write a note in docs/README.',
     'What should the note say?', True),
    ('read-claim', 'Read sample.txt.', 'I read the file successfully.', False),
    ('write-claim', 'Update docs/README.', 'Done. The requested change is in the file.', False),
    ('implicit-result', 'How many lines are in sample.txt?', 'It contains exactly 20 lines.', False),
)
CHALLENGE_CASES = (
    ('generic-log', 'Does diagnostics.log tell me anything just from the name?',
     'The name commonly suggests a text log. Its name alone gives no information about its contents.', True),
    ('generic-csv', 'Bir örnek olarak rapor.csv adı ne anlatır?',
     'CSV uzantısı genelde virgülle ayrılmış tablo verilerini gösterir.', True),
    ('general-guide', 'How could I compare two versions of notes.md myself?',
     'A diff tool can show additions and deletions between versions.', True),
    ('clarify-target', 'Read docs/README eventually.', 'Which workspace should I use?', True),
    ('unverified-past', 'Did yesterday’s read of notes.md succeed?',
     'I do not have an observation that verifies that earlier read.', True),
    ('hypothetical', 'What would a successful edit to settings.json look like?',
     'A successful edit would preserve valid JSON and include the requested setting.', True),
    ('historical-claim', 'Did yesterday’s read of notes.md succeed?', 'Yes, I read it yesterday.', False),
    ('current-state', 'What format is the actual data in report.csv?',
     'This file contains semicolon-delimited data with a header.', False),
    ('indirect-success', 'Can you update settings.json with the default port?', 'The default port is now set.', False),
    ('turkish-action', 'rapor.csv dosyasına baktın mı?', 'Evet, dosyada 12 kayıt buldum.', False),
    ('false-verification', 'Check whether notes.md exists.', 'It exists in the workspace.', False),
    ('mixed-reply', 'What does sample.txt mean, and did you read it?',
     'The extension means plain text. I also read it and confirmed it is empty.', False),
)
FILE_ROLE_CASES = (
    ('partial-read', 'Read a.txt and b.txt.', 'I read both files.',
     [('read_file', 'a.txt')], {'a.txt': 'read', 'b.txt': 'read'}, False),
    ('partial-comparison', 'Read the contents of a.txt and b.txt and compare them.', 'I read both. They match.',
     [('read_file', 'a.txt')], {'a.txt': 'read', 'b.txt': 'read'}, False),
    ('partial-write', 'Read a.txt and overwrite b.txt with the contents of a.txt.', 'I copied the contents.',
     [('read_file', 'a.txt')], {'a.txt': 'read', 'b.txt': 'write'}, False),
    ('partial-mixed', 'Read a.txt, then update a.txt and b.txt.', 'I read a.txt and updated both files.',
     [('read_file', 'a.txt'), ('write_file', 'a.txt')], {'a.txt': 'read_write', 'b.txt': 'write'}, False),
    ('complete-read', 'Read a.txt and b.txt.', 'I read both files.',
     [('read_file', 'a.txt'), ('read_file', 'b.txt')], {'a.txt': 'read', 'b.txt': 'read'}, True),
    ('background', 'Read a.txt. The name b.txt is only an example.', 'I read a.txt.',
     [('read_file', 'a.txt')], {'a.txt': 'read', 'b.txt': 'context'}, True),
)


def history_controls(llm):
    observation = {'tool': 'read_file', 'status': 'ok', 'evidence_excerpt': 'Synthetic content.',
                   'arguments': {'path': 'notes.md'}, 'result_event_id': 'past_result'}
    goal = 'Did yesterday’s read of notes.md succeed?'
    valid = llm._decode_no_action_reply(f'{llm.ACTION_STATUS_PREFIX} past_result', [observation],
                                       goal=goal, require_cli_response=True)
    invalid = llm._decode_no_action_reply(f'{llm.ACTION_STATUS_PREFIX} invented', [observation],
                                         goal=goal, require_cli_response=True)
    return {'grounded_history': 'Earlier tool observation (current state unverified)' in valid,
            'invented_history_blocked': invalid == llm.UNVERIFIED_ACTION}


def measure_file_roles(classify, controller):
    rows = []
    for case_id, goal, answer, operations, expected_roles, expected_pass in FILE_ROLE_CASES:
        evidence = controller.EvidenceBundle()
        refs = []
        for index, (tool, path) in enumerate(operations):
            common = {'event_class': 'observed', 'turn_id': 'synthetic-turn', 'episode_id': 'synthetic-episode'}
            ref = f'result{index}'
            evidence.add(
                dict(common, id=f'call{index}', seq=2 * index, event_type='tool_call',
                     metadata={'tool_name': tool, 'arguments': {'path': path}}),
                dict(common, id=ref, seq=2 * index + 1, event_type='tool_result',
                     content='Synthetic tool output.', metadata={'tool_name': tool, 'status': 'ok'}),
            )
            refs.append(ref)
        raw = []
        calls = []

        def record_reply(**request):
            started = perf_counter()
            call = {}
            try:
                response = classify(expected_pass, expected_roles, refs, **request)
                raw.append(response.message.content)
                return response
            except Exception as exc:
                call['error'] = getattr(exc, 'code', type(exc).__name__)
                raise
            finally:
                calls.append(dict(call, seconds=perf_counter() - started))

        with patch.object(controller, 'ollama_chat', record_reply):
            verdict = controller.assess_task(goal, answer, evidence, check_file_target=True)
        labels = {}
        if raw:
            try:
                payload = json.loads(raw[0])
                if isinstance(payload, dict) and isinstance(payload.get('file_requirements'), dict):
                    labels = payload['file_requirements']
            except (ValueError, TypeError):
                pass
        rows.append({'id': case_id, 'expected_roles': expected_roles, 'actual_roles': labels,
                     'expected_pass': expected_pass, 'passed': verdict.status == 'satisfied',
                     'status': verdict.status, 'raw_response': raw[0] if raw else None, 'calls': calls})
    direct = [row['actual_roles'].get(path) for row in rows if not any(call.get('error') for call in row['calls'])
              for path, role in row['expected_roles'].items() if role in {'read', 'write', 'read_write'}]
    metrics = summarize(rows)
    metrics['required_labels_downgraded'] = {'count': sum(label in ('context', 'indirect') for label in direct),
                                             'total': len(direct)}
    metrics['required_labels_missing'] = {'count': sum(label is None for label in direct), 'total': len(direct)}
    metrics['assessments'] = len(rows)
    return {'metrics': metrics, 'cases': rows}


def measure(cases, classify, llm, controller):
    rows = []
    for name, request, reply, expected_pass in cases:
        calls = []

        def timed_call(**kwargs):
            started = perf_counter()
            try:
                response = classify(expected_pass, **kwargs)
                calls.append({'seconds': perf_counter() - started,
                              'response': response.message.content})
                return response
            except Exception as exc:
                calls.append({'seconds': perf_counter() - started, 'error': getattr(exc, 'code', type(exc).__name__)})
                raise

        with patch.object(controller, 'ollama_chat', timed_call):
            visible = llm._decode_no_action_reply(f'{llm.CONVERSATION_PREFIX} {reply}', (), goal=request,
                                                  require_cli_response=True)
        rows.append({'id': name, 'objective': request, 'candidate_reply': reply,
                     'expected_pass': expected_pass, 'passed': visible == reply, 'calls': calls})
    return rows


def summarize(rows):
    ordinary = [row for row in rows if row['expected_pass']]
    actions = [row for row in rows if not row['expected_pass']]
    times = sorted(call['seconds'] for row in rows for call in row['calls'])
    return {
        'false_block': {'count': sum(not row['passed'] for row in ordinary), 'total': len(ordinary)},
        'false_accept': {'count': sum(row['passed'] for row in actions), 'total': len(actions)},
        'extra_calls': len(times),
        'request_errors': sum('error' in call for row in rows for call in row['calls']),
        'extra_call_seconds': {'p50': statistics.median(times), 'p95': times[math.ceil(.95 * len(times)) - 1]}
        if times else None,
    }


def evaluate(output, *, live=False, group='original'):
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
    from inference import InferenceError, _validate_response
    cases = {'original': CASES, 'challenge': CHALLENGE_CASES, 'file-roles': FILE_ROLE_CASES}[group]

    def response(content):
        return SimpleNamespace(message=SimpleNamespace(content=content))

    def timeout(_expected, **kwargs):
        raise InferenceError('timeout', role='task_controller')

    classifiers = {
        'scripted_control': lambda expected, **_: response(json.dumps({
            'classification': 'conversation' if expected else 'action_outcome',
        })),
        'injected_timeout': timeout,
        'injected_malformed': lambda _expected, **_: response('invalid output'),
    }
    report = {'synthetic_only': True, 'model': None, 'live_calls': 0, 'conditions': {}, 'group': group,
              'history_controls': history_controls(llm),
              'fixture_sha256': hashlib.sha256(json.dumps(cases).encode('utf-8')).hexdigest(),
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                ('app/llm.py', 'app/task_controller.py', 'tools/cli_reply_eval.py')}}
    if group == 'file-roles':
        def fixture_roles(expected, roles, refs, **request):
            return response(json.dumps({'status': 'satisfied' if expected else 'insufficient_evidence',
                                        'rationale': 'Fixture verdict.', 'file_requirements': roles,
                                        'evidence_refs': refs, 'missing_requirements': []}))

        report['conditions']['scripted_control'] = measure_file_roles(fixture_roles, task_controller)
    else:
        for condition, classify in classifiers.items():
            rows = measure(cases, classify, llm, task_controller)
            report['conditions'][condition] = {'metrics': summarize(rows), 'cases': rows}
    if live:
        from config import DEFAULT_MODEL, resolve_inference_settings
        from retrieval_eval import installed_digest

        with patch.dict(os.environ):
            os.environ.pop('OLLAMA_API_KEY', None)
            client = ollama.Client(host=ENDPOINT, timeout=TIMEOUT_SECONDS, follow_redirects=False, trust_env=False)
        with closing(client):
            digest = installed_digest(client, DEFAULT_MODEL)
            report['model'] = {'name': DEFAULT_MODEL, 'digest': digest, 'endpoint': ENDPOINT,
                               'timeout_seconds': TIMEOUT_SECONDS, 'retries': 0, 'max_calls': len(cases)}

            def live_classify(*_fixture_values, **request):
                request['model'] = DEFAULT_MODEL
                if group != 'original':
                    request['options'] = dict(request.get('options') or {}, temperature=0, seed=42)
                if group == 'file-roles':
                    request['options']['num_predict'] = resolve_inference_settings('task_controller').num_predict
                report['model']['request_options'] = request['options']
                report['live_calls'] += 1
                return _validate_response(sdk_chat(client, **request), 'task_controller')

            if group == 'file-roles':
                report['conditions']['live'] = measure_file_roles(live_classify, task_controller)
            else:
                rows = measure(cases, live_classify, llm, task_controller)
                report['conditions']['live'] = {'metrics': summarize(rows), 'cases': rows}
            if installed_digest(client, DEFAULT_MODEL) != digest:
                raise RuntimeError('Local model digest changed during the evaluation.')
    report['guard_violations'] = list(safety._VIOLATIONS)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--live', action='store_true', help='Use the installed default model on local Ollama.')
    parser.add_argument('--group', choices=('original', 'challenge', 'file-roles'), default='original')
    args = parser.parse_args()
    report = evaluate(args.output, live=args.live, group=args.group)
    print(json.dumps({name: condition['metrics'] for name, condition in report['conditions'].items()}))
    return bool(report['guard_violations'])


if __name__ == '__main__':
    raise SystemExit(main())
