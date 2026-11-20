"""Replay frozen synthetic memory-approval responses without an Ollama endpoint."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from statistics import median_high, median_low
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest.mock import patch

import httpx

import memory_acceptance_eval as evaluator


FIXTURES = evaluator.FIXTURE.parent
FROZEN_INPUTS = {
    'approval_replay_responses.json': '9d078bbe779b6d964ce42e3f0f5a5239afb07cc527ac851bde1abfe30f976723',
    'approval_replay_agent_review.json': '9ebf4f03846f25bed8f27eae93438e6b676d0d061dfe52cdb8e992532888e9c2',
}
HUMAN_CASES = (
    'ownership_holdout_user_language_choice',
    'inference_holdout_assistant_assigns_career',
    'inference_holdout_trying_new_schedule',
    'temporal_holdout_tea_to_coffee',
    'temporal_holdout_conditional_travel',
)
HUMAN_DECISIONS = ('approve', 'reject', 'defer')


class OfflineReplayNetworkError(RuntimeError):
    """An HTTP request was attempted while replaying captured responses."""


@contextmanager
def _reject_network_requests():
    attempted = False
    message = 'Network requests are disabled during offline memory replay.'

    def reject(*args, **kwargs):
        nonlocal attempted
        attempted = True
        raise OfflineReplayNetworkError(message)

    # Ollama's sync and async SDK clients share HTTPX's send boundary.
    with patch.object(httpx.Client, 'send', reject), patch.object(httpx.AsyncClient, 'send', reject):
        try:
            yield
        finally:
            if attempted:
                raise OfflineReplayNetworkError(message)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _frozen_json(name: str) -> dict:
    # Text mode normalizes line endings across Git checkout settings.
    text = (FIXTURES / name).read_text(encoding='utf-8')
    if _sha256(text) != FROZEN_INPUTS[name]:
        raise ValueError(f'Frozen replay input hash changed: {name}')
    return json.loads(text)


def load_replay_inputs() -> tuple[list[dict], dict, dict]:
    fixture = evaluator.load_fixture()
    captures = _frozen_json('approval_replay_responses.json')
    review = _frozen_json('approval_replay_agent_review.json')
    case_data = json.dumps(fixture, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    if captures['case_data_sha256'] != _sha256(case_data):
        raise ValueError('Frozen replay case data hash changed.')
    if (review['responses_sha256'] != FROZEN_INPUTS['approval_replay_responses.json']
            or review['archive_report_sha256'] != captures['archive_report_sha256']
            or review['reviewer'] != 'agent'):
        raise ValueError('Agent judgments do not identify the replay archive.')
    cases = [case for case in fixture['cases'] if case['split'] == 'holdout']
    if [case['id'] for case in cases] != list(captures['cases']):
        raise ValueError('Holdout cases changed since the archived run.')
    return cases, captures, review


def _run_recorded_case(case, capture, memory, make_temp) -> dict:
    responses = iter(capture['responses'])

    def reply(**_request):
        text = next(responses)
        payload = json.loads(text)
        user_memory = payload.get('user_memory')
        if user_memory and user_memory.get('supersedes_memory_id'):
            previous, = memory._get_store().recent_records()
            if (user_memory['supersedes_memory_id'] != capture.get('archived_target_id')
                    or previous['content'] != case['seed_memory']['content']):
                raise ValueError('Archived supersession does not identify this case seed.')
            user_memory['supersedes_memory_id'] = previous['id']
            text = json.dumps(payload, ensure_ascii=False)
        return SimpleNamespace(message=SimpleNamespace(content=text))

    budget = {'calls': 0, 'max_calls': len(capture['responses']),
              'started': time.perf_counter(), 'max_seconds': 30}
    with make_temp() as directory:
        result = evaluator.run_case(case, Path(directory), model_call=reply,
                                    budget=budget, decision_condition='fixture')

    if result['error'] or budget['calls'] != len(capture['responses']):
        raise RuntimeError(f"Replay failed for {case['id']}: {result['error'] or 'response count changed'}")
    seed_ids = {item['id'] for item in result['before_memories']}
    text_outputs = [item['content'] for item in result['pending']]
    text_outputs.extend(
        item['content'] for item in result['after_memories']
        if item['subject'] == 'user' and item['active'] and item['id'] not in seed_ids)
    if sorted(text_outputs) != capture['proposal_texts']:
        raise ValueError(f"Replay no longer reproduces the reviewed text for {case['id']}.")
    return dict(result, status='complete')


def _human_review_packet(cases: list[dict], queued: dict, report_hash: str) -> dict:
    by_id = {row['id']: row for row in queued['cases']}
    included = {case['id']: case for case in cases}
    review_cases = {}
    for case_id in HUMAN_CASES:
        case = included[case_id]
        row = by_id[case_id]
        proposals = [{key: proposal[key] for key in ('content', 'category', 'target_content')}
                     for proposal in row['pending']]
        review_cases[case_id] = {
            'dialogue': case['messages'],
            'prior_user_observation': case.get('seed_observation'),
            'prior_active_memory': case.get('seed_memory'),
            'proposals': proposals,
            'human_response': {'decision': None, 'verdict': None, 'reason': None,
                               'review_seconds': None},
        }
    return {
        'synthetic_only': True, 'reviewer': 'human', 'status': 'awaiting_independent_review',
        'report_sha256': report_hash,
        'coverage': {'decisions': 0, 'fidelity': 0, 'duration': 0, 'cases_in_packet': len(review_cases)},
        'cases': review_cases,
    }


def score_human_review(packet_path: Path, queued_path: Path, output: Path,
                       *, overwrite: bool = False) -> dict:
    """Score only judgments supplied in a returned independent human review packet."""
    output = output.resolve()
    if output in {packet_path.resolve(), queued_path.resolve()}:
        raise ValueError('Human review output must be separate from both inputs.')
    if output.exists() and not overwrite:
        raise FileExistsError('Human review output exists. Use --overwrite to replace it.')

    cases, captures, _ = load_replay_inputs()
    raw = queued_path.read_bytes()
    report_hash = hashlib.sha256(raw).hexdigest()
    queued = json.loads(raw)
    if (not isinstance(queued, dict)
            or queued.get('inference_source') != 'archived_model_responses'
            or queued.get('require_reflection_user_memory_approval') is not True
            or queued.get('complete') is not True
            or queued.get('frozen_input_sha256') != FROZEN_INPUTS
            or queued.get('case_data_sha256') != captures['case_data_sha256']
            or 'semantic_reviewer' in queued):
        raise ValueError('Human review requires the unscored queued replay report.')

    packet_raw = packet_path.read_bytes()
    packet = json.loads(packet_raw)
    expected = _human_review_packet(cases, queued, report_hash)
    if (not isinstance(packet, dict)
            or packet.get('reviewer') != 'human' or packet.get('synthetic_only') is not True
            or packet.get('report_sha256') != report_hash
            or not isinstance(packet.get('cases'), dict)
            or packet['cases'].keys() != expected['cases'].keys()):
        raise ValueError('Human review requires the complete packet for this queued report.')

    decisions, verdicts = Counter(), Counter()
    durations = []
    responses = {}
    for case_id, original in expected['cases'].items():
        submitted = packet['cases'][case_id]
        if not isinstance(submitted, dict):
            raise ValueError(f'Invalid packet case {case_id}.')
        response = submitted.get('human_response')
        if not isinstance(response, dict) or response.keys() != original['human_response'].keys():
            raise ValueError(f'Invalid human response fields for {case_id}.')
        if dict(submitted, human_response=original['human_response']) != original:
            raise ValueError(f'Packet evidence changed for {case_id}.')

        decision, verdict = response['decision'], response['verdict']
        reason, duration = response['reason'], response['review_seconds']
        if decision is not None and decision not in HUMAN_DECISIONS:
            raise ValueError(f'Invalid human decision for {case_id}.')
        if verdict is not None and (not isinstance(verdict, str)
                                    or verdict not in evaluator.SEMANTIC_VERDICTS):
            raise ValueError(f'Invalid human fidelity verdict for {case_id}.')
        if (reason is not None or verdict is not None) and (
                not isinstance(reason, str) or not reason.strip()):
            raise ValueError(f'A fidelity verdict needs a reason for {case_id}.')
        if duration is not None:
            try:
                valid_seconds = (type(duration) in (int, float)
                                 and duration >= 0 and math.isfinite(duration))
            except OverflowError:
                valid_seconds = False
            if not valid_seconds:
                raise ValueError(f'Invalid human review time for {case_id}.')
            durations.append(duration)
        if decision is not None:
            decisions[decision] += 1
        if verdict is not None:
            verdicts[verdict] += 1
        responses[case_id] = response

    decided, reviewed, timed = sum(decisions.values()), sum(verdicts.values()), len(durations)
    median_seconds = None
    if durations:
        lower, upper = median_low(durations), median_high(durations)
        median_seconds = lower + (upper - lower) / 2
    scored = {
        'reviewer': 'human', 'synthetic_only': True,
        'status': 'judgments_reported' if decided or reviewed or timed else 'awaiting_independent_review',
        'queued_report_sha256': report_hash,
        'submitted_packet_sha256': hashlib.sha256(packet_raw).hexdigest(),
        'coverage': {'decisions': decided, 'fidelity': reviewed, 'duration': timed,
                     'cases_in_packet': len(responses)},
        'decision_counts': {decision: decisions[decision] for decision in HUMAN_DECISIONS},
        'fidelity': {'faithful': verdicts['faithful'], 'unfaithful': verdicts['unfaithful'],
                     'rate': verdicts['faithful'] / reviewed if reviewed else None},
        'median_review_seconds': median_seconds,
        'cases': responses,
    }
    evaluator._write_report(output, scored)
    return scored


def replay(output: Path, *, overwrite: bool = False, make_temp=None) -> dict:
    """Write raw and agent-scored policy runs and a blank human review packet."""
    cases, captures, review = load_replay_inputs()
    output = output.resolve()
    if output.is_relative_to((evaluator.ROOT / 'data').resolve()):
        raise ValueError('Replay output cannot be under production data.')
    names = [f'{policy}-{suffix}.json' for policy in ('automatic', 'queued')
             for suffix in ('raw', 'agent-review', 'scored')]
    names.append('human-review-packet.json')
    if not overwrite and any((output / name).exists() for name in names):
        raise FileExistsError('Replay output exists. Choose a new directory or use --overwrite.')
    output.mkdir(parents=True, exist_ok=True)
    if make_temp is None:
        def make_temp():
            return TemporaryDirectory(prefix='memory-approval-replay-', dir=output)
    with _reject_network_requests(), make_temp() as directory:
        with evaluator._isolated_stores(Path(directory)) as (memory, _, reflection, _, _):
            pass

    source = evaluator._source_hashes()
    replay_source = Path(__file__).resolve()
    source[replay_source.relative_to(evaluator.ROOT).as_posix()] = evaluator._digest(replay_source)
    summary = {'ollama_calls': 0, 'policies': {}}
    for policy, require_approval in (('automatic', False), ('queued', True)):
        with _reject_network_requests(), \
                patch.object(reflection, 'MODEL', captures['model']), \
                patch.object(reflection, 'MEMORY_OPTIONS', captures['memory_options']), \
                patch.object(reflection, 'REQUIRE_REFLECTION_USER_MEMORY_APPROVAL', require_approval):
            results = [_run_recorded_case(case, captures['cases'][case['id']], memory, make_temp)
                       for case in cases]
        report = {
            'schema_version': 2, 'synthetic_only': True, 'live': False, 'complete': True,
            'inference_source': 'archived_model_responses', 'network_model_calls_attempted': 0,
            'replayed_responses': sum(len(row['model_calls']) for row in results),
            'selected_split': 'holdout', 'cases': results,
            'decision_condition': 'fixture', 'decision_origin': 'deterministic_fixture',
            'require_reflection_user_memory_approval': require_approval,
            'model': captures['model'], 'model_metadata': {'digest': captures['model_digest']},
            'model_settings': {'memory_options': captures['memory_options']},
            'fixture_sha256': evaluator._digest(evaluator.FIXTURE),
            'case_data_sha256': captures['case_data_sha256'],
            'frozen_input_sha256': FROZEN_INPUTS, 'source_sha256': source,
            'archive_report_sha256': captures['archive_report_sha256'],
            'archive_case_fixture_sha256': captures['archive_case_fixture_sha256'],
            'archive_source_sha256': captures['archive_source_sha256'],
            'archive_agent_review_sha256': review['archive_review_sha256'],
            'semantic_status': 'unreviewed',
            'metrics': evaluator.summarize(results),
        }
        raw_path = output / f'{policy}-raw.json'
        evaluator._write_report(raw_path, report)
        raw_hash = evaluator._digest(raw_path)
        review_path = output / f'{policy}-agent-review.json'
        evaluator._write_report(review_path, {
            'report_sha256': raw_hash, 'reviewer': 'agent', 'cases': review['cases'],
        })
        scored = evaluator.score_semantic_review(raw_path, review_path,
                                                output / f'{policy}-scored.json',
                                                overwrite=overwrite)
        metrics = scored['metrics']['holdout']
        summary['policies'][policy] = {
            'raw_report_sha256': raw_hash, 'replayed_responses': report['replayed_responses'],
            'after_decision': metrics['after_decision'],
            'agent_fidelity': {'faithful': metrics['semantic_faithful'],
                               'reviewed': metrics['semantic_reviewed']},
        }
        if require_approval:
            packet = _human_review_packet(cases, report, raw_hash)
            evaluator._write_report(output / 'human-review-packet.json', packet)
            summary['independent_human_review'] = packet['coverage']
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output-dir', type=Path)
    mode.add_argument('--score-human-review', type=Path, metavar='PACKET')
    parser.add_argument('--queued-report', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if args.score_human_review and not (args.queued_report and args.output):
        parser.error('Human review scoring requires --queued-report and --output.')
    if args.output_dir and (args.queued_report or args.output):
        parser.error('--queued-report and --output are for human review scoring.')
    output = (args.output or args.output_dir).resolve()
    if not output.is_relative_to((evaluator.ROOT / '.reliability-tmp').resolve()):
        parser.error('Output must be under .reliability-tmp.')
    result = (score_human_review(args.score_human_review, args.queued_report, output,
                                 overwrite=args.overwrite)
              if args.score_human_review else replay(output, overwrite=args.overwrite))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
