"""Day 1 synthetic retrieval baseline extending conversation_eval's runtime.

No inference is permitted. Actor results measure its existing selector-failure
fallback; the separately labelled gold-ID selector is a contract/cap probe only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'evaluations' / 'reliability_day1'


def build_fixture():
    """Original fictional records; fixed IDs/timestamps and a frozen 12/36 split."""
    records = []

    def memory(key, content, subject='user', **extra):
        records.append(dict(id=f'm{len(records) + 1:011d}', key=key, content=content,
                            subject=subject, category='fact', importance=3, **extra))

    # These oldest facts deliberately have no query-token overlap and fall outside
    # the selector's recent tail after the distractors have been inserted.
    memory('old_dawn', 'Leonie reserves sunrise for concentrated tasks.')
    memory('old_quiet', 'Mert kulaklık yerine sessiz bir oda seçiyor.')
    memory('tea', 'Leonie drinks rooibos tea after lunch.')
    memory('bicycle', 'Leonie rides a copper bicycle to the observatory.')
    memory('cat', 'Leonie named her cat Juniper.')
    memory('language', 'Leonie studies Finnish on Thursdays.')
    memory('meeting', 'The Juniper workshop begins at 14:30 on Saturday.', 'project')
    memory('backup', 'The Cobalt project backup runs every Tuesday.', 'project')
    memory('garden', 'Leonie grows basil on the balcony.')
    memory('music', 'Leonie practices the cello before dinner.')
    memory('station', 'The Cobalt field team meets at Northgate station.', 'project')
    memory('museum', 'Leonie visits the maritime museum in November.')
    memory('tr_food', 'Mert öğle yemeğinde mercimek çorbası içiyor.')
    memory('tr_book', 'Mert hafta sonları bilimkurgu romanları okuyor.')
    memory('tr_walk', 'Mert akşam yürüyüşünü nehir kıyısında yapıyor.')
    memory('user_color', 'Leonie prefers amber interface colors.')
    memory('self_color', 'APERTURE previously expressed a preference for violet interface colors.', 'aperture')
    memory('user_editor', 'Leonie edits notes with Maplepad.')
    memory('self_editor', 'APERTURE used Quartzpad during a synthetic tool experiment.', 'aperture')
    memory('old_city', 'Leonie lives in Frosthaven.')
    memory('city', 'Leonie now lives in Cedarwick.', supersedes='old_city')
    memory('old_schedule', 'The Cobalt review is on Monday.', 'project')
    memory('schedule', 'The Cobalt review is now on Friday.', 'project', supersedes='old_schedule')
    memory('forgotten_code', 'The fictional locker code is marigold-seven.', forgotten=True)
    memory('forgotten_trip', 'Leonie planned a fictional trip to Gullport.', forgotten=True)
    ids = {row['key']: row['id'] for row in records}
    cases = []

    def case(identifier, split, kind, query, gold=(), **extra):
        cases.append(dict(id=identifier, split=split, kind=kind, query=query,
                          gold_ids=[ids[key] for key in gold], recent_dialogue='', **extra))

    for identifier, kind, query, gold, extra in [
        ('T01', 'direct', 'rooibos tea', ['tea'], {}),
        ('T02', 'direct', 'copper bicycle observatory', ['bicycle'], {}),
        ('T03', 'direct', 'Juniper cat', ['cat'], {}),
        ('T04', 'direct', 'Cobalt backup Tuesday', ['backup'], {}),
        ('T05', 'paraphrase', 'What instrument does Leonie rehearse?', ['music'], {}),
        ('T06', 'paraphrase', 'Which herbs are grown outside her apartment?', ['garden'], {}),
        ('T07', 'cross_language', 'Leonie hangi dili öğreniyor?', ['language'], {}),
        ('T08', 'cross_language', 'What soup does Mert have for lunch?', ['tr_food'], {}),
        ('T09', 'no_answer', 'What is the launch date of the Pelican spacecraft?', [], {}),
        ('T10', 'correction', 'Where does Leonie live now?', ['city'], {'stale_ids': [ids['old_city']]}),
        ('T11', 'ownership', 'Which interface color does Leonie prefer?', ['user_color'], {'expected_subject': 'user'}),
        ('T12', 'follow_up', 'And what day is that?', ['backup'], {'dialogue': 'user: When is the Cobalt project backup?'}),
    ]:
        if 'dialogue' in extra:
            dialogue = extra.pop('dialogue')
        else:
            dialogue = ''
        case(identifier, 'tune', kind, query, gold, **extra)
        cases[-1]['recent_dialogue'] = dialogue

    held = [
        ('direct', 'Leonie Finnish Thursdays', ['language'], {}),
        ('direct', 'Juniper workshop Saturday', ['meeting'], {}),
        ('direct', 'balcony basil', ['garden'], {}),
        ('direct', 'cello dinner', ['music'], {}),
        ('direct', 'Northgate station', ['station'], {}),
        ('direct', 'maritime museum November', ['museum'], {}),
        ('direct', 'Mert mercimek çorbası', ['tr_food'], {}),
        ('direct', 'Mert bilimkurgu romanları', ['tr_book'], {}),
        ('paraphrase', 'What hot beverage follows her midday meal?', ['tea'], {}),
        ('paraphrase', 'How does she commute to the astronomy building?', ['bicycle'], {}),
        ('paraphrase', 'What is the name of her pet feline?', ['cat'], {}),
        ('paraphrase', 'When is the weekly data preservation task?', ['backup'], {}),
        ('paraphrase', 'Where does the field crew rendezvous?', ['station'], {}),
        ('paraphrase', 'Which season includes her visit to the naval collection?', ['museum'], {}),
        ('cross_language', 'Leonie öğle yemeğinden sonra ne içer?', ['tea'], {}),
        ('cross_language', 'Leonie hangi çalgıyı çalışıyor?', ['music'], {}),
        ('cross_language', 'Leonie balkonda ne yetiştiriyor?', ['garden'], {}),
        ('cross_language', 'What kind of novels does Mert read on weekends?', ['tr_book'], {}),
        ('cross_language', 'Where does Mert take an evening stroll?', ['tr_walk'], {}),
        ('cross_language', 'Juniper atölyesi hangi saatte başlıyor?', ['meeting'], {}),
        ('no_answer', 'What is Leonie\'s passport number?', [], {}),
        ('no_answer', 'Who won the fictional Nebula chess championship?', [], {}),
        ('no_answer', 'Does Mert own a submarine?', [], {}),
        ('no_answer', 'What is the Cobalt project payroll amount?', [], {}),
        ('no_answer', 'quasarflux xenolithium', [], {}),
        ('no_answer', 'Where did Leonie attend elementary school?', [], {}),
        ('correction', 'Leonie Cedarwick Frosthaven current home', ['city'], {'stale_ids': [ids['old_city']]}),
        ('correction', 'Cobalt review Monday Friday current schedule', ['schedule'], {'stale_ids': [ids['old_schedule']]}),
        ('forgotten', 'marigold-seven fictional locker code', [], {'inactive_ids': [ids['forgotten_code']]}),
        ('forgotten', 'Leonie Gullport fictional trip', [], {'inactive_ids': [ids['forgotten_trip']]}),
        ('ownership', 'Leonie amber violet interface color preference', ['user_color'], {'expected_subject': 'user'}),
        ('ownership', 'Leonie Maplepad Quartzpad notes editor', ['user_editor'], {'expected_subject': 'user'}),
        ('follow_up', 'And at what time?', ['meeting'], {'dialogue': 'user: Tell me about the Juniper workshop on Saturday.'}),
        ('follow_up', 'Where does he do that?', ['tr_walk'], {'dialogue': 'user: Mert akşam yürüyüşüne çıkıyor.'}),
        ('old_zero_overlap', 'Which hours suit deep concentration?', ['old_dawn'], {}),
        ('old_zero_overlap', 'What acoustic setting supports focused work?', ['old_quiet'], {}),
    ]
    for index, (kind, query, gold, extra) in enumerate(held, 1):
        dialogue = extra.pop('dialogue', '')
        case(f'H{index:02d}', 'held_out', kind, query, gold, **extra)
        cases[-1]['recent_dialogue'] = dialogue
    for index in range(128):
        memory(f'distractor_{index:03d}',
               f'Synthetic inventory shelf {index:03d} contains ceramic sample z{index:03d}.', 'world')
    return dict(schema_version=1, synthetic=True, split_rule='T01-T12 tune; H01-H36 held_out; frozen before measurement',
                records=records, cases=cases)


@contextmanager
def isolated_runtime():
    """Keep the existing evaluator and unittest guards, including their ledgers."""
    sys.dont_write_bytecode = True
    for path in (str(ROOT / 'tests'), str(ROOT / 'tools')):
        if path not in sys.path:
            sys.path.insert(0, path)
    safety = importlib.import_module('isolation_support')
    safety.install_guards()  # No app imports precede this.
    support = importlib.import_module('conversation_eval')
    scratch = ROOT / '.reliability-tmp'
    scratch.mkdir(exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(patch.object(tempfile, 'tempdir', str(scratch)))
        stack.enter_context(patch.dict(os.environ, {'APERTURE_DYNAMIC_SELF': '0',
            'APERTURE_RELATIONSHIP': '0', 'APERTURE_SELF_CONSOLIDATION': '0'}))
        directory = stack.enter_context(safety.TestTemporaryDirectory(prefix='aperture-retrieval-day1-'))
        llm, _ = stack.enter_context(support.synthetic_runtime(directory))
        memory = importlib.import_module('memory')
        goals = importlib.import_module('goals')
        # Audited modules are lazy at import; all three stores are temp before
        # any measured calls. Replace the helper's unrelated synthetic seed.
        store = memory.MemoryStore(Path(directory) / 'reliability-memory.db')
        stack.enter_context(patch.object(memory, '_STORE', store))
        stack.enter_context(patch.object(goals, '_STORE', goals.GoalStore(Path(directory) / 'goals.db')))
        for flag in ('ENABLE_DYNAMIC_SELF', 'ENABLE_RELATIONSHIP', 'ENABLE_SELF_CONSOLIDATION'):
            stack.enter_context(patch.object(llm, flag, False))
        yield memory, store, llm, safety
        if safety._VIOLATIONS:
            raise AssertionError('Reliability evaluator encountered isolation guard violations')


def seed_fixture(memory, store, fixture):
    by_key = {}
    for index, item in enumerate(fixture['records']):
        # Only synthetic writes. Stable clock/IDs preserve rank ties across runs.
        timestamp = f'2026-01-01T00:{index // 60:02d}:{index % 60:02d}+00:00'
        kwargs = {key: item[key] for key in ('content', 'category', 'importance', 'subject')}
        kwargs['source'] = 'synthetic_day1_fixture'
        with patch.object(memory.uuid, 'uuid4', return_value=SimpleNamespace(hex=item['id'])), \
                patch.object(memory, '_now', return_value=timestamp):
            if item.get('supersedes'):
                record = store.supersede(by_key[item['supersedes']], **kwargs)
            else:
                record = store.remember(**kwargs)
            if record['id'] != item['id']:
                raise AssertionError('Synthetic fixture ID was not preserved')
            if item.get('forgotten'):
                store.forget(record['id'])
        by_key[item['key']] = record['id']


def _rate(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                rate=numerator / denominator if denominator else None)


def summarize(rows):
    # Failed executions remain in the report/latency samples, but provide no
    # retrieval measurement and must not become zero-quality observations.
    completed = [row for row in rows if not row.get('error')]
    answerable = [row for row in completed if row['gold_ids']]
    no_answer = [row for row in completed if not row['gold_ids']]
    corrections = [row for row in completed if row['kind'] == 'correction']
    # No-answer rows do not enter Recall/MRR denominators.
    recalls = [len(set(row['returned_ids'][:5]) & set(row['gold_ids'])) / len(set(row['gold_ids']))
               for row in answerable]
    reciprocal = [next((1 / rank for rank, value in enumerate(row['returned_ids'][:5], 1)
                        if value in row['gold_ids']), 0) for row in answerable]
    latencies = [row['latency_seconds'] for row in rows if row.get('latency_seconds') is not None]
    covered = [row for row in answerable if row.get('candidate_ids') is not None]
    return dict(cases=len(rows), completed_measurements=len(completed), answerable_queries=len(answerable),
        macro_recall_at_5=_rate(sum(recalls), len(answerable)),
        mrr_at_5=_rate(sum(reciprocal), len(answerable)),
        no_answer_false_positive_rate=_rate(sum(bool(row['returned_ids']) for row in no_answer), len(no_answer)),
        current_version_accuracy=_rate(sum(set(row['gold_ids']).issubset(row['returned_ids'][:5])
            and not row['stale_returned_ids'] for row in corrections), len(corrections)),
        stale_returns=sum(len(row['stale_returned_ids']) for row in rows),
        inactive_returns=sum(len(row['inactive_returned_ids']) for row in rows),
        ownership_tag_violations=sum(len(row['ownership_tag_violations']) for row in rows),
        other_subject_returns_on_owner_scoped_queries=sum(len(row['other_subject_ids']) for row in rows),
        candidate_gold_coverage=_rate(sum(len(set(row['candidate_ids']) & set(row['gold_ids'])) /
            len(set(row['gold_ids'])) for row in covered), len(covered)),
        retrieval_latency=dict(samples=len(latencies), condition='single process; fixture/schema seeded before timing; first timed query has no retrieval warmup, later queries warm; OS cache uncontrolled; no model calls',
            total_seconds=sum(latencies), mean_seconds=statistics.mean(latencies) if latencies else None,
            min_seconds=min(latencies) if latencies else None, max_seconds=max(latencies) if latencies else None),
        errors=sum(bool(row.get('error')) for row in rows))


def measure_case(memory, item, path, records):
    candidates = None

    def unavailable_selector(**kwargs):
        nonlocal candidates
        candidates = [record['id'] for record in kwargs['candidates']]
        raise RuntimeError('Deterministic selector-unavailable probe; no inference')

    def gold_contract_selector(**kwargs):
        nonlocal candidates
        candidates = [record['id'] for record in kwargs['candidates']]
        return [value for value in item['gold_ids'] if value in candidates][:kwargs['limit']]

    started = time.perf_counter()
    error = None
    try:
        if path == 'lexical_tool':
            text = memory.search_memory(item['query'], limit=5)
            if text.startswith('MEMORY_ERROR:'):
                raise RuntimeError(text)
            returned = re.findall(r'^\[([^\]]+)\]', text, flags=re.MULTILINE)
            subjects = dict(re.findall(r'^\[([^\]]+)\] \[([^\]]+)\]', text, flags=re.MULTILINE))
        else:
            selector = unavailable_selector if path == 'actor_context_fallback' else gold_contract_selector
            text = memory.build_actor_memory_context(item['query'], recent_dialogue=item['recent_dialogue'], selector=selector)
            returned = re.findall(r'^- id=(\S+) ', text, flags=re.MULTILINE)
            subjects = dict(re.findall(r'^- id=(\S+) \[([^\]]+)\]', text, flags=re.MULTILINE))
    except Exception as exc:
        returned, subjects = [], {}
        error = f'{type(exc).__name__}: {exc}'
    elapsed = time.perf_counter() - started
    row = {key: item[key] for key in ('id', 'split', 'kind', 'query', 'recent_dialogue', 'gold_ids')}
    stale = [record['id'] for record in records.values() if record.get('superseded')]
    inactive = [record['id'] for record in records.values() if record.get('forgotten') or record.get('superseded')]
    row.update(path=path, returned_ids=returned, returned_subjects=subjects,
        gold_ranks={value: returned.index(value) + 1 if value in returned else None for value in item['gold_ids']},
        candidate_ids=candidates, candidate_count=len(candidates) if candidates is not None else None,
        candidate_gold_ids=[value for value in item['gold_ids'] if candidates is not None and value in candidates]
            if candidates is not None else None,
        stale_returned_ids=[value for value in returned if value in stale],
        inactive_returned_ids=[value for value in returned if value in inactive],
        ownership_tag_violations=[value for value in returned if value not in records or subjects.get(value) != records[value]['subject']],
        other_subject_ids=[value for value in returned if item.get('expected_subject') and
            subjects.get(value) != item['expected_subject']],
        latency_seconds=elapsed, error=error)
    return row


def evaluate(output=OUTPUT / 'retrieval_report.json'):
    started = time.perf_counter()
    destination = Path(output).resolve()
    if destination.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('Reliability output cannot be under production data')
    fixture = build_fixture()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fixture_path = destination.parent / 'synthetic_fixture.json'
    fixture_text = json.dumps(fixture, ensure_ascii=False, indent=2) + '\n'
    fixture_path.write_text(fixture_text, encoding='utf-8')
    report = dict(schema_version=1, synthetic_only=True,
        reproduction_command=[sys.executable, '-B', 'tools/reliability_eval.py', '--output', str(destination)],
        python_version=sys.version, model_calls=0,
        guards=dict(production_sqlite=True, production_directory=True, unmocked_inference=True),
        live_model=dict(status='NOT RUN',
        reason='Unmocked-inference guard kept enabled; local model availability unverified; no model downloads or status probes.'),
        full_response_latency=dict(status='NOT RUN', samples=0, seconds=None,
            reason='Deterministic retrieval-only evaluator; response behavior covered by isolated regression tests.'),
        unsupported_history_claims=dict(status='NOT RUN', count=None,
            reason='No generated response was evaluated; retrieval success is not response or history accuracy.'),
        fixture_sha256=hashlib.sha256(fixture_text.encode('utf-8')).hexdigest(),
        fixture='synthetic_fixture.json', split_rule=fixture['split_rule'],
        metric_rules=dict(recall='Mean |top5 intersection gold| / |gold| over successfully measured nonempty-gold queries only; execution errors excluded and counted separately.',
            mrr='Mean reciprocal first relevant rank in top5 over the same nonempty-gold denominator; misses zero.',
            no_answer='Empty-gold queries returning any record / all empty-gold queries; actor context can return ten.',
            current_version='All gold current IDs appear in top5 AND no superseded ID is returned anywhere.',
            ownership='Wrong/lost subject tags are structural violations. Other-owner context on owner-scoped queries is counted separately; correct tags alone cannot establish answer misattribution.',
            candidate_coverage='Mean fraction of gold IDs in selector-supplied pool over nonempty-gold queries; lexical path unavailable.'),
        paths=dict(lexical_tool='Existing search_memory(query, limit=5); deterministic lexical retrieval.',
            actor_context_fallback='Existing build_actor_memory_context default limit=10 with selector deliberately unavailable; current selector pool and lexical fallback, no LLM quality claim.',
            mocked_selector_contract='Gold-ID mock selects only supplied gold IDs; measures validation/candidate availability, never semantic quality.'),
        headline_split='held_out', source_sha256={str(path.relative_to(ROOT)).replace('\\', '/'):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__), ROOT / 'tools/conversation_eval.py', ROOT / 'app/memory.py', ROOT / 'app/llm.py')},
        results=[], aggregates={})
    with isolated_runtime() as (memory, store, llm, safety):
        seed_fixture(memory, store, fixture)
        records = {row['id']: dict(row) for row in fixture['records']}
        keys = {row['key']: row['id'] for row in fixture['records']}
        for item in fixture['records']:
            if item.get('supersedes'):
                records[keys[item['supersedes']]]['superseded'] = True
        report['records'] = dict(total=len(records), active=sum(not (row.get('forgotten') or row.get('superseded'))
            for row in records.values()), distractors=128)
        report['experimental_flags'] = {flag: getattr(llm, flag) for flag in
            ('ENABLE_DYNAMIC_SELF', 'ENABLE_RELATIONSHIP', 'ENABLE_SELF_CONSOLIDATION')}
        for path in report['paths']:
            rows = [measure_case(memory, item, path, records) for item in fixture['cases']]
            report['results'].extend(rows)
            report['aggregates'][path] = {split: summarize([row for row in rows if row['split'] == split])
                for split in ('tune', 'held_out')}
        report['guard_violations'] = list(safety._VIOLATIONS)
    report['duration_seconds'] = time.perf_counter() - started
    report['bytecode_disabled'] = sys.dont_write_bytecode
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'retrieval_report.json')
    args = parser.parse_args()
    report = evaluate(args.output)
    print(json.dumps(dict(report=str(args.output), aggregates=report['aggregates']), indent=2))
    return int(any(row['error'] for row in report['results']))


if __name__ == '__main__':
    raise SystemExit(main())
