"""Score lexical fallback abstention gates on frozen retrieval reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from retrieval_eval import ROOT, SCRATCH, canonical_hash, rate, result_quality


GATES = {
    'baseline': ('shared_tokens', 0),
    'two_shared_tokens': ('shared_tokens', 2),
    'three_shared_tokens': ('shared_tokens', 3),
    'quarter_query_overlap': ('query_fraction', 0.25),
    'half_query_overlap': ('query_fraction', 0.5),
}


def tokens(text):
    return set(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def compare(rows):
    # Only decide whether to abstain on the full delivered lexical result.
    # Its ranking and per-query result limits come from the production store.
    reports = {}
    gold_hits = [set(row['returned_ids'][:5]) & set(row['gold_ids']) for row in rows]
    negatives_with_results = sum(bool(row['returned_ids']) and not row['gold_ids'] for row in rows)
    for name, (feature, minimum) in GATES.items():
        delivered = [
            dict(row, returned_ids=[]) if row[feature] < minimum else row
            for row in rows
        ]
        lost = sum(len(previous.difference(row['returned_ids'][:5]))
                   for previous, row in zip(gold_hits, delivered))
        suppressed_negatives = sum(bool(before['returned_ids']) and not before['gold_ids']
                                   and not after['returned_ids']
                                   for before, after in zip(rows, delivered))
        reports[name] = {
            **result_quality(delivered),
            'lost_correct_ids_at_5': rate(lost, sum(map(len, gold_hits))),
            'suppressed_false_positives': rate(suppressed_negatives, negatives_with_results),
            'abstained_case_ids': [before['id'] for before, after in zip(rows, delivered)
                                   if before['returned_ids'] and not after['returned_ids']],
        }
    return reports


def load_groups(path, required_groups):
    raw = path.read_bytes()
    report = json.loads(raw)
    if (report.get('synthetic_only') is not True or report.get('model_calls') != 0
            or report.get('guard_violations') or set(report['groups']) != set(required_groups)):
        raise ValueError(f'Expected an isolated, offline report for {required_groups}.')

    rows = []
    for group, paths in report['groups'].items():
        corpus = report['corpus_by_group'][group]
        if corpus not in {'evaluations/retrieval/baseline.json', 'evaluations/retrieval/independent.json',
                          'evaluations/retrieval/pressure.json'}:
            raise ValueError(f'Unexpected corpus in {group}.')
        fixture = json.loads((ROOT / corpus).read_text(encoding='utf-8'))
        if canonical_hash(fixture) != report['fixture_sha256'][corpus]:
            raise ValueError(f'Corpus digest mismatch for {group}.')
        records = {record['id']: record for record in fixture['records']}
        rows_by_path = paths['actor_fallback']
        if rows_by_path['metrics']['completed_measurements'] != len(rows_by_path['results']):
            raise ValueError(f'Incomplete fallback report for {group}.')
        for row in rows_by_path['results']:
            query = tokens(row['query'])
            returned = row['returned_ids']
            shared = len(query & tokens(records[returned[0]]['content'])) if returned else 0
            rows.append(dict(row, shared_tokens=shared, query_fraction=shared / len(query) if query else 0))
    provenance = {
        'sha256': hashlib.sha256(raw).hexdigest(),
        'source_sha256': report['source_sha256'],
        'fixture_sha256': report['fixture_sha256'],
    }
    return rows, provenance


def evaluate(tuning_fresh, tuning_independent, holdout, output):
    fresh, fresh_provenance = load_groups(tuning_fresh, ('fresh',))
    independent, independent_provenance = load_groups(tuning_independent, ('independent',))
    tuning = compare(fresh + independent)
    eligible = [name for name in GATES if tuning[name]['lost_correct_ids_at_5']['numerator'] == 0
                and tuning[name]['suppressed_false_positives']['numerator'] > 0]
    chosen = max(eligible, key=lambda name: tuning[name]['suppressed_false_positives']['numerator'],
                 default=None)

    frozen, frozen_provenance = load_groups(holdout, ('pressure',))
    provenances = [fresh_provenance, independent_provenance, frozen_provenance]
    if any(item['source_sha256'] != fresh_provenance['source_sha256'] for item in provenances):
        raise ValueError('Run the tuning and holdout inputs from the same sources.')
    validation = compare(frozen)
    recommend = bool(chosen and validation[chosen]['lost_correct_ids_at_5']['numerator'] == 0
                     and validation[chosen]['suppressed_false_positives']['numerator'] > 0)
    report = {
        'schema_version': 1,
        'synthetic_only': True,
        'model_calls': 0,
        'policy_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': dict(fresh=fresh_provenance, independent=independent_provenance, pressure=frozen_provenance),
        'tuning': tuning,
        'holdout': validation,
        'chosen_from_tuning': chosen,
        'meets_both_splits': recommend,
    }
    output = output.resolve()
    if output.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('Evaluation output cannot be under production data.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tuning-fresh', type=Path, required=True)
    parser.add_argument('--tuning-independent', type=Path, required=True)
    parser.add_argument('--holdout', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=SCRATCH / 'retrieval-fallback-comparison.json')
    args = parser.parse_args()
    report = evaluate(args.tuning_fresh, args.tuning_independent, args.holdout, args.output)
    print(json.dumps({key: report[key] for key in ('chosen_from_tuning', 'meets_both_splits', 'tuning', 'holdout')},
                     indent=2))


if __name__ == '__main__':
    main()
