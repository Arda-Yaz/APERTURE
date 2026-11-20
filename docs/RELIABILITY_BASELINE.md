# Day 1 retrieval baseline

The baseline uses 48 original fictional queries: T01–T12 are tuning cases and
H01–H36 are held out. The split is fixed before measurement; only held-out
results are headline results. Gold IDs, subjects, fixed timestamps and lifecycle
operations are reproducible from `tools/reliability_eval.py::build_fixture`.
The run saves the complete synthetic fixture beside its JSON report. No existing
evaluation output, conversation, log or production database is an input.

From the repository root, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -B tools/reliability_eval.py --output evaluations/reliability_day1/retrieval_report.json
.\.venv\Scripts\python.exe -B tools/run_isolated_tests.py --pattern test_reliability_eval.py --report evaluations/reliability_day1/evaluator-tests.json
```

The evaluator extends `conversation_eval.synthetic_runtime`; it installs the
existing production SQLite/directory and unmocked Ollama guards before app
imports. Audited lazy modules do not initialize stores at import. Memory,
experience and goals are redirected to registered temporary directories before
measurement. Temporary files use the workspace `.reliability-tmp` scratch parent
because the managed host's default temporary directory is restricted. All three
experimental cognition flags are disabled. A caught guard violation still fails
the run through the existing violation ledger. Temporary databases are removed.

The dataset covers direct matches, paraphrases, Turkish/English translations,
unknown facts, follow-ups, superseded corrections, forgotten records, and user
versus APERTURE subjects. It also includes 128 recent distractors and two oldest
zero-overlap facts, exposing the existing selector candidate pool of at most
96 records and 20,000 serialized UTF-8 bytes. No retrieval algorithm is changed.

Three paths are reported separately:

* `lexical_tool`: actual `search_memory` output, limit five.
* `actor_context_fallback`: actual `build_actor_memory_context`, its default
  limit ten, and a deliberately unavailable selector. This measures candidate
  construction and the existing lexical fallback; it does not measure LLM
  selection quality. The actor normally supplies a model selector.
* `mocked_selector_contract`: a gold-ID mock selects relevant IDs only when they
  are in the supplied candidate pool. This is a candidate-availability and ID
  selection contract measurement. It cannot establish semantic quality.

Macro Recall@5 is the mean fraction of each nonempty gold set found in the first
five returned records. Execution failures are counted separately and excluded
from quality denominators, since they provide no retrieval measurement. MRR@5
uses the same successfully measured nonempty-gold query denominator, with
zero for no relevant top-five result. Empty-gold cases are excluded from both;
their false-positive rate counts queries returning any record, divided by all
empty-gold queries. The actor can return ten, so no-answer errors count any of
those ten. Each rate retains its numerator and denominator. Missing measurements
use `null` and `NOT RUN`, never an invented zero score.

Correction success requires every current gold ID in the first five and no stale
ID anywhere in the returned context. Stale and inactive returned IDs are counted
across all cases. Ownership-tag violations mean a missing, unknown or changed
subject tag. Other-owner records returned on explicitly owner-scoped queries
are counted separately: a correctly tagged APERTURE record does not establish
that a generated answer attributed its contents to the user. No generated answer
or unsupported-history claim is scored by this retrieval-only evaluator.

Every query records IDs, gold ranks, candidate coverage where available, and
elapsed retrieval time. Each path has 12 tuning and 36 held-out latency samples;
fixtures/schema are seeded before timing. The first timed query has no retrieval
warmup and subsequent queries are warm; OS filesystem caching is uncontrolled.
There is no model cost in these timings. Actual LLM retrieval quality and full
response latency are **NOT RUN** because unmocked inference remains guarded;
model availability is unverified. No model download or status probe is performed.

Deterministic final-answer preservation, memory-warning truthfulness and
next-turn recovery are verified separately by the Day 1 response regression
tests. Those assertions are implementation behavior checks, not live response
quality or a claim of production readiness.

The recorded run completed all 144 path/case measurements with zero execution
errors, zero guard violations and zero model calls. The final standalone command
completed in 3.188 seconds of process wall time. It seeded 153 records: 149 active,
two superseded and two forgotten.
The report is `evaluations/reliability_day1/retrieval_report.json`; the complete
original input is `evaluations/reliability_day1/synthetic_fixture.json`.

| Path / split | Recall@5 | MRR@5 | No-answer false positives | Current version | Mean retrieval latency |
| --- | --- | --- | --- | --- | --- |
| Lexical tool / held out | 21/28 = 75.0% | 17.2833/28 = 61.73% | 6/8 = 75.0% | 2/2 | 4.487 ms; n=36 |
| Actor fallback / held out | 21/28 = 75.0% | 17.2833/28 = 61.73% | 6/8 = 75.0% | 2/2 | 6.899 ms; n=36 |
| Mock contract / held out | 25/28 = 89.29% | 25/28 = 89.29% | 0/8 | 2/2 | 2.713 ms; n=36 |
| Lexical tool / tune | 7/11 = 63.64% | 6.2/11 = 56.36% | 1/1 | 1/1 | 4.499 ms; n=12 |
| Actor fallback / tune | 7/11 = 63.64% | 6.2/11 = 56.36% | 1/1 | 1/1 | 6.577 ms; n=12 |
| Mock contract / tune | 10/11 = 90.91% | 10/11 = 90.91% | 0/1 | 1/1 | 2.831 ms; n=12 |

The mock rows are contract evidence only. Candidate coverage was 25/28 held-out
and 10/11 tuning answerable queries, with 96 candidates supplied on every
selector invocation. H09, H35 and H36 missed the held-out candidate pool; H35
and H36 are the deliberately oldest zero-overlap facts. T06 missed on the tuning
split. Candidate coverage is unavailable for the lexical tool path, which has no
selector pool.

All paths returned zero stale/inactive records and preserved subject tags.
Lexical and actor fallback each returned two correctly tagged other-subject
records on the held-out owner-scoped queries (one in tuning); the mock returned
none. These are retrieval relevance errors counted separately from structural
ownership violations. The six held-out no-answer false positives and seven
held-out Recall@5 misses remain part of this baseline; no retrieval tuning was
performed against the fixtures.

The evaluator's eight isolated regression tests passed: exit 0, 8 pass, 0 fail,
0 error, 0 skip, 2.697-second harness duration (2.562 seconds reported by
unittest), with zero guard violations. Exact command and JSON evidence are
`evaluations/reliability_day1/evaluator-tests.json`. The sprint's separate
baseline, response regression and complete-suite reports record their own
commands, counts and timings.
