# Synthetic retrieval evaluation

From the repository root, with the pinned dependencies installed, run:

```powershell
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --output .reliability-tmp/retrieval-report.json
```

The offline run makes zero model calls. It seeds one temporary SQLite store
from the [historical corpus](../evaluations/retrieval/baseline.json) and
measures the H01–H36 historical group and the [F01–F10 fresh
group](../evaluations/retrieval/fresh.json) there. It uses separate stores for
the [I01–I08 independent group](../evaluations/retrieval/independent.json)
and the [P01–P08 pressure holdout](../evaluations/retrieval/pressure.json).
The report identifies the corpus for each group in `corpus_by_group`.
The twelve original tuning queries remain in the historical fixture to
preserve its identity. The SQLite and inference guards apply throughout the run.
Choose a fresh output filename for each run.

Each group reports three paths:

- **Direct lexical tool.** `search_memory` ranks active memories by weighted
  overlap with the request and returns up to five.
- **Actor fallback.** The actor builds the usual candidate pool. The evaluator
  forces selector unavailability, so the store falls back to the query-only
  lexical ranking and returns up to ten records.
- **Actor selector.** The offline selector is given the gold IDs and can return
  only those in the supplied pool. Its results measure candidate coverage and
  ID validation. In live mode, APERTURE's selector supplies the IDs.

Recall@5 averages the fraction of gold IDs among the first five results for
each answerable query. MRR@5 uses the reciprocal rank of the first correct
result. A false positive is any result for an empty-gold query. The report
includes per-case candidate and return IDs, ownership tags, inactive returns,
selector errors, source SHA-256 hashes, and canonical fixture SHA-256 hashes.
Live selector errors retain the returned fallback IDs in the case record and
are excluded from the existing selector quality averages. The
`delivered_results` metrics score all queries, including the fallback IDs
returned on error. Check the completed and error counts and both sets of
denominators when comparing rates.

## Offline baseline

| Group | Path | Recall@5 | MRR@5 | Empty-gold false positives | Candidate gold coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| Historical (36) | Direct lexical | 21/28 | 0.617 | 6/8 | — |
| Historical (36) | Actor fallback | 21/28 | 0.617 | 6/8 | 28/28 |
| Historical (36) | Actor gold-ID mock | 28/28 | 1.000 | 0/8 | 28/28 |
| Fresh (10) | Direct lexical | 3/6 | 0.333 | 4/4 | — |
| Fresh (10) | Actor fallback | 3/6 | 0.333 | 4/4 | 6/6 |
| Fresh (10) | Actor gold-ID mock | 6/6 | 1.000 | 0/4 | 6/6 |
| Independent (8) | Direct lexical | 2/4 | 0.500 | 4/4 | — |
| Independent (8) | Actor fallback | 2/4 | 0.500 | 4/4 | 2/4 |
| Independent (8) | Actor gold-ID mock | 2/4 | 0.500 | 0/4 | 2/4 |

The original 54 cases completed for each path with zero ownership tag violations
or inactive returns. The independent group has 112 records, including 100
recent entries owned by the user and older pottery and printmaking facts
owned by the same user. The two older gold records share zero query tokens
and fell outside the candidate pool. All four strongly overlapping
unanswered queries returned fallback records. These observations identify
two distinct failure modes on frozen cases. They provide a baseline for
comparing cutoff and candidate policies before a production change.

## Local model evaluation

With Ollama already serving an installed model on the loopback endpoint, use
`--live` explicitly. An example bounded to the ten fresh cases:

```powershell
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --live --group fresh --model qwen3.5:4b --output .reliability-tmp/retrieval-live-fresh.json
```

Live mode uses the production retrieval prompt and parser with fixed
experimental transport settings. It calls the local Ollama SDK once per case
without retries, with a 45-second timeout and the prompt's requested
512-token generation limit. The report records the actual options and model
digest. The `runtime_retrieval_settings` snapshot records the configured
runtime model, timeout, retry count, token ceiling, and effective token limit
after applying the ceiling. Runtime retrieval goes through `app/inference.py`
and applies those role settings. The fixed conditions let runs of this
evaluation be compared across source versions.

### Fresh group, 20 November 2026

The fresh group completed 10/10 cases in 10 calls with zero selector errors
or isolation violations. The evaluation used `qwen3.5:4b` with digest
`d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`.

| Path | Recall@5 | MRR@5 | Empty-gold false positives |
| --- | ---: | ---: | ---: |
| Actor selector, successful cases | 6/6 | 1.000 (6/6) | 2/4 |
| Actor selector, delivered results | 6/6 | 1.000 (6/6) | 2/4 |
| Forced actor fallback | 3/6 | 0.333 (2/6) | 4/4 |

The two selector false positives were F08, where the Tuesday backup
schedule did not establish retention duration, and F10, where a prior
review schedule was selected after the topic changed. This small group
tests one synthetic record corpus. The run measures retrieval results for
the actor's context. The raw report, written locally as
`.reliability-tmp/handoff-next-retrieval-live.json`, contains all source
hashes, the fresh fixture SHA-256
`1621da914d23ca99688a7df2d5c28a01d43f8a6c58d1302f4840e67489586786`,
and per-query output. The observed runtime settings used a 120-second
timeout, zero retries, and a ceiling of 2,048 tokens. The effective token
limit was 512 for both paths.

### Independent corpus, 20 November 2026

```powershell
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --live --group independent --model qwen3.5:4b --output .reliability-tmp/handoff-independent-retrieval-live.json
```

The eight cases completed in eight calls with zero selector errors and
zero isolation violations. The live selector and delivered results both
had Recall@5 of 2/4, MRR@5 of 0.500 (2/4), and false positives on 2/4
unanswered queries. I01 and I02 missed the older same-owner gold records
at candidate collection; I03 and I04 selected the expected record first.
The selector returned the loan record for the missing fee in I06 and the
tide gauge record for the missing manufacturer in I07. Forced lexical
fallback returned records on all four unanswered queries.

The run used the same model digest and fixed transport settings as the
fresh live run. The raw JSON written by the command records every source
hash, candidate, and return ID. The frozen independent fixture SHA-256 is
`e9f13a006594c288fc1d65978751cdcbbc8e91c3e0ddc05e5d850cbddfa65b05`.
This second corpus exposes a gap in candidate coverage for crowded
same-owner histories. Its eight queries establish a baseline for testing
a change to candidate selection or the lexical fallback.

### Historical group

Use `--group historical` to run the original comparison group.

The earlier live experiment used the historical fixture (canonical SHA-256
`904fba12cece60c9a3c1b42b77e68d66ce8a66d33ed551a576bd34011fab91c3`).
With 36 calls to `qwen3.5:4b` (digest
`d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`),
it reported Recall@5 of 28/28, MRR@5 of 0.982, and 1/8 empty-gold false
positives. Those cases had already been seen in development. The experiment
used earlier versions of `app/memory.py` (SHA-256
`56f34464a3622e67223890f1f5ed7fdd445bc175a18a661847851a495cb45a0c`)
and `app/llm.py` (SHA-256
`0398136812d11a3f6721b5a5ae2b6686fe988efb5a9b07bd09e58c65311f292c`).

## Candidate age and fallback overlap experiments

The pressure holdout uses 134 records and eight new queries, frozen before
policy tuning. Its canonical SHA-256 is
`62c1306a37421705095067fa9f7336e2e19a43b2ed647ef902d62a319d16ef01`.
It includes two old, zero-overlap records under heavy same-owner history,
one strong and one sparse answerable lexical match, and four unanswered
queries. The reports separate old-record candidate coverage from live
Recall@5, MRR@5, unanswered false positives, and selector call cost.

For candidate age, the reserve places the oldest active
records from each owner with more than 96 active rows ahead of the usual
interleaved history, after current and dialogue lexical candidates. Candidate
count and payload ceilings remain 96 records and 20,000 UTF-8 bytes. On the
known independent tuning group, reserving zero, one, or two rows per crowded
owner found respectively 0/2, 1/2, and 2/2 old zero-overlap gold IDs, while
total candidate count stayed at 768 across eight queries. The selected
two-row reserve was then compared with the zero-row baseline on the
pressure holdout.

```powershell
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --group independent --history-older-per-owner 0 --output .reliability-tmp/pressure-tuning-independent-0.json
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --group independent --history-older-per-owner 1 --output .reliability-tmp/pressure-tuning-independent-1.json
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --group independent --history-older-per-owner 2 --output .reliability-tmp/pressure-tuning-independent-2.json
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --group pressure --history-older-per-owner 0 --output .reliability-tmp/pressure-holdout-offline-0.json
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --group pressure --history-older-per-owner 2 --output .reliability-tmp/pressure-holdout-offline-2.json
```

Holdout old-record coverage rose from 0/2 to 1/2 and overall coverage from
2/4 to 3/4. The reserve found P01. P02 sits deeper in the old history and
remained outside the pool. The strongly matched P03 stayed in the pool
and at rank one with the gold-ID mock. Across eight holdout queries each
variant supplied 768 records. The total candidate payload was 117,280 bytes
for the baseline and 117,248 bytes with the reserve.

### Paired live pressure result

The paired experiment used the installed `qwen3.5:4b` digest recorded above
and fixed local transport settings of 45 seconds, zero retries, and 512
generation tokens. Each run completed eight cases and made eight calls
with zero selector errors and zero isolation violations. These commands
recreate the paired conditions with the current runner.

```powershell
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --live --group pressure --history-older-per-owner 0 --model qwen3.5:4b --output .reliability-tmp/pressure-holdout-live-0.json
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --live --group pressure --history-older-per-owner 2 --model qwen3.5:4b --output .reliability-tmp/pressure-holdout-live-2.json
```

| Live condition | Recall@5 | MRR@5 | Strong lexical Recall@5 | Unanswered false positives | Selector time |
| --- | ---: | ---: | ---: | ---: | ---: |
| No reserve | 2/4 | 0.500 | 1/1 | 1/4 | 10.96 s over 8 calls |
| Two oldest records per crowded owner | 3/4 | 0.750 | 1/1 | 1/4 | 11.16 s over 8 calls |

The delivered results equal the successful selector results in both runs.
The reserve recovered P01 and preserved the IDs selected for every other
case. The sparse match P04 stayed at rank one. Both runs' sole false
positive was P06, a loan record returned for a question about duration.
The small timing sample gives insufficient evidence of a latency change.

The two-row reserve is the production default. Across a zero-inference
comparison of all 62 cases, candidate selection matched the experimental
two-row run. The historical and fresh gold-ID results matched the zero-row
baseline. The independent group recovered I01 and I02 and the pressure
group recovered P01. Both variants retained all earlier gold-ID results,
the candidate budgets, and active-record and ownership validation. The
bounded reserve still misses deeper records such as P02.

### Lexical fallback abstention

The experiment takes the actual query-only fallback results and replays four
simple abstention rules on the top-ranked record's overlap. A rule had to
suppress false positives and retain every correct top-five ID on the
combined fresh and independent tuning sets before it could advance to the
pressure holdout. Correct IDs lost here are measured against the IDs
already delivered by the fallback. The script reports full Recall@5,
MRR@5, false-positive denominators, and per-case abstentions.

```powershell
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --group fresh --history-older-per-owner 0 --output .reliability-tmp/pressure-tuning-fresh.json
.\.venv\Scripts\python.exe -B tools/retrieval_fallback_eval.py --tuning-fresh .reliability-tmp/pressure-tuning-fresh.json --tuning-independent .reliability-tmp/pressure-tuning-independent-0.json --holdout .reliability-tmp/pressure-holdout-offline-0.json --output .reliability-tmp/pressure-fallback-comparison.json
```

| Abstention rule | Tuning false positives | Tuning correct IDs lost | Holdout false positives | Holdout correct IDs lost |
| --- | ---: | ---: | ---: | ---: |
| Existing fallback | 8/8 | 0/5 | 4/4 | 0/2 |
| At least two shared tokens | 7/8 | 1/5 | 4/4 | 1/2 |
| At least three shared tokens | 6/8 | 2/5 | 4/4 | 1/2 |
| At least a quarter of query terms | 7/8 | 2/5 | 4/4 | 0/2 |
| At least half of query terms | 4/8 | 3/5 | 4/4 | 1/2 |

No overlap gate qualified on the tuning set. On the holdout, all gates
still returned lexical records for the four unanswered queries, and three
gates dropped the sparse correct match P04. The existing fallback was kept.
