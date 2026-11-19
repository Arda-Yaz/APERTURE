# Synthetic retrieval evaluation

From the repository root, with the pinned dependencies installed, run:

```powershell
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --output .reliability-tmp/retrieval-report.json
```

The offline run makes zero model calls. It seeds a temporary SQLite store
from the [historical corpus and queries](../evaluations/retrieval/baseline.json)
and measures both the H01–H36 historical group and the [F01–F10 fresh
group](../evaluations/retrieval/fresh.json). The fresh queries reuse the same
corpus and were fixed before this evaluation. The twelve original tuning
queries remain in the historical fixture to preserve its identity. The
SQLite and inference guards apply throughout the run.
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
are excluded from selector quality averages. Check the completed and error
counts when comparing rates.

## Offline baseline

| Group | Path | Recall@5 | MRR@5 | Empty-gold false positives | Candidate gold coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| Historical (36) | Direct lexical | 21/28 | 0.617 | 6/8 | — |
| Historical (36) | Actor fallback | 21/28 | 0.617 | 6/8 | 28/28 |
| Historical (36) | Actor gold-ID mock | 28/28 | 1.000 | 0/8 | 28/28 |
| Fresh (10) | Direct lexical | 3/6 | 0.333 | 4/4 | — |
| Fresh (10) | Actor fallback | 3/6 | 0.333 | 4/4 | 6/6 |
| Fresh (10) | Actor gold-ID mock | 6/6 | 1.000 | 0/4 | 6/6 |

All 46 cases completed for each path with zero ownership tag violations or
inactive returns. The fresh unanswered queries probe token overlap with
unrelated facts. Scoring a fallback cutoff would require comparing the
reduction in false positives with the loss of relevant results. The measured
baseline uses the existing fallback.

## Local model evaluation

With Ollama already serving an installed model on the loopback endpoint, use
`--live` explicitly. An example bounded to the ten fresh cases:

```powershell
.\.venv\Scripts\python.exe -B tools/retrieval_eval.py --live --group fresh --model qwen3.5:4b --output .reliability-tmp/retrieval-live-fresh.json
```

Live mode uses the production retrieval prompt and parser with the local
Ollama client. It logs the model digest and request settings, and allows
one retrieval call per case with a 45-second timeout per call. Evaluate
`--group historical` separately for comparison with the earlier experiment.

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
The current baseline was run offline. Current live selector quality still
needs measurement.
