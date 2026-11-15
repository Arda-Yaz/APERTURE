# Memory acceptance evaluation

This evaluation measures APERTURE's reflection path from observed conversation to an active, pending, or rejected **user memory**. It complements the existing first-response conversation evaluations and retrieval checks. All cases in `evaluations/memory_acceptance/cases.json` are synthetic; no production SQLite data or exported private conversations are fixtures.

The fixture has ten development cases and twenty held-out cases across speaker ownership, unsupported user inference, assistant-only activity claims, temporal updates, and semantic fidelity. The activity cases test whether an assistant's unsupported claim leaks into user memory. They do **not** test whether the actor accurately describes completed tool work; that requires a separate activity evaluation.

## Prepare without inference

From the repository root:

```powershell
.\.venv\Scripts\python.exe -B tools\memory_acceptance_eval.py --output .reliability-tmp\memory-acceptance-prepared.json
```

The default run validates the fixture and writes a preparation report. It does not call Ollama or score model quality.
The command-line output must be inside `.reliability-tmp`. It refuses to replace an existing report unless `--overwrite` is supplied deliberately; use a new filename for a new measurement.

## Optional bounded local-model run

```powershell
.\.venv\Scripts\python.exe -B tools\memory_acceptance_eval.py --live --split dev --max-cases 10 --max-calls 24 --timeout-seconds 120 --output .reliability-tmp\memory-acceptance-dev.json
```

After the implementation and review criteria are fixed, run the untouched evaluation split with `--split holdout --max-cases 20 --max-calls 44` and a different output path. Live mode runs cases sequentially with temporary Memory, Experience, and Goals databases. Each case starts from a fresh synthetic episode. A previous active memory, when needed for an update, is seeded from an explicit prior user observation in a separate synthetic episode. The evaluator calls the real reflection analyzer once and applies that result through the normal reflection write path without a second model analysis. It does not run the interactive actor or dispatch tools. The report records fixture and source hashes, model identity, raw analysis, resulting memory state, provenance, errors, and latency. Case and call limits cap the number of attempts. `--max-seconds` stops admitting model calls after its wall-time budget; Ollama's HTTP timeout applies to each network operation and is not a hard deadline for an entire response.

## Reading results

Read correct creation and correct rejection separately. A system that never saves a memory fails the positive cases even if it rejects every negative case. Pending proposals are counted separately from active memories. Temporal updates must leave one current version and retain the prior record as history. Structural checks verify subject, source event ownership, and target lineage.

The expected decision is a case-level benchmark label. A matching decision alone cannot establish that the model preserved negation, conditions, uncertainty, or attribution in its generated sentence. Each case retains a human review criterion and the report leaves semantic fidelity unreviewed until a person checks the raw output. The held-out labels should remain fixed while development cases guide changes. These synthetic results cannot prove performance on the user's original conversations.
