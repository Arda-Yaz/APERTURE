# Synthetic turn latency

Run from the repository root with a new output path:

```powershell
.\.venv\Scripts\python.exe -B tools/turn_latency_eval.py --samples 1 --output .reliability-tmp/turn-latency-offline.json
.\.venv\Scripts\python.exe -B tools/turn_latency_eval.py --live --samples 4 --output .reliability-tmp/turn-latency-live.json
```

The offline command drives the turn with scripted responses and checks the
measurement wiring. The live command uses the already installed source-default
model at the local loopback endpoint. It interleaves three turn conditions at
each of two initial sizes: an empty store and 149 active records from the
frozen [synthetic corpus](../evaluations/retrieval/baseline.json). Each turn gets
a fresh temporary SQLite store and one synthetic user message. The experiment
disables dispatch and rejects unsolicited actor tool calls. Reflection starts either below its usual threshold or one turn
before it. A separate below-threshold condition asks about a named file's
format to exercise the extra CLI reply check without performing an action.

The report records the model digest, fixture and source hashes, call counts,
wall-clock seconds and failures. The local experiment uses zero retries, a
45-second request timeout, and fixed output ceilings of 2,048 actor, 512
retrieval, 1,024 reflection and 16 CLI-controller tokens. Those ceilings are
experiment settings. Eight calls per turn and an eight-minute between-call
deadline bound the run. Timing excludes database seeding and episode setup.

Per-role p50 and p95 describe the summed inference duration on turns where that
role ran. Retrieval inference includes any selection triggered by reflection.
The inclusive retrieval-context and reflection spans also include local work
and can overlap the inference timings. The p95 uses nearest rank, so with four
samples it is the largest observation. This experiment does not estimate a
production tail percentile or a cold model start.

## 20 November 2026

The local `qwen3.5:4b` run recorded all 24 turn attempts and made 68 Ollama
calls. Entries below are seconds, shown as p50 / p95. Parentheses give the
number of measured call attempts for the role. Each row contains four turns.

| Active records | Condition | Whole turn | Actor | Retrieval | Reflection | CLI check |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Ordinary | 13.83 / 19.76 | 13.80 / 19.73 (4) | — | — | — |
| 0 | Reflection due | 18.15 / 35.71 | 16.15 / 33.92 (5) | — | 1.71 / 2.25 (8) | — |
| 0 | File discussion | 12.07 / 40.48 | 11.87 / 40.47 (5) | — | — | 0.17 / 0.17 (3) |
| 149 | Ordinary | 31.63 / 42.02 | 30.50 / 40.66 (7) | 1.33 / 1.34 (4) | — | — |
| 149 | Reflection due | 14.00 / 39.03 | 10.71 / 35.80 (5) | 1.60 / 1.69 (8) | 1.58 / 1.65 (8) | — |
| 149 | File discussion | 14.29 / 41.84 | 12.82 / 40.84 (5) | 1.35 / 1.36 (4) | — | 0.17 / 0.18 (3) |

There were ten actor generations that reached the token cap. Three turns
stopped after the bounded recovery attempt. Five file discussion turns returned
an unverified reply. In one of those turns the experiment deadline blocked the
last CLI check before the SDK call, so 69 measured attempts yielded 68 calls.
The table includes those outcomes. The process reported failure status for
the observed errors. There were zero isolation violations and zero attempted
tool dispatches.

Actor generation consumed 440.36 of 471.27 measured inference seconds (93.4%),
including retries. The inclusive reflection span had a p95 of 3.15 seconds at
149 records. This sample points to actor generation and truncation as the main
sources of delay. It provides no basis for moving reflection to the background
or changing retrieval indexes. The later stages also depend on the actor
producing a usable reply.

The local raw report is `.reliability-tmp/handoff-turn-latency-guarded-live.json`.
It records fixture SHA-256
`7614f2d08a3e1a85accd06c8c1df3dec00c85a9c5ce01da094b83f869245d2bd`
and model digest
`d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`.

## Runtime token-budget comparison

```powershell
.\.venv\Scripts\python.exe -B tools/turn_latency_eval.py --live --budget runtime --samples 4 --output .reliability-tmp/turn-latency-runtime.json
```

`--budget runtime` reads the configured role ceilings and applies the same
minimum of caller request and role ceiling as `app/inference.py`. The report
records both configured and effective limits and keeps the same synthetic
turns and store sizes. The SDK transport remains a bounded experiment with a
45-second timeout, zero retries, and a twelve-minute between-call deadline.
The ordinary actor loop still owns its single incomplete-generation retry.
This reproduces the runtime token budget under a fixed local transport.

Both runs used the same fixture hash and local model digest. The runtime
comparison recorded effective limits of 4,096 actor, 512 retrieval, 2,048
reflection, and 16 CLI-controller tokens. The file classifier source hash
matched between the two timing runs.

| Measure | Earlier fixed experiment | Runtime token budget |
| --- | ---: | ---: |
| Turn attempts | 24 | 24 |
| Ollama calls | 68 | 65 |
| Turns that returned an answer | 16 | 18 |
| Unverified replies | 5 | 6 |
| Stopped actor turns | 3 | 0 |
| Actor generations reaching token limit | 10 | 0 |
| Actor retries | 7 | 1 |
| Whole-turn p50 / p95 | 16.27 / 41.84 s | 14.15 / 31.46 s |
| Actor inference total | 440.36 s | 338.99 s |
| Retrieval inference total | 16.34 s | 17.43 s |
| Reflection inference total | 13.72 s | 14.53 s |
| CLI-controller inference total | 0.85 s | 1.57 s |

The earlier run used different actor and reflection ceilings and reached
its between-call deadline once. These are two small, separate samples with
uncontrolled model warmth and actor sampling. The comparison confirms that
the ten actor truncations did not recur under the existing runtime budget.
It provides no quantitative estimate of the effect of raising any one
ceiling or of production latency. The remaining unverified replies are an
outcome distinct from a truncated actor generation.

The local comparison report is
`.reliability-tmp/handoff-runtime-budget-latency.json`. It contains per-group
p50 and p95 values, role call counts, and zero isolation violations.
