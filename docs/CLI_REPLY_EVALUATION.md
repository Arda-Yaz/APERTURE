# No-action CLI reply evaluation

The evaluator passes six frozen actor replies through the normal CLI reply
decoder without a tool result or a database. Three replies are explanations
or requests for clarification. Three claim an unobserved action or result.
Every fixture names an explicit file, so the decoder makes an additional
controller call. Use a new report filename for each run.

```powershell
.\.venv\Scripts\python.exe -B tools/cli_reply_eval.py --output .reliability-tmp/cli-reply-offline.json
.\.venv\Scripts\python.exe -B tools/cli_reply_eval.py --live --output .reliability-tmp/cli-reply-live.json
```

The first command uses three scripted controller conditions. The second also
checks the installed default model at `127.0.0.1:11434`, with six calls, no
retries, a 45-second timeout, and a 16-token generation limit. The experiment
uses the production classification prompt. Its timing covers the added
controller call, including the local request, and excludes actor generation,
retrieval, reflection, and other turn processing. Injected timeouts exercise
failure handling immediately and provide no estimate of time spent waiting
for a real timeout.

## 20 November 2026

| Controller condition | Valid replies blocked | Unsupported claims passed | Extra calls | Added call p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| Scripted control | 0/3 | 0/3 | 6 | — |
| Injected timeout | 3/3 | 0/3 | 6 | — |
| Injected malformed output | 3/3 | 0/3 | 6 | — |
| Live `qwen3.5:4b` | 1/3 | 0/3 | 6 | 0.117 s / 0.266 s |

The live controller classified a general explanation of `.txt` as an action
outcome. It let the Markdown explanation and the clarification through.
The live model digest was
`d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`.
The raw report `.reliability-tmp/handoff-next-cli-reply-live.json` records the
fixture and source SHA-256 hashes, per-case classifications, call timings, and
zero isolation violations. Six cases characterize this sample only. The
experiment did not compare a revised validation policy.

## File-operation assessments

A separate five-case smoke check used the same local model and synthetic tool
observations. The controller accepted a successful single-file read, a
two-file read with both observations, and a read with a second file mentioned
as an example. It returned `incomplete` for the two-file read with one
observation and `insufficient_evidence` for the write claim supported only by
a read. These decisions used the structured file-assessment response format.
The local report is `.reliability-tmp/handoff-next-file-controller-structured.json`.

## Follow-up classification comparison

The follow-up freezes twelve replies that name a file, six valid and six
unsupported. It includes general explanations, advice, hypothetical results,
a clarification, an unverified past result, and unsupported current and past
claims. Both runs use the same cases, default local model, temperature 0,
seed 42, 16-token limit, timeout and retry policy.

```powershell
.\.venv\Scripts\python.exe -B tools/cli_reply_eval.py --live --group challenge --output .reliability-tmp/cli-challenge.json
.\.venv\Scripts\python.exe -B tools/cli_reply_eval.py --live --group file-roles --output .reliability-tmp/cli-file-roles.json
```

| Classifier | Valid replies blocked | Unsupported claims passed | Extra calls | Added call p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| Current-source reference with plain output | 4/6 | 0/6 | 12 | 0.105 s / 0.147 s |
| Structured classification | 1/6 | 0/6 | 12 | 0.247 s / 0.328 s |
| Structured classifier, injected timeout | 6/6 | 0/6 | 12 | — |
| Structured classifier, injected malformed response | 6/6 | 0/6 | 12 | — |

Both live rows had zero request errors. A prompt-only trial emitted prose
outside the accepted labels and blocked all six valid replies. The final
classifier requires one JSON field with an enumerated value. The application
accepts only the exact `conversation` classification; a malformed or
unavailable classification still fails closed. The remaining false block
was explicit uncertainty about a past read. A matching historical reference
still returns a report marked as an earlier observation, while an invented
reference is blocked. Those two controls need no classifier call.

The six structured file-role cases used matching seeded output settings
with the controller's effective token ceiling. They returned **0/4**
unsupported successes, **0/2** valid decisions blocked, and **0/11** required
file labels downgraded to `context` or `indirect`. No label was missing and no
request failed. The probe can still detect a wrong semantic label that makes
an unsupported decision pass, as the scripted regression demonstrates.
These counts describe one small synthetic sample.

The comparison raw reports are `.reliability-tmp/handoff-cli-challenge-before.json`,
`.reliability-tmp/handoff-cli-challenge-structured.json`, and
`.reliability-tmp/handoff-cli-file-roles-final.json`. They record source and
fixture SHA-256 hashes, settings and per-case replies.
