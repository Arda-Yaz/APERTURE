# Day 1 reliability changes

This sprint measures the existing dirty working tree and adds focused failure
containment, shared inference configuration and safe default dispatch. Retrieval
ranking and paths are unchanged. Experimental cognition defaults remain off.

## Configuration

`app/config.py` is authoritative. Settings are read from the process environment;
no local configuration file or user override is rewritten. Role names are
`ACTOR`, `RETRIEVAL`, `REFLECTION`, `SELF_STATE`, `RELATIONSHIP`, `TASK_CONTROLLER`.

| Setting | Precedence and default |
| --- | --- |
| Model | `APERTURE_MODEL_<ROLE>` then `APERTURE_MODEL` then `qwen3.5:4b` |
| Endpoint | `APERTURE_OLLAMA_HOST` then compatible `OLLAMA_HOST` then `http://127.0.0.1:11434` |
| Timeout | `APERTURE_INFERENCE_TIMEOUT_SECONDS_<ROLE>` then global name without suffix then 120 seconds; finite 0.01–3600 |
| Retries | `APERTURE_INFERENCE_RETRIES_<ROLE>` then global then 0; integers 0–2; actor always 0 and explicit nonzero actor override rejected |
| Output ceiling | `APERTURE_INFERENCE_NUM_PREDICT_<ROLE>` then global then actor 4096 / internal 2048; integers 1–32768; a smaller positive integer caller limit is preserved |

Invalid settings fail explicitly without echoing the supplied value. Startup
prints resolved settings for every role. Endpoint credentials, path, query and
fragment are removed from diagnostics. `MODEL` and patchable `ollama_chat` module
symbols remain available for existing callers and test doubles.

The synchronous wrapper sets finite HTTPX operation timeouts, disables redirects,
closes clients, and retries only recognized transport/server failures. A timeout
is an HTTP operation/inactivity timeout, not a hard wall-clock deadline for a
server that keeps transmitting bytes. Missing models and malformed responses
fail immediately. There are no model downloads or automatic fallback models.
An internal call makes at most three transport attempts when explicitly configured;
the default is one. The actor uses one transport attempt per step and retains its
eight-step bound and one incomplete-generation recovery within that bound.

## Tool profiles

Default `APERTURE_TOOL_PROFILE=memory` selects `MEMORY_ASSISTANT`:

- Memory save/search/forget and recent activity retain existing ownership,
  explicit self-memory request and provenance behavior.
- Scoped discovery/source/search/snapshot are allowed only with a structured
  absolute root contained within `APERTURE_WORKSPACE` (default repository root).
  Existing workspace traversal, sensitive-path and link checks still apply.
- Scoped writes retain active-goal checks and individual confirmation.
- General reads/listing/writes, terminal execution and launching applications are
  denied before tool execution. Unknown tools fail closed without a prompt.

Set `$env:APERTURE_TOOL_PROFILE = 'advanced'` explicitly before startup to restore
AUTONOMOUS general tools; BALANCED and SAFE remain supported by the permission
module for existing callers. Advanced general tools have host access. A terminal
working directory and command blocklist do not provide filesystem containment.
The scoped tools' checks are not race-proof against hostile filesystem changes.

## Failure behavior

Actor connection, timeout, missing-model and malformed SDK responses produce an
explicit unsuccessful outcome; they do not become assistant observations or
trigger reflection. Actor programming errors and process interrupts propagate.

Once the actor completes an answer, ordinary answer-history, reflection or
experimental cognition failures return that answer once with
`[WARNING] Memory updating failed (...)`; the next CLI turn remains available.
The warning includes a sanitized failure category, never raw model output or
exception text. A best-effort system event records it. Malformed reflection JSON
raises an explicit failure; legitimate null/no-memory results do not. Failed
memory writes cannot produce `MEMORY_SAVED`. Writes that succeeded before a later
postprocessing failure may remain persisted; the warning says updates may be
incomplete. There is no new worker, transaction spanning all stores, or retry of
partially completed reflection. Failure to record the original observed answer
skips cognition and stops bounded execution before accepting satisfaction, while
still returning the answer with a warning. Bounded runs return their final answer
to the CLI for one display, alongside existing progress diagnostics.

## Reproduction and evidence

From the repository root, use the existing environment. No dependencies are
installed and no interactive application or live model is launched by these runs.
The numerical Day 1 results below describe historical local artifacts. To
verify a source checkout, write new reports to `.reliability-tmp/`:

```powershell
.\.venv\Scripts\python.exe -B tools/run_isolated_tests.py --report .reliability-tmp/reliability-tests.json
.\.venv\Scripts\python.exe -B tools/reliability_eval.py --output .reliability-tmp/day1-retrieval.json
```

```powershell
.\.venv\Scripts\python.exe -B tools/run_isolated_tests.py --report evaluations/reliability_day1/baseline-tests.json
.\.venv\Scripts\python.exe -B tools/run_isolated_tests.py --pattern 'test_reliability*.py' --report evaluations/reliability_day1/targeted-tests.json
.\.venv\Scripts\python.exe -B tools/reliability_eval.py
.\.venv\Scripts\python.exe -B tools/run_isolated_tests.py --report evaluations/reliability_day1/final-tests.json
git diff --check
```

The first command describes the captured pre-application-change baseline; rerunning
it after changes measures the current suite. Do not overwrite the captured
baseline when comparing a later run.

The runner installs the existing production-SQLite, production-directory and
unmocked-inference guards **before** importing lazy application stores. It
redirects all three lazy stores (memory, experience and goals) to newly created
registered temporary directories. Constructor defaults remain inert guard
sentinels so existing guard-probe tests still validate fail-closed behavior.
`-B`, `sys.dont_write_bytecode` and `PYTHONDONTWRITEBYTECODE=1` prevent bytecode
writes; experimental flags are set to zero. Fixtures use synthetic data only.
Temporary storage uses `.reliability-tmp` under this authorized workspace because
the host's default sandbox Temp failed with SQLite unable-to-open and WinError 5
during the initial setup attempt. No tests ran in that setup attempt, and no
production storage was accessed. No restrictions or guards were relaxed.

The baseline completed with 170 tests: 167 passed, 2 failed, 1 errored, 0 skipped;
exit 1, 24.962114 seconds including setup/discovery. The two failures are fixture
`mklink /J` returning exit 1 with access denied, and the error is fixture
`os.link` raising WinError 5. Each occurs before the application's boundary
assertions; those filesystem boundary assertions remain unverified here.
There were zero unexpected isolation guard violations. See the JSON reports for
exact test identifiers, tracebacks, commands and timings.

The captured baseline had `PYTHONDONTWRITEBYTECODE` unset; `-B` and
`sys.dont_write_bytecode=True` still disabled bytecode. Experimental environment
overrides were unset (off by default). Later runner invocations explicitly set
the bytecode variable to 1 and experimental flags to 0 inside the child process.
No parent/global environment settings were changed.

Final verification used the exact commands above. Harness timings include setup
and discovery; unittest's own narrower duration may differ.

| Run / JSON report | Passed | Failed | Errored | Skipped | Exit | Harness seconds |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline-tests.json` | 167 | 2 | 1 | 0 | 1 | 24.962114 |
| `targeted-tests.json` (`test_reliability*.py`) | 38 | 0 | 0 | 0 | 0 | 4.781812 |
| `evaluator-compatibility-tests.json` (`test_*conversation_eval.py`) | 18 | 0 | 0 | 0 | 0 | 4.878915 |
| `final-tests.json` | 205 | 2 | 1 | 0 | 1 | 29.830519 |

The compatibility command was:

```powershell
.\.venv\Scripts\python.exe -B tools/run_isolated_tests.py --pattern 'test_*conversation_eval.py' --report evaluations/reliability_day1/evaluator-compatibility-tests.json
```

All runs have zero unexpected guard violations and no skips. The first broader
verification (`verification-first-pass.json`) recorded 208 tests, 201 passed,
3 failed, 4 errored, exit 1, 28.745968 seconds. It exposed an introduced evaluator
adapter signature mismatch and outdated broad-tool schema expectations; these
were corrected with structured argument forwarding, explicit safe-schema
assertions and a local advanced-mode virtual-read fixture. No real host tool was
executed. The final suite retains only the three baseline environment failures.
No test was skipped or weakened to bypass the filesystem fixture restrictions.

`tests/test_reliability_runtime.py` exercises actual CLI output across two turns
after reflection exception/timeout, real malformed reflection processing,
failed writes, missing actor and subsequent recovery, interrupt propagation,
safe/unknown/out-of-workspace dispatch and authorized memory writes. Transport
and role inheritance/override tests live in `test_reliability_config.py`.

Retrieval cases, definitions, split results, candidate coverage and measurement
limits are in [the synthetic baseline](RELIABILITY_BASELINE.md) and
`evaluations/reliability_day1/retrieval_report.json`. These results are measurement,
not evidence of improved retrieval or production semantic quality. Live-model
quality, live full-response latency and unsupported-history claims were outside
the Day 1 runs. The separate [synthetic turn latency evaluation](TURN_LATENCY_EVALUATION.md)
documents the later local measurement.

## Scope and existing work

The starting tree already contained changes in experience, memory, actor context,
reflection, self-state, evaluator tooling and conversation tests, plus modified
tracked bytecode and unrelated reports. Those changes were preserved. Day 1 adds
only configuration/client boundaries, final-answer postprocessing containment,
permission enforcement, synthetic evaluation, regression tests and documentation.
No data migration, retrieval unification, ranking replacement, embedding backend,
new memory CLI or expanded experimental cognition was added. No files were staged,
committed or pushed, and no real user databases/conversations were opened or used
as fixtures.

Files changed by this sprint (some already contained unrelated changes):

- Application: `app/config.py`, new `app/inference.py`, `app/llm.py`,
  `app/main.py`, `app/permissions.py`, `app/goal_execution.py`, `app/reflection.py`,
  `app/self_state.py`, `app/relationship.py`, `app/task_controller.py`.
- Evaluation and harness: new `tools/run_isolated_tests.py`, new
  `tools/reliability_eval.py`, and the existing untracked
  `tools/english_conversation_eval.py` (permission adapter only).
- Tests: new `tests/test_reliability_config.py`, `test_reliability_runtime.py`,
  `test_reliability_eval.py`; existing `tests/test_workspace_write_execution.py`
  (structured permission assertion) and pre-existing untracked
  `tests/test_conversation_eval.py`, `tests/test_english_conversation_eval.py`
  (safe-default schema / explicit virtual advanced-mode expectations).
- Documentation: `README.md`, new `docs/RELIABILITY_DAY1.md`, new
  `docs/RELIABILITY_BASELINE.md`.
- Synthetic artifacts: `evaluations/reliability_day1/synthetic_fixture.json`,
  `retrieval_report.json`, `baseline-tests.json`, `targeted-tests.json`,
  `evaluator-tests.json`, `evaluator-compatibility-tests.json`,
  `verification-first-pass.json`, `final-tests.json`.

The pre-existing `app/memory.py`, `app/experience.py`, `app/context_budget.py`,
tracked bytecode, private reports and real local data were not edited by Day 1.
