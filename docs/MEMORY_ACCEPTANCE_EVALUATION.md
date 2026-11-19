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

## Compare approval policies

Run both conditions against the same local model and fixture. The published cases
serve as a regression set. For a new evaluation, freeze its cases and review
criteria before tuning.

```powershell
$approvalBeforeEval = $env:APERTURE_REQUIRE_REFLECTION_USER_MEMORY_APPROVAL
try {
    foreach ($policy in @('0', '1')) {
        $env:APERTURE_REQUIRE_REFLECTION_USER_MEMORY_APPROVAL = $policy
        .\.venv\Scripts\python.exe -B tools\memory_acceptance_eval.py --live --split holdout --max-cases 20 --max-calls 44 --timeout-seconds 120 --output ".reliability-tmp\memory-approval-$policy.json"
        if ($LASTEXITCODE -ne 0) { throw "Incomplete evaluation for policy $policy." }
    }
} finally {
    $env:APERTURE_REQUIRE_REFLECTION_USER_MEMORY_APPROVAL = $approvalBeforeEval
}
```

Confirm both reports are complete and use the same case IDs, `fixture_sha256`,
`source_sha256`, model ID, nonempty `model_metadata.digest`, `model_settings`,
and limits. The policy field `require_reflection_user_memory_approval` identifies
the condition. Archive both raw reports. Compare `false_active_writes`,
`false_active_supersessions`, `pending_proposals`, and
`cases_with_pending_proposals` alongside semantic fidelity and review coverage.
The false-supersession count measures replacements that conflict with the gold
decision or fail the recorded lineage check. A structurally correct replacement
can still lose meaning, which requires reviewing the resulting text. Expected
decisions stay fixed across policies, so mandatory queuing defers even a
correct proposed creation or replacement until the user approves it.

## November 2026 policy comparison

Both conditions completed the same 20 published regression cases and 40
inference calls on the runtime from `bafcec6`. The model was `qwen3.5:4b`
with digest `d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`.
The fixture SHA-256 was
`84aa14c8000bab47f7ad8068d3c037fe8b35c9505e9c9b397dc010faada35e28`.

| Result | Allow automatic writes | Require approval |
| --- | ---: | ---: |
| Wrong active writes | 2 | 0 |
| Wrong active supersessions | 0 | 0 |
| Pending proposals | 1 | 11 |
| Faithful texts in agent review | 9/11 | 9/11 |
| Inference errors or provenance failures | 0 | 0 |

The same eleven texts were produced under both conditions. The agent review
found that nondisclosure was turned into a standing memory in one case and
that uncertainty was lost from a conditional statement in another. An
independent human did not score these runs. The evaluator supplies a
prebuilt memory index and isolates reflection from the interactive actor.
These measurements use a published regression set.

Approval is the default because it kept both wrong writes out of active
memory. It also deferred six otherwise correct creations and two otherwise
correct replacements in this sample. The explicit opt-out is documented in
[Pending user memory approval](MEMORY_APPROVAL.md).

## Reading results

Read correct creation and correct rejection separately. A system that never saves a memory fails the positive cases even if it rejects every negative case. Pending proposals are counted separately from active memories. Temporal updates must leave one current version and retain the prior record as history. Structural checks verify subject, source event ownership, and target lineage.

Each case has a decision label and a human review criterion. Compare the newly saved or changed active user memory and any pending proposal with the case dialogue and criterion. Check conditions, exceptions, negation, uncertainty, and attribution. Mark the case `faithful` only if all its new or changed content preserves the relevant meaning. A case without saved or proposed text has only a decision score.

Save judgments in a separate JSON file with a `report_sha256` field for the exact raw report and a `cases` object keyed by case ID. Set `reviewer` to `human` or `agent` to identify who made the judgments. Each judgment has a `verdict` of `faithful` or `unfaithful` and a short `reason`. For the paired evaluation, also include a `checks` object with separate `condition`, `exception`, `negation`, `uncertainty`, and `attribution` entries. Each entry is `preserved`, `lost`, or `not_applicable`.

```powershell
.\.venv\Scripts\python.exe -B tools\memory_acceptance_eval.py --score-report .reliability-tmp\memory-acceptance-holdout.json --semantic-review .reliability-tmp\holdout-review.json --output .reliability-tmp\holdout-scored.json
```

Offline scoring preserves the raw report, model identity, fixture hash, and decision metrics. Compare `semantic_fidelity` alongside `semantic_reviewed` and `semantic_reviewable`. Fidelity is the fraction of reviewed cases judged faithful. A faithful statement can still have an incorrect decision, such as making an uncertain observation active before approval. Freeze case dialogue, decision labels, and review criteria before tuning. Use the same fixture hash when comparing runs.
