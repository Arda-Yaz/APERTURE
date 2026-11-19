# Installation and verification

APERTURE is a Windows-first Python application. The development baseline uses Python 3.10 or newer and a local Ollama service. The Python packages are listed in `requirements.txt`; Ollama models are installed separately.

## Set up Python

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The requirements pin the direct Python dependencies used by the application. They are not a complete transitive lock file. No installation command modifies APERTURE's local SQLite data.

## Configure Ollama

Install and start Ollama separately. With no model overrides, all inference
roles use `qwen3.5:4b`.

```powershell
ollama pull qwen3.5:4b
```

Set `APERTURE_MODEL` to change the default for every role. A role-specific
override such as `APERTURE_MODEL_ACTOR` or `APERTURE_MODEL_REFLECTION` takes
precedence over that default. The role suffixes are `ACTOR`, `RETRIEVAL`,
`REFLECTION`, `SELF_STATE`, `RELATIONSHIP`, and `TASK_CONTROLLER`. Pull the
model selected for each role you use.

The model download is needed for interactive use. The automated tests do not need a running Ollama service and block unmocked model calls.

## Run and verify

```powershell
.\.venv\Scripts\python.exe -B app\main.py
.\.venv\Scripts\python.exe -B tools\run_isolated_tests.py --report .reliability-tmp\local-tests.json
```

Run the two commands separately. The first starts the interactive application and uses local persistent state. The second runs synthetic tests with temporary SQLite databases and an inference guard. The JSON report records counts, failures, and isolation violations. On managed Windows hosts, hardlink or junction creation may be denied before those filesystem tests reach their assertions; inspect the report rather than treating the entire run as passed.

The default tool profile is `memory`. It advertises memory and scoped workspace tools while denying general terminal, application-launch, and legacy file tools. Set `APERTURE_TOOL_PROFILE=advanced` explicitly only when host-level access is intended. See [pending memory approval](MEMORY_APPROVAL.md) for the user decision commands and their provenance rules.

The GitHub Actions workflow runs the guarded test suite on Windows with Python 3.12. It does not start the interactive application or Ollama. A green workflow validates the synthetic regression suite; it does not establish the semantic quality of real-model conversations.
