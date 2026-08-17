from pathlib import Path
import os
import subprocess
import shutil

def get_downloads_folder() -> Path:
    userprofile = Path(os.environ["USERPROFILE"])

    candidates = [
        userprofile / "Downloads",
        userprofile / "İndirilenler",
    ]

    for path in candidates:
        if path.exists():
            return path

    return userprofile / "Downloads"


KNOWN_FOLDERS = {
    "downloads": get_downloads_folder(),
    "indirilenler": get_downloads_folder(),
    "desktop": Path(os.environ["USERPROFILE"]) / "Desktop",
    "masaustu": Path(os.environ["USERPROFILE"]) / "Desktop",
    "documents": Path(os.environ["USERPROFILE"]) / "Documents",
    "belgeler": Path(os.environ["USERPROFILE"]) / "Documents",
}


def resolve_path(path: str) -> Path:
    cleaned = path.strip().strip("/\\")

    parts = Path(cleaned).parts

    if parts:
        first = normalize_key(parts[0])

        if first in KNOWN_FOLDERS:
            base = KNOWN_FOLDERS[first]

            if len(parts) > 1:
                return (base.joinpath(*parts[1:])).resolve()

            return base.resolve()

    return Path(path).expanduser().resolve()


def list_directory(path: str) -> str:
    folder = resolve_path(path)

    if not folder.exists():
        return f"Path does not exist: {folder}"

    if not folder.is_dir():
        return f"Path is not a directory: {folder}"

    items = list(folder.iterdir())

    if not items:
        return "Directory is empty."

    return "\n".join(
        f"[{'DIR' if item.is_dir() else 'FILE'}] {item.name}"
        for item in items
    )


import unicodedata

def normalize_key(text: str) -> str:
    text = text.strip().strip("/\\").casefold()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))

def read_file(path: str, max_chars: int = 50000) -> str:


    file = resolve_path(path)

    if not file.exists():
        return f"READ_ERROR: File does not exist: {file}"

    if not file.is_file():
        return f"READ_ERROR: Path is not a file: {file}"

    try:
        content = file.read_text(encoding="utf-8")

    except UnicodeDecodeError:
        try:
            content = file.read_text(encoding="cp1254")
        except Exception as e:
            return f"READ_ERROR: Could not decode file: {type(e).__name__}: {e}"

    except PermissionError:
        return f"READ_ERROR: Permission denied: {file}"

    except Exception as e:
        return f"READ_ERROR: {type(e).__name__}: {e}"

    if not content.strip():
        return f"File is empty: {file.name}"

    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[FILE TRUNCATED]"

    return content

def write_file(path: str, content: str) -> str:
    file = resolve_path(path)

    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
        return f"File written successfully: {file}"
    except Exception as e:
        return f"WRITE_ERROR: {type(e).__name__}: {e}"

def open_app(app_name: str) -> str:
    apps = {
        "vscode": "code",
        "vs code": "code",
        "notepad": "notepad",
        "hesap makinesi": "calc",
        "calculator": "calc",
        "explorer": "explorer",
        "dosya gezgini": "explorer",
    }

    key = app_name.strip().lower()
    command = apps.get(key, app_name)

    try:
        executable = shutil.which(command)

        if executable:
            subprocess.Popen([executable])
        else:
            subprocess.Popen(
                ["cmd", "/c", "start", "", command],
                shell=False
            )

        return f"Application opened: {app_name}"

    except Exception as e:
        return f"OPEN_APP_ERROR: {type(e).__name__}: {e}"

def run_terminal(
    command: str,
    cwd: str = "Documents/Aperture",
    timeout: int = 30
) -> str:
    """
    Run a PowerShell command inside the APERTURE workspace.
    Use this for coding, git, Python, file inspection and development tasks.
    """

    workspace = Path.home() / "Documents" / "Aperture"

    # V0.1: terminali APERTURE workspace'e sabitliyoruz
    working_directory = workspace.resolve()

    blocked_commands = [
        "remove-item",
        "del ",
        "erase ",
        "rmdir",
        "rd ",
        "format ",
        "diskpart",
        "shutdown",
        "reg delete",
    ]

    command_lower = command.lower()

    if any(blocked in command_lower for blocked in blocked_commands):
        return "TERMINAL_ERROR: Destructive command blocked."

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                command,
            ],
            cwd=str(working_directory),
            capture_output=True,
            text=True,
            timeout=min(timeout, 60),
            encoding="utf-8",
            errors="replace",
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        output = f"Exit code: {result.returncode}"

        if stdout:
            output += f"\nSTDOUT:\n{stdout}"

        if stderr:
            output += f"\nSTDERR:\n{stderr}"

        return output

    except subprocess.TimeoutExpired:
        return "TERMINAL_ERROR: Command timed out."

    except Exception as e:
        return f"TERMINAL_ERROR: {type(e).__name__}: {e}"





