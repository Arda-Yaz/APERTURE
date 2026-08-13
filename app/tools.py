from pathlib import Path
import os

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
    "masaüstü": Path(os.environ["USERPROFILE"]) / "Desktop",
    "documents": Path(os.environ["USERPROFILE"]) / "Documents",
    "belgeler": Path(os.environ["USERPROFILE"]) / "Documents",
}


def resolve_path(path: str) -> Path:
    key = normalize_key(path)

    if key in KNOWN_FOLDERS:
        return KNOWN_FOLDERS[key].resolve()

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