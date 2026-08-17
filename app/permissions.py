from pathlib import Path

MODE = "AUTONOMOUS"

WORKSPACE = Path.home() / "Documents" / "Aperture"


def check_permission(tool_name: str, target: str = "") -> bool:

    # Tamamen güvenli/read-only işlemler
    READ_TOOLS = {
        "list_directory",
        "read_file",
    }

    # Dış dünyada etkisi olan işlemler
    SENSITIVE_TOOLS = {
        "send_email",
        "publish_linkedin",
        "git_push",
        "delete_file",
    }

    AUTONOMOUS_TOOLS = {
    "list_directory",
    "read_file",
    "open_app",
    "run_terminal",
}

    if MODE == "AUTONOMOUS":

        if tool_name in AUTONOMOUS_TOOLS:
            return True

        if tool_name in SENSITIVE_TOOLS:
            return ask(tool_name, target)

        return ask(tool_name, target)

    if MODE == "BALANCED":
        if tool_name in READ_TOOLS:
            return True

        return ask(tool_name, target)

    # SAFE
    return ask(tool_name, target)


def ask(tool_name: str, target: str) -> bool:
    print(f"\n[PERMISSION] {tool_name} -> {target}")

    choice = input("Allow? (y/n): ").strip().lower()

    return choice == "y"