"""Application tool authorization; scoped tools enforce their own paths too."""
import os
from pathlib import Path
from workspace import _root


def resolve_mode(environ=None):
    environ = os.environ if environ is None else environ
    profile = environ.get('APERTURE_TOOL_PROFILE', 'memory').strip().lower()
    if profile not in {'memory', 'advanced'}:
        raise ValueError('APERTURE_TOOL_PROFILE must be memory or advanced')
    return 'AUTONOMOUS' if profile == 'advanced' else 'MEMORY_ASSISTANT'


MODE = resolve_mode()
WORKSPACE = Path(os.environ.get('APERTURE_WORKSPACE', str(Path(__file__).resolve().parent.parent)))
MEMORY_TOOLS = {'save_memory', 'search_memory', 'save_self_memory', 'forget_memory', 'recall_recent_activity'}
SCOPED_READS = {'discover_workspace', 'read_source', 'search_workspace', 'snapshot_text'}
SCOPED_WRITES = {'create_text', 'update_text'}
HOST_TOOLS = {'list_directory', 'read_file', 'write_file', 'open_app', 'run_terminal'}
KNOWN_TOOLS = MEMORY_TOOLS | SCOPED_READS | SCOPED_WRITES | HOST_TOOLS


def tool_enabled(tool_name):
    return tool_name in KNOWN_TOOLS and (MODE != 'MEMORY_ASSISTANT' or tool_name not in HOST_TOOLS)


def _workspace_allowed(arguments):
    if not isinstance(arguments, dict) or not isinstance(arguments.get('root'), str):
        return False
    try:
        root = _root(arguments['root'])
        return root.is_relative_to(WORKSPACE.resolve())
    except (OSError, ValueError):
        return False


def check_permission(tool_name: str, target: str = '', *, arguments=None) -> bool:
    if tool_name not in KNOWN_TOOLS:
        return False
    if MODE == 'MEMORY_ASSISTANT':
        if tool_name == 'forget_memory':
            return ask(tool_name, target)
        if tool_name in MEMORY_TOOLS:
            return True
        if tool_name in SCOPED_READS | SCOPED_WRITES and _workspace_allowed(arguments):
            return True if tool_name in SCOPED_READS else ask(tool_name, target)
        return False
    if MODE not in {'AUTONOMOUS', 'BALANCED', 'SAFE'}:
        return False
    if tool_name in SCOPED_WRITES or tool_name == 'write_file':
        return ask(tool_name, target)
    if MODE == 'AUTONOMOUS':
        return True
    if MODE == 'BALANCED' and tool_name in SCOPED_READS | {'list_directory', 'read_file', 'search_memory', 'recall_recent_activity'}:
        return True
    return ask(tool_name, target)


def ask(tool_name: str, target: str) -> bool:
    print(f'\n[PERMISSION] {tool_name} -> {target}')
    return input('Allow? (y/n): ').strip().lower() == 'y'
