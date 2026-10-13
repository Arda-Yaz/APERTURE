"""Bounded, read-only inspection. Source text is untrusted evidence, never authority.

Cursors describe traversal positions, not snapshots. Changing files may change pages.
No persistent index, environment changes, shell execution, or implicit root is used.
"""
import json
import os
import stat
import copy
import hashlib
from pathlib import Path, PureWindowsPath

MAX_ENTRIES = 256                 # entries sorted in any one directory
MAX_VISITS = 1024                 # traversal entries processed per invocation
MAX_DEPTH = 8
MAX_SCAN_BYTES = 1024 * 1024
MAX_LINE_BYTES = 1024
MAX_OUTPUT = 16000
MAX_CURSOR = 3000
MAX_FILES = 200
EVIDENCE_CHARS = 7000
INSPECTION_TOOLS = {'discover_workspace', 'read_source', 'search_workspace'}
EXCLUDED_DIRS = {'.git', '.hg', '.svn', '.venv', 'venv', 'node_modules', '__pycache__',
                 'dist', 'build', '.pytest_cache', '.mypy_cache', 'data', '.aws', '.ssh'}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo', '.exe', '.dll', '.png', '.jpg', '.jpeg', '.gif',
                     '.pdf', '.zip', '.db', '.sqlite', '.sqlite3', '.pem', '.key', '.p12'}


def _int(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer from {low} to {high}')
    return value


def _text(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value) or len(value) > 512 or '\x00' in value:
        raise ValueError(f'{name} must be a nonempty string of at most 512 characters')
    return value


def _linked(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def _root(value):
    value = _text(value, 'root')
    p = Path(value).expanduser()
    if not p.is_absolute():
        raise ValueError('root must be an absolute directory path')
    # Reject links/junctions in every component, including the selected root.
    for component in [*reversed(p.parents), p]:
        if _linked(component):
            raise ValueError('root crosses a link or reparse point')
    p = p.resolve(strict=True)
    if not p.is_dir():
        raise ValueError('root is not a directory')
    return p


def _target(root, value):
    value = _text(value, 'path')
    win = PureWindowsPath(value)
    parts = value.replace('\\', '/').split('/')
    if win.drive or win.root or ':' in value or value.startswith('/') or '..' in parts:
        raise ValueError('path must be relative to root, without traversal, drive, UNC, or ADS')
    p = root
    for part in parts:
        if part in {'', '.'}:
            continue
        # Windows strips trailing spaces/dots and recognizes device names.
        reserved = os.path.isreserved(part) if hasattr(os.path, 'isreserved') else PureWindowsPath(part).is_reserved()
        if part.endswith((' ', '.')) or reserved:
            raise ValueError('ambiguous or reserved path component')
        p = p / part
        if _linked(p):
            raise ValueError('path crosses a link or reparse point')
    if not p.resolve(strict=True).is_relative_to(root):
        raise ValueError('path escapes root')
    return p


def _inspect_target(root, value):
    p = _target(root, value)
    relative = p.relative_to(root)
    for index, part in enumerate(relative.parts):
        if _excluded(part, index < len(relative.parts) - 1 or p.is_dir()):
            raise ValueError('path is excluded from workspace inspection')
    return p


def _excluded(name, directory=False):
    lower = name.casefold()
    return (lower in EXCLUDED_DIRS if directory else
            lower.startswith('.env') or lower in {'credentials', 'secrets', 'secrets.json'}
            or Path(lower).suffix in EXCLUDED_SUFFIXES)


def _result(tool, root, path='.', **fields):
    value = {'tool': tool, 'status': 'ok', 'root': str(root), 'path': path, **fields}
    rendered = json.dumps(value, ensure_ascii=False)
    if len(rendered) > MAX_OUTPUT:
        raise ValueError('result metadata exceeds output bound; narrow path')
    return rendered


def _cursor(value, tool, root, path, query=None):
    if not isinstance(value, str):
        raise ValueError('continuation must be a string')
    if value == '':
        return {'stack': [{'path': path, 'after': ''}]}
    if len(value) > MAX_CURSOR:
        raise ValueError('invalid continuation')
    data = json.loads(value)
    if not isinstance(data, dict) or data.get('tool') != tool or data.get('root') != str(root) or data.get('path') != path or data.get('query') != query:
        raise ValueError('continuation does not belong to this inspection')
    checksum = data.pop('checksum', None)
    expected = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    if checksum != expected:
        raise ValueError('continuation was modified; pass the returned cursor unchanged')
    stack = data.get('stack')
    if not isinstance(stack, list) or len(stack) > MAX_DEPTH + 1:
        raise ValueError('invalid traversal continuation')
    for frame in stack:
        if not isinstance(frame, dict):
            raise ValueError('invalid traversal frame')
        item = _inspect_target(root, frame['path'])
        if not item.is_relative_to(_target(root, path)):
            raise ValueError('continuation escapes selected path')
        _text(frame['after'], 'after', empty=True)
        relative = Path(frame['path'])
        if len(relative.parts) > MAX_DEPTH + len(Path(path).parts):
            raise ValueError('continuation exceeds depth bound')
    return data


def _pack(state, tool, root, path, query=None):
    data = {**state, 'tool': tool, 'root': str(root), 'path': path, 'query': query}
    data.pop('checksum', None)
    data['checksum'] = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    result = json.dumps(data, ensure_ascii=False)
    if len(result) > MAX_CURSOR:
        raise ValueError('continuation exceeds metadata bound; narrow path')
    return result


def _entries(folder):
    # Never materialize more than MAX_ENTRIES+1, even for enormous directories.
    with os.scandir(folder) as iterator:
        entries = []
        for entry in iterator:
            entries.append(entry.name)
            if len(entries) > MAX_ENTRIES:
                return None
    return sorted(entries)


def _walk(root, state, coverage):
    visits = 0
    snapshots = {}
    while state['stack'] and visits < MAX_VISITS:
        frame = state['stack'][-1]
        folder = _target(root, frame['path'])
        try:
            if frame['path'] not in snapshots:
                snapshots[frame['path']] = _entries(folder)
                coverage['enumerated_entries'] += min(MAX_ENTRIES + 1, len(snapshots[frame['path']]) if snapshots[frame['path']] is not None else MAX_ENTRIES + 1)
            names = snapshots[frame['path']]
        except OSError:
            coverage['skipped'].append({'path': frame['path'], 'reason': 'inaccessible directory'})
            state['stack'].pop()
            visits += 1
            continue
        if names is None:
            coverage['skipped'].append({'path': frame['path'], 'reason': 'directory entry cap'})
            state['stack'].pop()
            visits += 1
            continue
        following = [name for name in names if name > frame['after']]
        if not following:
            state['stack'].pop()
            continue
        name = following[0]
        frame['after'] = name
        visits += 1
        coverage['visited'] = visits
        relative = (Path(frame['path']) / name).as_posix()
        try:
            item = _target(root, relative)
            directory = item.is_dir()
        except (OSError, ValueError):
            coverage['skipped'].append({'path': relative[:512], 'reason': 'link, invalid, or inaccessible target'})
            continue
        if _excluded(name, directory):
            coverage['skipped'].append({'path': relative[:512], 'reason': 'excluded'})
            continue
        if directory:
            if len(state['stack']) <= MAX_DEPTH:
                state['stack'].append({'path': relative, 'after': ''})
            else:
                coverage['skipped'].append({'path': relative[:512], 'reason': 'depth cap'})
        yield relative, directory
    coverage['visited'] = visits


def _coverage():
    return {'visited': 0, 'enumerated_entries': 0, 'files_visited': 0, 'scanned_bytes': 0, 'skipped': []}


def _finish_coverage(coverage, state):
    count = len(coverage['skipped'])
    coverage['skipped_count'] = count
    coverage['skipped'] = coverage['skipped'][:8]
    coverage['complete'] = not state.get('stack') and not state.get('file') and count == 0
    coverage['scope'] = 'this page; prior pages and excluded paths are not certified'


def discover_workspace(root: str, path: str = '.', limit: int = 100, continuation: str = '') -> str:
    """Discover a deterministic bounded page below an explicit root. Pass continuation unchanged."""
    try:
        root = _root(root)
        folder = _inspect_target(root, path)
        if not folder.is_dir():
            raise ValueError('path is not a directory')
        _int(limit, 'limit', 1, 100)
        state = _cursor(continuation, 'discover_workspace', root, path)
        coverage, evidence, size = _coverage(), [], 0
        iterator = _walk(root, state, coverage)
        while len(evidence) < limit:
            previous = copy.deepcopy(state)
            try:
                relative, directory = next(iterator)
            except StopIteration:
                break
            record = {'path': relative, 'kind': 'directory' if directory else 'file'}
            cost = len(json.dumps(record, ensure_ascii=False))
            if size + cost > EVIDENCE_CHARS:
                state = previous
                break
            evidence.append(record)
            size += cost
        iterator.close()
        _finish_coverage(coverage, state)
        more = _pack(state, 'discover_workspace', root, path) if state['stack'] else None
        return _result('discover_workspace', root, path, coverage=coverage,
                       continuation=more, truncated=bool(more), evidence=evidence)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return f'TOOL_ERROR: {type(exc).__name__}: {exc}'


def _fragment(stream, remaining):
    raw = stream.readline(min(MAX_LINE_BYTES, remaining))
    if b'\x00' in raw:
        raise ValueError('binary content is not source text')
    # An arbitrary chunk can end within a UTF-8 character; move trailing bytes
    # back into the stream, while genuine malformed UTF-8 is still an error.
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        if exc.reason == 'unexpected end of data' and stream.tell() < os.fstat(stream.fileno()).st_size:
            stream.seek(-(len(raw) - exc.start), 1)
            raw = raw[:exc.start]
            text = raw.decode('utf-8')
        else:
            raise ValueError('source is not valid UTF-8') from exc
    return raw, text


def read_source(root: str, path: str, start_line: int = 1, line_count: int = 50,
                continuation: str = '') -> str:
    """Read numbered UTF-8 source fragments, at most 1 MiB scanned and 50 lines per page."""
    try:
        root = _root(root)
        file = _inspect_target(root, path)
        if not file.is_file() or _excluded(file.name):
            raise ValueError('path is not an inspectable source file')
        _int(start_line, 'start_line', 1, 10000000)
        _int(line_count, 'line_count', 1, 50)
        offset, line = 0, 1
        if continuation:
            data = _cursor(continuation, 'read_source', root, path)
            offset = _int(data.get('offset'), 'offset', 0, file.stat().st_size)
            line = _int(data.get('line'), 'line', 1, 10000000)
            if data.get('start_line') != start_line or data.get('line_count') != line_count:
                raise ValueError('continuation range differs')
        scanned, evidence, size, eof = 0, [], 0, False
        with file.open('rb') as stream:
            stream.seek(offset)
            while scanned < MAX_SCAN_BYTES and line < start_line + line_count:
                if MAX_SCAN_BYTES - scanned < 4:
                    break
                fragment_offset = stream.tell()
                raw, text = _fragment(stream, MAX_SCAN_BYTES - scanned)
                if not raw:
                    eof = True
                    break
                scanned += len(raw)
                ended = raw.endswith(b'\n')
                if line >= start_line:
                    record = {'line': line, 'byte_start': fragment_offset, 'byte_end': stream.tell(),
                              'text': text.rstrip('\r\n'), 'fragment': not ended}
                    cost = len(json.dumps(record, ensure_ascii=False))
                    if size + cost > EVIDENCE_CHARS:
                        stream.seek(fragment_offset)
                        break
                    evidence.append(record)
                    size += cost
                if ended:
                    line += 1
                if size >= EVIDENCE_CHARS:
                    break
            offset = stream.tell()
            eof = eof or offset == file.stat().st_size
        more = None
        if not eof and line < start_line + line_count:
            more = _pack({'stack': [], 'offset': offset, 'line': line,
                          'start_line': start_line, 'line_count': line_count}, 'read_source', root, path)
        return _result('read_source', root, path, range={'requested_start': start_line,
                       'requested_count': line_count, 'returned_start': evidence[0]['line'] if evidence else None,
                       'returned_end': evidence[-1]['line'] if evidence else None},
                       coverage={'scanned_bytes': scanned, 'eof': eof, 'empty': file.stat().st_size == 0,
                                 'complete': more is None}, continuation=more,
                       truncated=bool(more), evidence=evidence)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return f'TOOL_ERROR: {type(exc).__name__}: {exc}'


def search_workspace(root: str, query: str, path: str = '.', limit: int = 30,
                     continuation: str = '') -> str:
    """Case-sensitive literal UTF-8 text search, bounded to 1 MiB and 30 matches per page."""
    try:
        root = _root(root)
        _text(query, 'query')
        if '\n' in query or '\r' in query:
            raise ValueError('query must be a single-line literal')
        _int(limit, 'limit', 1, 30)
        if not _inspect_target(root, path).is_dir():
            raise ValueError('path is not a directory')
        state = _cursor(continuation, 'search_workspace', root, path, query)
        coverage, evidence, size = _coverage(), [], 0
        output_full = False
        iterator = _walk(root, state, coverage)
        while coverage['scanned_bytes'] < MAX_SCAN_BYTES - 3 and len(evidence) < limit and size < EVIDENCE_CHARS and coverage['files_visited'] < MAX_FILES:
            current = state.pop('file', None)
            if current is None:
                for relative, directory in iterator:
                    if not directory:
                        current = {'path': relative, 'offset': 0, 'line': 1, 'tail': ''}
                        break
            if current is None:
                break
            file = _inspect_target(root, current['path'])
            if not file.is_relative_to(_target(root, path)) or not file.is_file():
                raise ValueError('continuation file escapes selected search directory')
            coverage['files_visited'] += 1
            offset = _int(current['offset'], 'offset', 0, file.stat().st_size)
            line = _int(current['line'], 'line', 1, 10000000)
            tail = _text(current.get('tail', ''), 'tail', empty=True)
            try:
                with file.open('rb') as stream:
                    stream.seek(offset)
                    while coverage['scanned_bytes'] < MAX_SCAN_BYTES - 3:
                        fragment_offset = stream.tell()
                        raw, text = _fragment(stream, MAX_SCAN_BYTES - coverage['scanned_bytes'])
                        if not raw:
                            break
                        coverage['scanned_bytes'] += len(raw)
                        ended = raw.endswith(b'\n')
                        combined = tail + text
                        if query in combined:
                            pos = combined.find(query)
                            excerpt = combined[max(0, pos - 80):pos + len(query) + 160]
                            evidence.append({'path': current['path'], 'line': line, 'text': excerpt,
                                             'fragment': not ended})
                            cost = len(json.dumps(evidence[-1], ensure_ascii=False))
                            if cost > EVIDENCE_CHARS:
                                evidence[-1]['text'] = query
                                evidence[-1]['excerpt_truncated'] = True
                                cost = len(json.dumps(evidence[-1], ensure_ascii=False))
                            if size + cost > EVIDENCE_CHARS and size:
                                evidence.pop()
                                stream.seek(fragment_offset)
                                output_full = True
                                break
                            size += cost
                        tail = '' if ended else combined[-(len(query) - 1):] if len(query) > 1 else ''
                        if ended:
                            line += 1
                        if len(evidence) >= limit or size >= EVIDENCE_CHARS:
                            break
                    if stream.tell() < file.stat().st_size:
                        state['file'] = {'path': current['path'], 'offset': stream.tell(), 'line': line, 'tail': tail}
            except (OSError, ValueError) as exc:
                coverage['skipped'].append({'path': current['path'], 'reason': str(exc)[:120]})
            if output_full:
                break
        iterator.close()
        _finish_coverage(coverage, state)
        more = _pack(state, 'search_workspace', root, path, query) if state['stack'] or state.get('file') else None
        return _result('search_workspace', root, path, query=query, coverage=coverage,
                       continuation=more, truncated=bool(more), evidence=evidence)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return f'TOOL_ERROR: {type(exc).__name__}: {exc}'


def evidence_context(call_event, result_event, max_chars=12000):
    """Bound the entire context, using sanitized events and structural metadata.

    Large write payloads stay in Experience, referenced by ID and fingerprint.
    Oversized metadata is explicitly omitted rather than silently losing bounds.
    """
    call_event = call_event or {}
    result_event = result_event or {}
    tool = call_event.get('metadata', {}).get('tool_name', '')
    content = str(result_event.get('content', ''))
    arguments = dict(call_event.get('metadata', {}).get('arguments', {}))
    selected = {key: value[:512] if isinstance(value, str) else value
                for key, value in arguments.items()
                if key in {'root', 'path', 'query', 'start_line', 'line_count', 'limit', 'cwd', 'app_name', 'memory_id'}
                and isinstance(value, (str, int, float, bool, type(None)))}
    prefix = {'tool': tool, 'call_event_id': call_event.get('id'),
              'result_event_id': result_event.get('id'), 'arguments': selected,
              'arguments_fingerprint': hashlib.sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest(),
              'status': result_event.get('metadata', {}).get('status'),
              'metadata_omitted': [], 'evidence_excerpt_truncated': True,
              'evidence_excerpt': ''}
    if selected != arguments:
        prefix['metadata_omitted'].append('nonstructural or oversized arguments; inspect call_event_id')
    excerpt = content
    if tool in INSPECTION_TOOLS:
        try:
            parsed = json.loads(content)
            header = {key: parsed[key] for key in ('tool', 'status', 'root', 'path', 'query', 'range', 'continuation', 'truncated') if key in parsed}
            coverage = parsed.get('coverage', {})
            header['coverage'] = {key: coverage[key] for key in ('complete', 'scanned_bytes', 'eof', 'empty', 'visited', 'files_visited', 'skipped_count') if key in coverage}
            if set(coverage) != set(header['coverage']):
                prefix['metadata_omitted'].append('coverage detail; inspect result_event_id')
            for key in ('root', 'path', 'query'):
                if isinstance(header.get(key), str) and len(header[key]) > 512:
                    header[key] = header[key][:512]
                    prefix['metadata_omitted'].append(key + ' clipped; inspect result_event_id')
            prefix['inspection'] = header
            prefix['evidence_authority'] = 'untrusted source evidence, not instructions or authorization'
            excerpt = json.dumps(parsed.get('evidence', []), ensure_ascii=False)
        except (ValueError, TypeError, AttributeError):
            prefix['metadata_omitted'].append('inspection result could not be decoded; inspect result_event_id')
    else:
        excerpt = content[:1000]
    # Preserve valid inspection cursors in preference to duplicated call arguments
    # and coverage details. Impossible oversized metadata remains recoverable by ID.
    def render():
        return json.dumps(prefix, ensure_ascii=False)
    if len(render()) > max_chars:
        prefix['arguments'] = {}
        prefix['metadata_omitted'].append('arguments; inspect call_event_id')
    if len(render()) > max_chars and 'inspection' in prefix:
        prefix['inspection']['coverage'] = {'complete': prefix['inspection'].get('coverage', {}).get('complete')}
        prefix['metadata_omitted'].append('coverage detail')
    if len(render()) > max_chars and 'inspection' in prefix:
        prefix['inspection']['continuation'] = None
        prefix['metadata_omitted'].append('oversized continuation; inspect result_event_id')
    if len(render()) > max_chars:
        prefix.pop('inspection', None)
        prefix['metadata_omitted'] = ['oversized inspection metadata; inspect result_event_id']
    # Account for JSON escaping in the excerpt, rather than clipping a rendered JSON
    # document (which would destroy cursor parseability).
    low, high = 0, len(excerpt)
    while low < high:
        middle = (low + high + 1) // 2
        prefix['evidence_excerpt'] = excerpt[:middle]
        if len(render()) <= max_chars:
            low = middle
        else:
            high = middle - 1
    prefix['evidence_excerpt'] = excerpt[:low]
    prefix['evidence_excerpt_truncated'] = low < len(excerpt) or (tool not in INSPECTION_TOOLS and len(content) > 1000)
    # false is one character longer than true.
    if len(render()) > max_chars:
        prefix['evidence_excerpt'] = prefix['evidence_excerpt'][:-1]
        prefix['evidence_excerpt_truncated'] = True
    return render()
