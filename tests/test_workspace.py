import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import isolation_support as safety
safety.install_guards()
import workspace as ws
import permissions
import goals
import llm


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        directory = safety.TestTemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def file(self, path, text='hello\n'):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding='utf-8', newline='')
        return file

    def result(self, value):
        self.assertFalse(value.startswith('TOOL_ERROR'), value)
        self.assertLessEqual(len(value), ws.MAX_OUTPUT)
        return json.loads(value)

    def test_discovery_sorted_pages_without_omissions(self):
        for name in ['z.py', 'a.py', 'sub/b.py', 'sub/a.py']:
            self.file(name)
        found, cursor = [], ''
        for _ in range(10):
            page = self.result(ws.discover_workspace(str(self.root), limit=1, continuation=cursor))
            found.extend(item['path'] for item in page['evidence'])
            cursor = page['continuation']
            if not cursor:
                break
        self.assertEqual(found, ['a.py', 'sub', 'sub/a.py', 'sub/b.py', 'z.py'])
        self.assertEqual(len(found), len(set(found)))

    def test_exclusions_depth_and_oversized_directory(self):
        self.file('.env', 'secret')
        self.file('data/a.py', 'secret')
        self.file('node_modules/a.py')
        self.file('normal.py')
        big = self.root / 'big'
        big.mkdir()
        for index in range(ws.MAX_ENTRIES + 1):
            (big / str(index)).touch()
        page = self.result(ws.discover_workspace(str(self.root)))
        self.assertEqual([item['path'] for item in page['evidence']], ['big', 'normal.py'])
        self.assertFalse(page['coverage']['complete'])
        self.assertTrue(any(item['reason'] == 'directory entry cap' for item in page['coverage']['skipped']))
        for tool, kwargs in [(ws.read_source, {'path': 'data/a.py'}),
                             (ws.discover_workspace, {'path': 'node_modules'}),
                             (ws.search_workspace, {'path': 'data', 'query': 'secret'})]:
            self.assertTrue(tool(str(self.root), **kwargs).startswith('TOOL_ERROR'))
        deep = '/'.join(['folder'] * (ws.MAX_DEPTH + 2)) + '/source.py'
        self.file(deep)
        page = self.result(ws.discover_workspace(str(self.root)))
        self.assertTrue(any(item['reason'] == 'depth cap' for item in page['coverage']['skipped']))

    def test_later_ranges_unicode_crlf_empty_and_eof(self):
        self.file('large.py', ''.join(f'line {i}: café 東京\r\n' for i in range(1, 3000)))
        page = self.result(ws.read_source(str(self.root), 'large.py', start_line=2500, line_count=2))
        self.assertEqual([x['line'] for x in page['evidence']], [2500, 2501])
        self.assertEqual(page['evidence'][0]['text'], 'line 2500: café 東京')
        self.assertFalse(page['coverage']['eof'])
        self.assertIsNone(page['continuation'])
        self.file('empty.py', '')
        page = self.result(ws.read_source(str(self.root), 'empty.py'))
        self.assertTrue(page['coverage']['empty'])
        self.assertTrue(page['coverage']['eof'])
        page = self.result(ws.read_source(str(self.root), 'large.py', start_line=4000))
        self.assertEqual(page['evidence'], [])
        self.assertTrue(page['coverage']['eof'])

    def test_long_line_fragments_keep_byte_identity_and_continuation(self):
        self.file('long.py', 'é' * 15000 + '\nsecond\n')
        cursor, fragments = '', []
        for _ in range(10):
            page = self.result(ws.read_source(str(self.root), 'long.py', line_count=2, continuation=cursor))
            fragments.extend(page['evidence'])
            cursor = page['continuation']
            if not cursor:
                break
        self.assertIsNone(cursor)
        self.assertEqual(''.join(item['text'] for item in fragments if item['line'] == 1), 'é' * 15000)
        self.assertEqual(fragments[-1]['line'], 2)
        self.assertEqual(fragments[-1]['text'], 'second')
        self.assertEqual([item['byte_start'] for item in fragments][1:], [item['byte_end'] for item in fragments][:-1])

    def test_scan_cap_before_requested_range_is_not_eof(self):
        self.file('scan.py', 'a' * (ws.MAX_SCAN_BYTES + 100) + '\nrequested\n')
        page = self.result(ws.read_source(str(self.root), 'scan.py', start_line=2, line_count=1))
        self.assertFalse(page['coverage']['eof'])
        self.assertTrue(page['truncated'])
        self.assertEqual(page['evidence'], [])
        page2 = self.result(ws.read_source(str(self.root), 'scan.py', start_line=2, line_count=1, continuation=page['continuation']))
        self.assertEqual(page2['evidence'][0]['text'], 'requested')

    def test_literal_search_locations_and_within_file_continuation(self):
        self.file('a.py', 'x\nneedle[.*]\nneedle[.*]\n')
        self.file('b.py', 'needle[.*]\n')
        self.file('binary.bin').write_bytes(b'\x00needle[.*]')
        matches, cursor = [], ''
        for _ in range(8):
            page = self.result(ws.search_workspace(str(self.root), 'needle[.*]', limit=1, continuation=cursor))
            matches.extend((item['path'], item['line']) for item in page['evidence'])
            cursor = page['continuation']
            if not cursor:
                self.assertFalse(page['coverage']['complete'])
                break
            self.assertTrue(page['truncated'])
        self.assertEqual(matches, [('a.py', 2), ('a.py', 3), ('b.py', 1)])

    def test_search_scan_and_file_caps_and_cross_fragment_match(self):
        self.file('a.py', 'x' * (ws.MAX_LINE_BYTES - 2) + 'needle\n')
        page = self.result(ws.search_workspace(str(self.root), 'needle'))
        self.assertEqual(page['evidence'][0]['line'], 1)
        self.file('a.py', 'x' * (ws.MAX_SCAN_BYTES + 100) + '\nneedle\n')
        page = self.result(ws.search_workspace(str(self.root), 'needle'))
        self.assertTrue(page['truncated'])
        self.assertLessEqual(page['coverage']['scanned_bytes'], ws.MAX_SCAN_BYTES)
        page2 = self.result(ws.search_workspace(str(self.root), 'needle', continuation=page['continuation']))
        self.assertEqual(page2['evidence'][0]['line'], 2)
        for index in range(ws.MAX_FILES + 1):
            self.file(f'empty/{index:03}.py', '')
        page = self.result(ws.search_workspace(str(self.root), 'needle', path='empty'))
        self.assertEqual(page['coverage']['files_visited'], ws.MAX_FILES)
        self.assertTrue(page['truncated'])

    def test_invalid_arguments_binary_decode_missing_and_inaccessible(self):
        self.file('a.py')
        for kwargs in [{'limit': True}, {'limit': 101}, {'continuation': 0}, {'continuation': '[]'}]:
            self.assertTrue(ws.discover_workspace(str(self.root), **kwargs).startswith('TOOL_ERROR'))
        for path in ['../outside', '/absolute', 'C:\\outside', 'C:outside', '\\\\server\\share', 'a.py:secret', 'a.py.', 'CON', 'missing.py']:
            with self.subTest(path=path):
                self.assertTrue(ws.read_source(str(self.root), path).startswith('TOOL_ERROR'))
        self.file('bad.py').write_bytes(b'\xff\xfe')
        self.file('binary.py').write_bytes(b'abc\x00def')
        for path in ['bad.py', 'binary.py']:
            self.assertTrue(ws.read_source(str(self.root), path).startswith('TOOL_ERROR'))
        with patch.object(Path, 'open', side_effect=PermissionError('denied')):
            self.assertTrue(ws.read_source(str(self.root), 'a.py').startswith('TOOL_ERROR'))
        self.assertTrue(ws.search_workspace(str(self.root), 'a\nb').startswith('TOOL_ERROR'))
        self.assertTrue(ws.discover_workspace('relative').startswith('TOOL_ERROR'))

    def test_incomplete_utf8_at_physical_eof_is_a_decoding_failure(self):
        for index, raw in enumerate([b'valid\xc3', b'\xe2\x82']):
            name = f'broken{index}.py'
            self.file(name).write_bytes(raw)
            self.assertTrue(ws.read_source(str(self.root), name).startswith('TOOL_ERROR'))
        page = self.result(ws.search_workspace(str(self.root), 'valid'))
        self.assertFalse(page['coverage']['complete'])
        self.assertEqual(page['coverage']['skipped_count'], 2)
        self.assertTrue(all('UTF-8' in item['reason'] for item in page['coverage']['skipped']))

    @unittest.skipUnless(os.name == 'nt', 'Windows junction boundary')
    def test_windows_junctions_are_never_followed(self):
        self.file('target/a.py')
        link = self.root / 'junction'
        completed = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(self.root / 'target')], capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.addCleanup(lambda: os.rmdir(link) if link.exists() else None)
        self.assertTrue(ws.read_source(str(self.root), 'junction/a.py').startswith('TOOL_ERROR'))
        self.assertTrue(ws.discover_workspace(str(link)).startswith('TOOL_ERROR'))
        page = self.result(ws.discover_workspace(str(self.root)))
        self.assertNotIn('junction', [item['path'] for item in page['evidence']])
        self.assertTrue(any(item['path'] == 'junction' for item in page['coverage']['skipped']))
        external = safety.TestTemporaryDirectory()
        self.addCleanup(external.cleanup)
        (Path(external.name) / 'outside.py').write_text('outside-only', encoding='utf-8')
        outside_link = self.root / 'outside-link'
        completed = subprocess.run(['cmd', '/c', 'mklink', '/J', str(outside_link), external.name], capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.addCleanup(lambda: os.rmdir(outside_link) if outside_link.exists() else None)
        self.assertTrue(ws.read_source(str(self.root), 'outside-link/outside.py').startswith('TOOL_ERROR'))
        searched = self.result(ws.search_workspace(str(self.root), 'outside-only'))
        self.assertEqual(searched['evidence'], [])
        self.assertTrue(any(item['path'] == 'outside-link' for item in searched['coverage']['skipped']))

    def test_cursor_binding_tampering_and_empty_string_validation(self):
        self.file('sub/a.py', 'needle\nneedle\n')
        self.file('other/b.py', 'needle\n')
        page = self.result(ws.search_workspace(str(self.root), 'needle', path='sub', limit=1))
        cursor = page['continuation']
        self.assertTrue(ws.search_workspace(str(self.root), 'different', path='sub', continuation=cursor).startswith('TOOL_ERROR'))
        self.assertTrue(ws.search_workspace(str(self.root), 'needle', path='other', continuation=cursor).startswith('TOOL_ERROR'))
        modified = json.loads(cursor)
        modified['file']['line'] = 100
        self.assertTrue(ws.search_workspace(str(self.root), 'needle', path='sub', continuation=json.dumps(modified)).startswith('TOOL_ERROR'))

    def test_permissions_exposure_and_read_only_replay(self):
        for mode in ['AUTONOMOUS', 'BALANCED', 'SAFE']:
            with patch.object(permissions, 'MODE', mode), patch.object(permissions, 'ask', return_value=False) as ask:
                for tool in ws.INSPECTION_TOOLS:
                    self.assertEqual(permissions.check_permission(tool, str(self.root)), mode != 'SAFE')
                    self.assertIn(tool, llm.TOOL_MAP)
                    self.assertIn(tool, {function.__name__ for function in llm.get_tools_for_goal('inspect')})
                    self.assertNotIn(tool, goals.MUTATING_TOOLS)
                self.assertEqual(ask.call_count, 3 if mode == 'SAFE' else 0)


if __name__ == '__main__':
    unittest.main()
