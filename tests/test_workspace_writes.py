import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import isolation_support as safety
safety.install_guards()
import workspace as ws
import llm
import permissions
import goals


class TextWorkspaceTests(unittest.TestCase):
    def setUp(self):
        directory = safety.TestTemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.file = self.root / 'config.txt'
        self.file.write_bytes(b'old\r\nkeep\n')

    def call(self, function, **kwargs):
        result = json.loads(function(str(self.root), **kwargs))
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), ws.MAX_OUTPUT)
        return result

    def fingerprint(self):
        return self.call(ws.snapshot_text, path='config.txt')['fingerprint']

    def test_complete_snapshot_and_exact_create_update(self):
        content = '\ufeffcafé\r\nkeep\n'
        raw = content.encode('utf-8')
        self.file.write_bytes(raw)
        snap = self.call(ws.snapshot_text, path='config.txt')
        self.assertEqual(snap['content'], content)
        self.assertEqual(snap['fingerprint'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(snap['byte_size'], len(raw))
        update = self.call(ws.update_text, path='config.txt', content=content + 'extra\r', expected_fingerprint=snap['fingerprint'])
        self.assertEqual(update['status'], 'verified')
        self.assertEqual(self.file.read_bytes(), raw + b'extra\r')
        created = self.call(ws.create_text, path='report.md', content='')
        self.assertEqual(created['status'], 'verified')
        self.assertEqual((self.root / 'report.md').read_bytes(), b'')
        self.assertEqual(list(self.root.glob('.aperture-text-*')), [])

    def test_conflicts_never_change_target(self):
        old = self.file.read_bytes()
        self.assertEqual(self.call(ws.create_text, path='config.txt', content='new')['status'], 'conflict')
        self.assertEqual(self.call(ws.update_text, path='config.txt', content='new', expected_fingerprint='0' * 64)['status'], 'conflict')
        self.assertEqual(self.file.read_bytes(), old)
        self.assertEqual(list(self.root.glob('.aperture-text-*')), [])

    def test_invalid_size_encoding_binary_and_paths(self):
        for path in ['missing/a.txt', '../a.txt', 'C:a.txt', '/a.txt', '\\\\server\\share', 'config.txt:ads', 'CON', 'a.', 'data/a.txt', '.env', '.git/a.txt']:
            with self.subTest(path=path):
                self.assertEqual(self.call(ws.create_text, path=path, content='text')['status'], 'rejected')
        for content in [None, '\x00', '\x01', '\x7f', '\ud800', 'é' * 2049, 'x' * 4097]:
            with self.subTest(content=repr(content)[:30]):
                self.assertEqual(self.call(ws.create_text, path='new.txt', content=content)['status'], 'rejected')
        self.assertFalse((self.root / 'new.txt').exists())
        for raw in [b'\xff', b'\x00', b'\x02', b'x' * 4097]:
            self.file.write_bytes(raw)
            self.assertEqual(self.call(ws.snapshot_text, path='config.txt')['status'], 'rejected')
        self.file.write_bytes(b'x' * 4096)
        self.assertEqual(self.call(ws.snapshot_text, path='config.txt')['byte_size'], 4096)
        self.assertEqual(self.call(ws.update_text, path='config.txt', content='new', expected_fingerprint='bad')['status'], 'rejected')

    def test_protected_absolute_root_and_link(self):
        protected = self.root / 'data'
        protected.mkdir()
        result = json.loads(ws.create_text(str(protected), 'report.txt', 'secret'))
        self.assertEqual(result['status'], 'rejected')
        self.assertFalse((protected / 'report.txt').exists())
        with patch.object(ws, '_linked', side_effect=lambda p: p == self.root):
            self.assertEqual(self.call(ws.create_text, path='report.txt', content='x')['status'], 'rejected')

    def test_prepare_failure_leaves_original_and_unrelated_temp(self):
        unrelated = self.root / '.aperture-text-user'
        unrelated.write_text('user')
        old = self.file.read_bytes()
        with patch.object(ws.os, 'chmod', side_effect=OSError('mode failed')):
            result = self.call(ws.update_text, path='config.txt', content='new', expected_fingerprint=self.fingerprint())
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['effects'], 'none')
        self.assertEqual(self.file.read_bytes(), old)
        self.assertEqual(list(self.root.glob('.aperture-text-*')), [unrelated])

    def test_fdopen_construction_failure_closes_descriptor_and_removes_owned_temp(self):
        real_close = ws.os.close
        with patch.object(ws.os, 'fdopen', side_effect=OSError('fdopen construction failed')), patch.object(ws.os, 'close', wraps=real_close) as close:
            result = self.call(ws.create_text, path='report.txt', content='new')
        self.assertEqual((result['status'], result['effects']), ('rejected', 'none'))
        close.assert_called_once()
        self.assertEqual(list(self.root.glob('.aperture-text-*')), [])
        self.assertFalse((self.root / 'report.txt').exists())

    def test_publish_failure_and_postpublication_mismatch_are_uncertain(self):
        expected = self.fingerprint()
        with patch.object(ws, '_publish_text', side_effect=OSError('publication uncertain')):
            result = self.call(ws.update_text, path='config.txt', content='new', expected_fingerprint=expected)
        self.assertEqual((result['status'], result['effects']), ('error', 'possible'))
        real = ws._publish_text
        def corrupt(temp, target, update):
            real(temp, target, update)
            target.write_bytes(b'other')
        with patch.object(ws, '_publish_text', side_effect=corrupt):
            result = self.call(ws.update_text, path='config.txt', content='new', expected_fingerprint=expected)
        self.assertEqual((result['status'], result['effects']), ('error', 'published'))
        self.assertEqual(list(self.root.glob('.aperture-text-*')), [])

    def test_second_check_and_create_publication_race(self):
        expected = self.fingerprint()
        real = ws._scoped_text_target
        calls = []
        def changed(root, path):
            calls.append(path)
            if len(calls) == 2:
                self.file.write_bytes(b'external')
            return real(root, path)
        with patch.object(ws, '_scoped_text_target', side_effect=changed):
            result = self.call(ws.update_text, path='config.txt', content='new', expected_fingerprint=expected)
        self.assertEqual(result['status'], 'conflict')
        self.assertEqual(self.file.read_bytes(), b'external')
        real_publish = ws._publish_text
        def appears(temp, target, update):
            target.write_bytes(b'external')
            real_publish(temp, target, update)
        with patch.object(ws, '_publish_text', side_effect=appears):
            result = self.call(ws.create_text, path='report.txt', content='new')
        self.assertEqual((result['status'], result['effects']), ('conflict', 'none'))
        self.assertEqual((self.root / 'report.txt').read_bytes(), b'external')

    def test_cleanup_failure_cannot_report_success(self):
        with patch.object(ws.os, 'chmod', side_effect=OSError('preparation failed')), patch.object(Path, 'unlink', side_effect=OSError('cleanup failed')):
            result = self.call(ws.update_text, path='config.txt', content='new', expected_fingerprint=self.fingerprint())
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['effects'], 'possible')

    def test_consumed_temp_name_is_not_cleaned_again(self):
        real = ws._publish_text
        recreated = []
        def recreate(temp, target, update):
            real(temp, target, update)
            if update or os.name == 'nt':
                temp.write_bytes(b'unrelated new file')
                recreated.append(temp)
        with patch.object(ws, '_publish_text', side_effect=recreate):
            result = self.call(ws.create_text, path='report.txt', content='new')
        self.assertEqual(result['status'], 'verified')
        for file in recreated:
            self.assertEqual(file.read_bytes(), b'unrelated new file')

    def test_prepare_failure_never_deletes_replaced_temporary_name(self):
        replaced = []
        real_fdopen = ws.os.fdopen
        class BrokenWriter:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def write(self, raw):
                self.stream.close()
                temp = next(self_root.glob('.aperture-text-*'))
                # Move preserves the old identity while replacement gets its own.
                temp.rename(self_root / 'old-owned-temp')
                temp.write_bytes(b'unrelated')
                replaced.append(temp)
                raise OSError('writer failed after temporary name changed')
            def __exit__(self, *args):
                self.stream.close()
        self_root = self.root
        with patch.object(ws.os, 'fdopen', side_effect=lambda fd, mode: BrokenWriter(real_fdopen(fd, mode))):
            result = self.call(ws.create_text, path='report.txt', content='new')
        self.assertEqual((result['status'], result['effects']), ('error', 'possible'))
        self.assertIn('cleanup failed', result['reason'])
        self.assertEqual(replaced[0].read_bytes(), b'unrelated')
        self.assertFalse((self.root / 'report.txt').exists())

    def test_changed_and_partial_snapshot_rejected(self):
        real = ws._identity
        counter = []
        def changed(info):
            counter.append(info)
            value = real(info)
            return (*value[:-1], value[-1] + len(counter))
        with patch.object(ws, '_identity', side_effect=changed):
            self.assertEqual(self.call(ws.snapshot_text, path='config.txt')['status'], 'rejected')
        real_open = Path.open
        from unittest.mock import MagicMock
        def partial(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            wrapper = MagicMock(wraps=stream)
            wrapper.__enter__.return_value = wrapper
            wrapper.__exit__.side_effect = lambda *args: stream.close()
            wrapper.read.return_value = b'old'
            return wrapper
        with patch.object(Path, 'open', partial):
            self.assertEqual(self.call(ws.snapshot_text, path='config.txt')['status'], 'rejected')

    def test_mode_readonly_hardlinks_and_worst_escaping(self):
        self.file.write_bytes(b'\t' * 4096)
        self.assertEqual(self.call(ws.snapshot_text, path='config.txt')['byte_size'], 4096)
        linked = self.root / 'linked.txt'
        os.link(self.file, linked)
        self.assertEqual(self.call(ws.snapshot_text, path='config.txt')['status'], 'rejected')
        self.assertEqual(self.call(ws.update_text, path='config.txt', content='new', expected_fingerprint=hashlib.sha256(self.file.read_bytes()).hexdigest())['status'], 'rejected')
        linked.unlink()
        if os.name == 'nt':
            os.chmod(self.file, 0o444)
            self.addCleanup(lambda: os.chmod(self.file, 0o666))
            self.assertEqual(self.call(ws.snapshot_text, path='config.txt')['status'], 'rejected')

    @unittest.skipUnless(os.name == 'nt', 'Windows junction boundary')
    def test_write_junction_boundary(self):
        import subprocess
        folder = self.root / 'target'
        folder.mkdir()
        link = self.root / 'junction'
        completed = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(folder)], capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.addCleanup(lambda: os.rmdir(link) if link.exists() else None)
        self.assertEqual(self.call(ws.create_text, path='junction/report.txt', content='new')['status'], 'rejected')
        self.assertFalse((folder / 'report.txt').exists())

    def test_permissions_and_fail_closed_status(self):
        for mode in ['SAFE', 'BALANCED', 'AUTONOMOUS']:
            with patch.object(permissions, 'MODE', mode), patch.object(permissions, 'ask', return_value=False) as ask:
                for tool in ws.WRITE_TOOLS:
                    self.assertFalse(permissions.check_permission(tool, str(self.root) + ' :: report.txt'))
                    self.assertIn(tool, llm.TOOL_MAP)
                    self.assertIn(tool, goals.MUTATING_TOOLS)
                self.assertEqual(ask.call_count, 2)
                self.assertEqual(permissions.check_permission('snapshot_text'), mode != 'SAFE')
        for value in ['File written', '{}', '{"status":"verified"}', '[]', '{"tool":"create_text","status":"conflict","effects":"possible"}']:
            self.assertEqual(llm._tool_result_status(value, 'create_text'), 'error')
        self.assertEqual(llm._tool_result_status('{"tool":"snapshot_text","status":"ok","effects":"none","complete":true}', 'snapshot_text'), 'error')
        result = ws.create_text(str(self.root), 'report.txt', 'new')
        self.assertEqual(llm._tool_result_status(result, 'create_text'), 'ok')
        self.assertEqual(llm._tool_result_status(ws.create_text(str(self.root), 'report.txt', 'new'), 'create_text'), 'conflict')


if __name__ == '__main__':
    unittest.main()
