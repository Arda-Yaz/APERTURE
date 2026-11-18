"""Terminal results must not report failed PowerShell cmdlets as success."""
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from isolation_support import TestTemporaryDirectory, install_guards

install_guards()

import tools


@unittest.skipUnless(shutil.which('powershell.exe'), 'Windows PowerShell required')
class TerminalErrorTests(unittest.TestCase):
    def setUp(self):
        directory = TestTemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name)
        (self.home / 'Documents' / 'Aperture').mkdir(parents=True)
        home = patch.object(tools.Path, 'home', return_value=self.home)
        home.start()
        self.addCleanup(home.stop)

    def test_missing_file_stops_before_following_command(self):
        result = tools.run_terminal(
            "Get-Item -LiteralPath './missing.txt'; Write-Output 'SHOULD_NOT_RUN'")
        self.assertTrue(result.startswith('Exit code: 1'), result)
        self.assertNotIn('STDOUT:', result)
        self.assertIn('STDERR:', result)

    def test_explicit_error_is_not_masked_by_later_success(self):
        result = tools.run_terminal("Write-Error 'operation failed'; Write-Output 'done'")
        self.assertTrue(result.startswith('Exit code: 1'), result)
        self.assertNotIn('STDOUT:', result)
        self.assertIn('operation failed', result)

    def test_successful_sequence_keeps_its_output(self):
        result = tools.run_terminal("Write-Output 'first'; Write-Output 'second'")
        self.assertEqual(result, 'Exit code: 0\nSTDOUT:\nfirst\nsecond')


if __name__ == '__main__':
    unittest.main()
