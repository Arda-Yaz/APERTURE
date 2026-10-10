import importlib.util
import inspect
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import isolation_support as safety

safety.install_guards()

import experience
import memory


class StoreInitializationTests(unittest.TestCase):
    def test_imports_do_not_connect_or_create_directories(self):
        for name in ("memory", "experience", "self_state", "relationship",
                     "reflection", "llm", "task_controller"):
            with self.subTest(module=name):
                spec = importlib.util.spec_from_file_location(
                    f"isolated_{name}", ROOT / "app" / f"{name}.py"
                )
                module = importlib.util.module_from_spec(spec)
                with patch("sqlite3.connect") as connect, patch.object(Path, "mkdir") as mkdir:
                    spec.loader.exec_module(module)
                connect.assert_not_called()
                mkdir.assert_not_called()
                if name in ("memory", "experience"):
                    self.assertIsNone(module._STORE)

    def test_default_paths_and_lazy_wrapper_reuse_are_preserved(self):
        cases = (
            (memory, "MemoryStore", "aperture_memory.db", "get_memory_record", ("missing",)),
            (experience, "ExperienceStore", "aperture_experience.db", "get_latest_episode_id", ()),
        )
        for module, class_name, filename, wrapper, args in cases:
            with self.subTest(store=class_name):
                expected = ROOT / "data" / filename
                constructor = getattr(module, class_name)
                self.assertEqual(module.DB_PATH, expected)
                self.assertEqual(inspect.signature(constructor).parameters["db_path"].default, expected)
                with patch.object(module, "_STORE", None), patch.object(module, class_name) as factory:
                    factory.assert_not_called()
                    getattr(module, wrapper)(*args)
                    getattr(module, wrapper)(*args)
                    factory.assert_called_once_with()
                    self.assertIs(module._STORE, factory.return_value)

    def test_lazy_wrappers_can_initialize_and_reuse_isolated_stores(self):
        for module, class_name, wrapper, args in (
            (memory, "MemoryStore", "get_memory_record", ("missing",)),
            (experience, "ExperienceStore", "get_latest_episode_id", ()),
        ):
            with self.subTest(store=class_name), safety.TestTemporaryDirectory() as directory:
                path = Path(directory) / "isolated.db"
                constructor = getattr(module, class_name)
                with patch.object(module, "_STORE", None), patch.object(
                    module, class_name, side_effect=lambda: constructor(path)
                ) as factory:
                    self.assertIsNone(getattr(module, wrapper)(*args))
                    self.assertTrue(path.is_file())
                    self.assertEqual(module._STORE.db_path, path)
                    self.assertIsNone(getattr(module, wrapper)(*args))
                    factory.assert_called_once_with()


class IsolationGuardTests(unittest.TestCase):
    def test_in_memory_sqlite_remains_available(self):
        connection = sqlite3.connect(":memory:")
        try:
            self.assertEqual(connection.execute("SELECT 1").fetchone(), (1,))
        finally:
            connection.close()

    def test_only_registered_temporary_roots_are_allowed(self):
        with safety.TestTemporaryDirectory() as directory:
            path = Path(directory) / "allowed.db"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE test (value TEXT)")
            connection.close()
            with safety.expected_violation(), self.assertRaises(AssertionError):
                sqlite3.connect(Path(directory).parent / "unregistered.db")
        with safety.expected_violation(), self.assertRaises(AssertionError):
            sqlite3.connect(path)

    def test_production_and_uri_connections_never_reach_sqlite(self):
        for database, kwargs in (
            (memory.DB_PATH, {}),
            (experience.DB_PATH, {}),
            ("file::memory:?cache=shared", {"uri": True}),
        ):
            with self.subTest(database=str(database)), patch.object(safety, "_CONNECT") as connect:
                with safety.expected_violation(), self.assertRaises(AssertionError):
                    sqlite3.connect(database, **kwargs)
                connect.assert_not_called()

    def test_default_constructors_are_blocked_before_data_directory_access(self):
        for constructor in (memory.MemoryStore, experience.ExperienceStore):
            with self.subTest(store=constructor.__name__), patch.object(safety, "_MKDIR") as mkdir:
                with safety.expected_violation(), self.assertRaises(AssertionError):
                    constructor()
                mkdir.assert_not_called()

    def test_unmocked_inference_is_blocked_in_all_captured_aliases(self):
        import ollama
        calls = [ollama.chat, ollama.Client.chat, ollama.AsyncClient.chat]
        for name in ("llm", "task_controller", "reflection", "self_state", "relationship"):
            calls.append(__import__(name).ollama_chat)
        for call in calls:
            with self.subTest(call=repr(call)):
                with safety.expected_violation(), self.assertRaises(AssertionError):
                    call(model="test", messages=[])

    def test_swallowed_guard_violations_fail_the_test_result(self):
        class SwallowsFailure(unittest.TestCase):
            def runTest(self):
                try:
                    sqlite3.connect(memory.DB_PATH)
                except Exception:
                    pass

        # A separate violation ledger lets us inspect a deliberate failing test
        # without suppressing or clearing any actual suite violation.
        with patch.object(safety, "_VIOLATIONS", []):
            result = unittest.TestResult()
            SwallowsFailure().run(result)
            self.assertFalse(result.wasSuccessful())
            self.assertEqual(len(result.failures), 1)
            self.assertIn("Test isolation violations", result.failures[0][1])

    def test_discovery_includes_all_six_restored_tests_once(self):
        restored = {
            "test_automatic_reflection_does_not_use_raw_actor_for_self_memory",
            "test_reflection_self_evidence_uses_latest_aperture_turn_only",
            "test_dynamic_self_current_turn_excludes_old_assistant_evidence",
            "test_ordinary_fact_can_semantically_resolve_to_null",
            "test_semantic_relationship_candidate_is_accepted",
            "test_action_turn_is_not_relationship_skipped",
        }
        loader = unittest.TestLoader()
        suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py")

        def test_ids(items):
            for item in items:
                if isinstance(item, unittest.TestSuite):
                    yield from test_ids(item)
                else:
                    yield item.id().rsplit(".", 1)[-1]

        names = list(test_ids(suite))
        self.assertEqual(loader.errors, [])
        for name in restored:
            self.assertEqual(names.count(name), 1, name)


if __name__ == "__main__":
    unittest.main()
