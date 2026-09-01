import sqlite3
import sys
import tempfile
import unittest

from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

APP_DIR = (
    ROOT
    / "app"
)

sys.path.insert(
    0,
    str(APP_DIR),
)


from memory import MemoryStore  # noqa: E402


class MemoryTemporalTests(
    unittest.TestCase
):

    def setUp(
        self,
    ):

        self.temp_dir = (
            tempfile.TemporaryDirectory()
        )

        self.db_path = (
            Path(
                self.temp_dir.name
            )
            / "memory.db"
        )

        self.store = (
            MemoryStore(
                self.db_path
            )
        )


    def tearDown(
        self,
    ):

        self.temp_dir.cleanup()


    def test_supersede_preserves_old_memory(
        self,
    ):

        old = (
            self.store.remember(
                content=(
                    "I currently prefer Python."
                ),
                category="preference",
                importance=3,
                subject="aperture",
                source="reflection",
                evidence_event_ids=[
                    "evt_old",
                ],
            )
        )

        new = (
            self.store.supersede(
                old["id"],
                content=(
                    "I currently prefer Java."
                ),
                category="preference",
                importance=3,
                subject="aperture",
                source="reflection",
                evidence_event_ids=[
                    "evt_new",
                ],
            )
        )

        old_record = (
            self.store.get_by_id(
                old["id"]
            )
        )

        new_record = (
            self.store.get_by_id(
                new["id"]
            )
        )

        self.assertIsNotNone(
            old_record
        )
        if old_record is None:
            self.fail("Old memory record was not found")

        self.assertIsNotNone(
            new_record
        )
        if new_record is None:
            self.fail("New memory record was not found")

        self.assertEqual(
            old_record["active"],
            0,
        )

        self.assertIsNotNone(
            old_record[
                "valid_until"
            ]
        )

        self.assertEqual(
            new_record["active"],
            1,
        )

        self.assertIsNone(
            new_record[
                "valid_until"
            ]
        )

        self.assertEqual(
            new_record[
                "supersedes_memory_id"
            ],
            old["id"],
        )

        self.assertEqual(
            new["status"],
            "superseded",
        )


    def test_current_context_hides_superseded_memory(
        self,
    ):

        old = (
            self.store.remember(
                content=(
                    "I currently prefer Python."
                ),
                category="preference",
                subject="aperture",
            )
        )

        new = (
            self.store.supersede(
                old["id"],
                content=(
                    "I currently prefer Java."
                ),
                category="preference",
                importance=3,
                source="reflection",
                subject="aperture",
            )
        )

        current = (
            self.store
            .top_memories_for_subject(
                "aperture",
                limit=20,
            )
        )

        ids = {
            memory["id"]
            for memory
            in current
        }

        self.assertNotIn(
            old["id"],
            ids,
        )

        self.assertIn(
            new["id"],
            ids,
        )


    def test_preference_can_return_to_old_text(
        self,
    ):

        python_v1 = (
            self.store.remember(
                content=(
                    "I currently prefer Python."
                ),
                category="preference",
                subject="aperture",
            )
        )

        java = (
            self.store.supersede(
                python_v1["id"],
                content=(
                    "I currently prefer Java."
                ),
                category="preference",
                importance=3,
                source="reflection",
                subject="aperture",
            )
        )

        python_v2 = (
            self.store.supersede(
                java["id"],
                content=(
                    "I currently prefer Python."
                ),
                category="preference",
                importance=3,
                source="reflection",
                subject="aperture",
            )
        )

        self.assertNotEqual(
            python_v1["id"],
            python_v2["id"],
        )

        old_python = (
            self.store.get_by_id(
                python_v1["id"]
            )
        )

        old_java = (
            self.store.get_by_id(
                java["id"]
            )
        )

        current_python = (
            self.store.get_by_id(
                python_v2["id"]
            )
        )

        self.assertIsNotNone(
            old_python,
        )
        if old_python is None:
            self.fail("Old Python memory record was not found")

        self.assertIsNotNone(
            old_java,
        )
        if old_java is None:
            self.fail("Old Java memory record was not found")

        self.assertIsNotNone(
            current_python,
        )
        if current_python is None:
            self.fail("Current Python memory record was not found")

        self.assertEqual(
            old_python["active"],
            0,
        )

        self.assertEqual(
            old_java["active"],
            0,
        )

        self.assertEqual(
            current_python["active"],
            1,
        )

        self.assertEqual(
            current_python[
                "supersedes_memory_id"
            ],
            java["id"],
        )


    def test_same_content_is_not_supersession(
        self,
    ):

        original = (
            self.store.remember(
                content=(
                    "I currently prefer Python."
                ),
                category="preference",
                subject="aperture",
                evidence_event_ids=[
                    "evt_one",
                ],
            )
        )

        result = (
            self.store.supersede(
                original["id"],
                content=(
                    "I currently prefer Python."
                ),
                category="preference",
                importance=4,
                source="reflection",
                subject="aperture",
                evidence_event_ids=[
                    "evt_two",
                ],
            )
        )

        self.assertEqual(
            result["status"],
            "existing",
        )

        self.assertEqual(
            result["id"],
            original["id"],
        )

        record = (
            self.store.get_by_id(
                original["id"]
            )
        )

        self.assertIsNotNone(
            record,
        )
        assert record is not None

        self.assertEqual(
            record["active"],
            1,
        )

        self.assertIsNone(
            record[
                "valid_until"
            ]
        )

        self.assertEqual(
            record[
                "evidence_event_ids"
            ],
            [
                "evt_one",
                "evt_two",
            ],
        )


    def test_cross_subject_supersession_is_rejected(
        self,
    ):

        user_memory = (
            self.store.remember(
                content=(
                    "Arda prefers Python."
                ),
                category="preference",
                subject="user",
            )
        )

        with self.assertRaises(
            ValueError
        ):

            self.store.supersede(
                user_memory["id"],
                content=(
                    "I prefer Java."
                ),
                category="preference",
                importance=3,
                source="reflection",
                subject="aperture",
            )


class LegacyMemoryMigrationTests(
    unittest.TestCase
):

    def setUp(
        self,
    ):

        self.temp_dir = (
            tempfile.TemporaryDirectory()
        )

        self.db_path = (
            Path(
                self.temp_dir.name
            )
            / "legacy_memory.db"
        )


    def tearDown(
        self,
    ):

        self.temp_dir.cleanup()


    def test_old_unique_schema_migrates(
        self,
    ):

        timestamp = (
            "2026-01-01T00:00:00+00:00"
        )

        connection = (
            sqlite3.connect(
                self.db_path
            )
        )

        try:

            connection.execute(
                """
                CREATE TABLE memories (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    subject TEXT NOT NULL DEFAULT 'user',
                    evidence_event_ids_json
                        TEXT NOT NULL DEFAULT '[]',
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(
                        subject,
                        normalized_content
                    )
                )
                """
            )

            connection.execute(
                """
                INSERT INTO memories (
                    id,
                    category,
                    content,
                    normalized_content,
                    importance,
                    source,
                    subject,
                    evidence_event_ids_json,
                    active,
                    created_at,
                    updated_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    "legacy_python",
                    "preference",
                    (
                        "I currently prefer "
                        "Python."
                    ),
                    (
                        "i currently prefer "
                        "python."
                    ),
                    3,
                    "reflection",
                    "aperture",
                    "[]",
                    1,
                    timestamp,
                    timestamp,
                ),
            )

            connection.commit()

        finally:
            connection.close()

        store = (
            MemoryStore(
                self.db_path
            )
        )

        migrated = (
            store.get_by_id(
                "legacy_python"
            )
        )

        self.assertIsNotNone(
            migrated
        )
        if migrated is None:
            self.fail(
                "Legacy record was not migrated."
            )

        self.assertEqual(
            migrated[
                "valid_from"
            ],
            timestamp,
        )

        java = (
            store.supersede(
                "legacy_python",
                content=(
                    "I currently prefer Java."
                ),
                category="preference",
                importance=3,
                source="reflection",
                subject="aperture",
            )
        )

        python_again = (
            store.supersede(
                java["id"],
                content=(
                    "I currently prefer Python."
                ),
                category="preference",
                importance=3,
                source="reflection",
                subject="aperture",
            )
        )

        self.assertNotEqual(
            python_again["id"],
            "legacy_python",
        )


if __name__ == "__main__":
    unittest.main()