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


from memory import (  # noqa: E402
    MemoryStore,
)


class MemoryProvenanceTests(
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


    def test_new_memory_keeps_evidence(
        self,
    ):

        result = (
            self.store.remember(
                content=(
                    "I prefer Python."
                ),
                category=(
                    "preference"
                ),
                importance=3,
                subject=(
                    "aperture"
                ),
                source=(
                    "reflection"
                ),
                evidence_event_ids=[
                    "evt_one",
                    "evt_two",
                ],
            )
        )

        stored = (
            self.store.get_by_id(
                result["id"]
            )
        )

        if stored is None:
            self.fail(
                "Stored memory was not found"
            )

        self.assertEqual(
            stored[
                "evidence_event_ids"
            ],
            [
                "evt_one",
                "evt_two",
            ],
        )


    def test_existing_memory_merges_evidence(
        self,
    ):

        first = (
            self.store.remember(
                content=(
                    "I prefer Python."
                ),
                category=(
                    "preference"
                ),
                importance=3,
                subject=(
                    "aperture"
                ),
                source=(
                    "reflection"
                ),
                evidence_event_ids=[
                    "evt_one",
                    "evt_two",
                ],
            )
        )

        second = (
            self.store.remember(
                content=(
                    "I prefer Python."
                ),
                category=(
                    "preference"
                ),
                importance=3,
                subject=(
                    "aperture"
                ),
                source=(
                    "reflection"
                ),
                evidence_event_ids=[
                    "evt_two",
                    "evt_three",
                ],
            )
        )

        self.assertEqual(
            first["id"],
            second["id"],
        )

        stored = (
            self.store.get_by_id(
                first["id"]
            )
        )

        if stored is None:
            self.fail(
                "Stored memory was not found"
            )

        self.assertEqual(
            stored[
                "evidence_event_ids"
            ],
            [
                "evt_one",
                "evt_two",
                "evt_three",
            ],
        )


    def test_same_content_different_subjects_remains_separate(
        self,
    ):

        user_memory = (
            self.store.remember(
                content=(
                    "Prefers Python."
                ),
                category=(
                    "preference"
                ),
                subject="user",
                evidence_event_ids=[
                    "evt_user",
                ],
            )
        )

        self_memory = (
            self.store.remember(
                content=(
                    "Prefers Python."
                ),
                category=(
                    "preference"
                ),
                subject=(
                    "aperture"
                ),
                evidence_event_ids=[
                    "evt_self",
                ],
            )
        )

        self.assertNotEqual(
            user_memory["id"],
            self_memory["id"],
        )


if __name__ == "__main__":
    unittest.main()