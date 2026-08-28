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


from experience import (  # noqa: E402
    ExperienceStore,
)


class ExperienceStoreTests(
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
            / "experience.db"
        )

        self.store = (
            ExperienceStore(
                self.db_path
            )
        )

        self.episode_id = (
            self.store.create_episode(
                metadata={
                    "test": True,
                }
            )
        )


    def tearDown(
        self,
    ):

        self.temp_dir.cleanup()


    def test_event_order_is_append_only(
        self,
    ):

        first = (
            self.store.append_event(
                episode_id=(
                    self.episode_id
                ),
                event_type=(
                    "user_message"
                ),
                event_class=(
                    "observed"
                ),
                actor="arda",
                content="first",
            )
        )

        second = (
            self.store.append_event(
                episode_id=(
                    self.episode_id
                ),
                event_type=(
                    "assistant_message"
                ),
                event_class=(
                    "observed"
                ),
                actor="aperture",
                content="second",
            )
        )

        events = (
            self.store
            .episode_events(
                self.episode_id
            )
        )

        self.assertEqual(
            [
                event["content"]
                for event in events
            ],
            [
                "first",
                "second",
            ],
        )

        self.assertLess(
            first["seq"],
            second["seq"],
        )

        self.assertFalse(
            hasattr(
                self.store,
                "update_event",
            )
        )

        self.assertFalse(
            hasattr(
                self.store,
                "delete_event",
            )
        )


    def test_sensitive_metadata_is_redacted(
        self,
    ):

        event = (
            self.store.append_event(
                episode_id=(
                    self.episode_id
                ),
                event_type=(
                    "tool_call"
                ),
                event_class=(
                    "observed"
                ),
                actor="aperture",
                content="example",
                metadata={
                    "api_key":
                        "SECRET_VALUE",

                    "nested": {
                        "token":
                            "TOKEN_VALUE",

                        "safe":
                            "visible",
                    },
                },
            )
        )

        loaded = (
            self.store.events_by_ids(
                [
                    event["id"]
                ]
            )[0]
        )

        self.assertEqual(
            loaded[
                "metadata"
            ][
                "api_key"
            ],
            "[REDACTED]",
        )

        self.assertEqual(
            loaded[
                "metadata"
            ][
                "nested"
            ][
                "token"
            ],
            "[REDACTED]",
        )

        self.assertEqual(
            loaded[
                "metadata"
            ][
                "nested"
            ][
                "safe"
            ],
            "visible",
        )


    def test_recent_observed_messages_ignore_derived_events(
        self,
    ):

        self.store.append_event(
            episode_id=(
                self.episode_id
            ),
            event_type=(
                "user_message"
            ),
            event_class=(
                "observed"
            ),
            actor="arda",
            content="one",
        )

        self.store.append_event(
            episode_id=(
                self.episode_id
            ),
            event_type=(
                "self_state_update"
            ),
            event_class=(
                "derived"
            ),
            actor="self_state",
            content="derived",
        )

        self.store.append_event(
            episode_id=(
                self.episode_id
            ),
            event_type=(
                "assistant_message"
            ),
            event_class=(
                "observed"
            ),
            actor="aperture",
            content="two",
        )

        events = (
            self.store
            .recent_observed_message_events(
                self.episode_id,
                limit=6,
            )
        )

        self.assertEqual(
            [
                event[
                    "event_type"
                ]
                for event
                in events
            ],
            [
                "user_message",
                "assistant_message",
            ],
        )


if __name__ == "__main__":
    unittest.main()