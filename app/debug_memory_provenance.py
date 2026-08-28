from pprint import pprint

from memory import (
    get_recent_memory_records,
)

from experience import (
    get_events_by_ids,
)


memories = (
    get_recent_memory_records(
        limit=10,
    )
)


if not memories:

    print(
        "No memories found."
    )

    raise SystemExit(0)


for memory in memories:

    print(
        "=" * 80
    )

    print(
        f"MEMORY ID:   "
        f"{memory['id']}"
    )

    print(
        f"SUBJECT:     "
        f"{memory['subject']}"
    )

    print(
        f"CATEGORY:    "
        f"{memory['category']}"
    )

    print(
        f"IMPORTANCE:  "
        f"{memory['importance']}"
    )

    print(
        f"SOURCE:      "
        f"{memory['source']}"
    )

    print(
        f"ACTIVE:      "
        f"{memory['active']}"
    )

    print(
        "\nCONTENT:"
    )

    print(
        memory["content"]
    )


    evidence_ids = (
        memory.get(
            "evidence_event_ids",
            [],
        )
    )

    print(
        "\nEVIDENCE EVENT IDS:"
    )

    pprint(
        evidence_ids
    )


    if evidence_ids:

        events = (
            get_events_by_ids(
                evidence_ids
            )
        )

        print(
            "\nSOURCE EXPERIENCE:"
        )

        for event in events:

            print(
                "-" * 60
            )

            print(
                f"{event['event_type']} "
                f"[{event['actor']}] "
                f"{event['id']}"
            )

            print(
                event["content"]
            )

    print()