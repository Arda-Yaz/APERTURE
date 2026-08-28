from pprint import pprint

from experience import (
    get_latest_episode_id,
    get_episode_events,
)


episode_id = (
    get_latest_episode_id()
)


if episode_id is None:

    print(
        "No experience episodes found."
    )

    raise SystemExit(0)


print(
    f"\nEPISODE: "
    f"{episode_id}\n"
)


events = (
    get_episode_events(
        episode_id,
        limit=500,
    )
)


for event in events:

    print(
        "=" * 72
    )

    print(
        f"SEQ:         "
        f"{event['seq']}"
    )

    print(
        f"EVENT ID:    "
        f"{event['id']}"
    )

    print(
        f"TURN ID:     "
        f"{event['turn_id']}"
    )

    print(
        f"CLASS:       "
        f"{event['event_class']}"
    )

    print(
        f"TYPE:        "
        f"{event['event_type']}"
    )

    print(
        f"ACTOR:       "
        f"{event['actor']}"
    )

    print(
        f"CREATED AT:  "
        f"{event['created_at']}"
    )

    print(
        "\nCONTENT:"
    )

    print(
        event["content"]
    )

    if event["metadata"]:

        print(
            "\nMETADATA:"
        )

        pprint(
            event["metadata"],
            sort_dicts=False,
            width=120,
        )

    print()
    