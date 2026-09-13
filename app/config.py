import os


def _env_flag(
    name: str,
    default: bool = False,
) -> bool:

    fallback = (
        "1"
        if default
        else "0"
    )

    value = (
        os.getenv(
            name,
            fallback,
        )
        .strip()
        .casefold()
    )

    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


# ============================================================
# EXPERIMENTAL COGNITION
# ============================================================

ENABLE_DYNAMIC_SELF = (
    _env_flag(
        "APERTURE_DYNAMIC_SELF",
        False,
    )
)

ENABLE_RELATIONSHIP = (
    _env_flag(
        "APERTURE_RELATIONSHIP",
        False,
    )
)

ENABLE_SELF_CONSOLIDATION = (
    _env_flag(
        "APERTURE_SELF_CONSOLIDATION",
        False,
    )
)