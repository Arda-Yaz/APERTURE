import os
import math
import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


INFERENCE_ROLES = ('actor', 'retrieval', 'reflection', 'self_state', 'relationship', 'task_controller')
DEFAULT_MODEL = 'qwen3.5:4b'


class ConfigurationError(ValueError):
    """Invalid local configuration; messages never include supplied values."""


@dataclass(frozen=True)
class InferenceSettings:
    host: str
    model: str
    timeout_seconds: float
    retries: int
    num_predict: int

    def sanitized(self):
        endpoint = urlsplit(self.host)
        # Paths, userinfo, queries and fragments may carry credentials.
        hostname = endpoint.hostname
        if ':' in hostname:
            hostname = f'[{hostname}]'
        origin = hostname + (f':{endpoint.port}' if endpoint.port else '')
        return {'endpoint': urlunsplit((endpoint.scheme, origin, '', '', '')),
                'model': self.model, 'timeout_seconds': self.timeout_seconds,
                'retries': self.retries, 'num_predict': self.num_predict}


def _setting(name, role, default):
    role_name = f'{name}_{role.upper()}'
    return os.environ.get(role_name, os.environ.get(name, str(default)))


def _number(name, role, default, *, integer=False, minimum=0, maximum=3600):
    value = _setting(name, role, default)
    try:
        parsed = int(value) if integer else float(value)
    except (ValueError, TypeError):
        raise ConfigurationError(f'{name} must be a valid finite number.') from None
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ConfigurationError(f'{name} must be between {minimum} and {maximum}.')
    return parsed


def resolve_inference_settings(role='actor'):
    """Resolve environment overrides without changing environment or local files."""
    if role not in INFERENCE_ROLES:
        raise ConfigurationError('Unknown inference role.')
    model = _setting('APERTURE_MODEL', role, DEFAULT_MODEL).strip()
    if not model or len(model) > 200 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/-]*', model):
        raise ConfigurationError('APERTURE_MODEL must be a nonempty model identifier.')
    host = os.environ.get('APERTURE_OLLAMA_HOST', os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')).strip()
    if '://' not in host:
        host = 'http://' + host
    try:
        endpoint = urlsplit(host)
        valid_host = endpoint.scheme in ('http', 'https') and endpoint.hostname and endpoint.port != 0
        if not valid_host or any(char.isspace() for char in host):
            raise ValueError
    except ValueError:
        raise ConfigurationError('APERTURE_OLLAMA_HOST / OLLAMA_HOST must be a valid HTTP(S) endpoint.') from None
    timeout = _number('APERTURE_INFERENCE_TIMEOUT_SECONDS', role, 120, minimum=0.01)
    retries = _number('APERTURE_INFERENCE_RETRIES', role, 0, integer=True, maximum=2)
    if role == 'actor':
        if 'APERTURE_INFERENCE_RETRIES_ACTOR' in os.environ and retries:
            raise ConfigurationError('APERTURE_INFERENCE_RETRIES_ACTOR must be zero; the actor loop owns recovery.')
        retries = 0
    num_predict = _number('APERTURE_INFERENCE_NUM_PREDICT', role, 4096 if role == 'actor' else 2048,
                          integer=True, minimum=1, maximum=32768)
    return InferenceSettings(host, model, timeout, retries, num_predict)


def resolved_inference_settings():
    """Safe startup/debug settings: no endpoint credentials or arbitrary env data."""
    return {role: resolve_inference_settings(role).sanitized() for role in INFERENCE_ROLES}


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

REQUIRE_REFLECTION_USER_MEMORY_APPROVAL = _env_flag(
    "APERTURE_REQUIRE_REFLECTION_USER_MEMORY_APPROVAL",
)
