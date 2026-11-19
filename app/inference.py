"""Small synchronous Ollama boundary with bounded calls and safe failures."""
import json
from collections.abc import Mapping

import httpx
import ollama
from pydantic import ValidationError

from config import resolve_inference_settings


class InferenceError(RuntimeError):
    """A transport or response failure; never include server payloads or prompts."""

    _MESSAGES = {
        'connection': 'Cannot connect to Ollama.',
        'missing_model': 'The configured Ollama model is unavailable.',
        'timeout': 'Ollama inference timed out.',
        'malformed_response': 'Ollama returned a malformed response.',
        'server_error': 'Ollama could not complete inference.',
    }

    def __init__(self, code, *, role='actor'):
        self.code = code
        self.role = role
        super().__init__(f'{role}: {self._MESSAGES[code]} [{code}]')


def _validate_response(response, role):
    # All callers consume the SDK's attribute API; do not invent missing content.
    message = getattr(response, 'message', None)
    if message is None:
        raise InferenceError('malformed_response', role=role)
    content = getattr(message, 'content', None)
    thinking = getattr(message, 'thinking', None)
    calls = getattr(message, 'tool_calls', None)
    if ((content is not None and not isinstance(content, str))
            or (thinking is not None and not isinstance(thinking, str))
            or (calls is not None and not isinstance(calls, list))
            or (content is None and thinking is None and calls is None)):
        raise InferenceError('malformed_response', role=role)
    for call in calls or []:
        function = getattr(call, 'function', None)
        name = getattr(function, 'name', None)
        arguments = getattr(function, 'arguments', None)
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
            raise InferenceError('malformed_response', role=role)
    return response


def chat(*, role='actor', **kwargs):
    settings = resolve_inference_settings(role)
    kwargs.setdefault('model', settings.model)
    if kwargs.get('stream'):
        raise ValueError('APERTURE uses synchronous non-streaming inference.')
    supplied_options = kwargs.get('options')
    if supplied_options is None:
        options = {}
    elif isinstance(supplied_options, Mapping):
        options = dict(supplied_options)
    else:
        options = supplied_options.model_dump(exclude_none=True)
    requested_limit = options.get('num_predict', settings.num_predict)
    if type(requested_limit) is not int or requested_limit <= 0:
        raise ValueError('num_predict must be a strictly positive integer.')
    # Configuration is a ceiling, including when actor/retrieval supply defaults.
    options['num_predict'] = min(requested_limit, settings.num_predict)
    kwargs['options'] = options
    # One client per call, deterministically closed. HTTPX has no transport retries.
    with ollama.Client(host=settings.host, timeout=settings.timeout_seconds,
                       follow_redirects=False) as client:
        for attempt in range(settings.retries + 1):
            try:
                return _validate_response(client.chat(**kwargs), role)
            except (TimeoutError, httpx.TimeoutException):
                failure = InferenceError('timeout', role=role)
            except (ConnectionError, httpx.NetworkError):
                failure = InferenceError('connection', role=role)
            except ollama.ResponseError as exc:
                code = 'missing_model' if exc.status_code == 404 else 'server_error'
                failure = InferenceError(code, role=role)
                if code == 'missing_model' or (exc.status_code is not None and exc.status_code < 500):
                    raise failure from None
            except (json.JSONDecodeError, ValidationError, httpx.ProtocolError, httpx.DecodingError):
                raise InferenceError('malformed_response', role=role) from None
            if attempt == settings.retries:
                raise failure from None


def role_chat(default_role):
    """Keep role modules' patchable ``ollama_chat`` seam and explicit MODEL API."""
    def call(**kwargs):
        role = kwargs.pop('role', default_role)
        return chat(role=role, **kwargs)
    return call
