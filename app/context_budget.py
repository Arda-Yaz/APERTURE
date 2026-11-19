"""Conservative request budgeting, without claiming access to a model tokenizer.

One UTF-8 byte is reserved per estimated token, plus JSON/framing overhead.
This deliberately overestimates ordinary text; Ollama's chat template/tokenizer
can still differ. Output reservation (num_predict) is separate from num_ctx.
"""
import json

from ollama._utils import convert_function_to_tool


class ActorContextError(ValueError):
    pass


def plain_message(message):
    if hasattr(message, 'model_dump'):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return dict(message)
    # Simple response doubles are also used by the isolated regression suite.
    result = {'role': getattr(message, 'role', 'assistant'),
              'content': getattr(message, 'content', '') or ''}
    calls = getattr(message, 'tool_calls', None)
    if calls:
        result['tool_calls'] = [
            {'function': {'name': call.function.name,
                          'arguments': call.function.arguments}} for call in calls]
    return result


def tool_schemas(tools):
    schemas = []
    for tool in tools or ():
        converted = convert_function_to_tool(tool) if callable(tool) else tool
        schemas.append(converted.model_dump(exclude_none=True)
                       if hasattr(converted, 'model_dump') else converted)
    return schemas


def estimate_request(messages, tools=None):
    payload = {'messages': messages, 'tools': tool_schemas(tools)}
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode('utf-8')) + 256 + 32 * len(messages)


def prepare_messages(messages, tools=None, options=None):
    options = options or {}
    num_ctx = int(options.get('num_ctx', 32768))
    num_predict = int(options.get('num_predict', 4096))
    if num_ctx <= 0 or num_predict <= 0 or num_predict >= num_ctx:
        raise ActorContextError('Actor requires positive num_ctx and a smaller positive num_predict.')
    messages = [plain_message(message) for message in messages]
    # Only runtime context explicitly marked by our caller is optional. System
    # instructions, including permission/controller instructions, stay whole.
    optional = []
    for message in messages:
        suffix = message.pop('_aperture_optional_suffix', None)
        whole = bool(message.pop('_aperture_optional_context', False))
        optional.append(suffix if isinstance(suffix, str) and suffix
                        and str(message.get('content', '')).endswith(suffix) else whole)
    current = max((i for i, message in enumerate(messages) if message.get('role') == 'user'), default=-1)
    groups = []
    for i, message in enumerate(messages):
        if i >= current or current < 0:
            break
        if message.get('role') == 'user':
            groups.append([])
        if groups and message.get('role') != 'system':
            groups[-1].append(i)
    kept = set(range(len(messages)))
    def render():
        return [message for i, message in enumerate(messages) if i in kept]
    def fits():
        return estimate_request(render(), tools) + num_predict <= num_ctx
    for group in groups:
        if fits():
            break
        kept.difference_update(group)
    for i, removable in enumerate(optional):
        if fits():
            break
        if isinstance(removable, str):
            messages[i]['content'] = messages[i]['content'][:-len(removable)]
        elif removable:
            kept.discard(i)
    if not fits():
        raise ActorContextError('Actor context exceeds num_ctx after reserving output; current request, instructions and tool groups cannot be safely reduced.')
    return render()
