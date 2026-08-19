from ollama import chat as ollama_chat
from persona import build_persona_context
from reflection import maybe_reflect



from tools import (
    list_directory,
    read_file,
    write_file,
    open_app,
    run_terminal,
)

from permissions import check_permission
from task_controller import is_task_complete

from memory import (
    save_memory,
    save_self_memory,
    search_memory,
    forget_memory,
    build_memory_context,
)

MODEL = "qwen3:8b"
MAX_STEPS = 8


TOOLS = [
    list_directory,
    read_file,
    write_file,
    open_app,
    run_terminal,
    save_memory,
    save_self_memory,
    search_memory,
    forget_memory,
]


TOOL_MAP = {
    "list_directory": list_directory,
    "read_file": read_file,
    "write_file": write_file,
    "open_app": open_app,
    "run_terminal": run_terminal,
    "save_memory": save_memory,
    "search_memory": search_memory,
    "forget_memory": forget_memory,
    "save_self_memory": save_self_memory,
}

NON_ACTION_TOOLS = {
    "save_memory",
    "save_self_memory",
    "search_memory",
    "forget_memory",
}


def get_current_goal(messages):
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return message.get("content", "")

    return ""

def inject_runtime_context(
    messages,
    user_message: str,
):
    runtime_messages = list(messages)

    persona_context = build_persona_context(
        user_message=user_message,
    )

    memory_context = build_memory_context()

    runtime_context = (
        persona_context
        + "\n\n"
        + memory_context
    )

    if (
        runtime_messages
        and isinstance(runtime_messages[0], dict)
        and runtime_messages[0].get("role") == "system"
    ):
        system_message = dict(runtime_messages[0])

        system_message["content"] = (
            system_message.get("content", "")
            + "\n\n"
            + runtime_context
        )

        runtime_messages[0] = system_message

    else:
        runtime_messages.insert(
            0,
            {
                "role": "system",
                "content": runtime_context,
            },
        )

    return runtime_messages



def finalize_answer(
    messages,
    answer: str,
    *,
    used_action_tool: bool,
    memory_operation_used: bool,
):
    messages.append({
        "role": "assistant",
        "content": answer,
    })

    reflection_result = maybe_reflect(
        messages,
        used_action_tool=used_action_tool,
        memory_operation_used=memory_operation_used,
    )

    if reflection_result:
        print(
            f"\n[REFLECTION] "
            f"{reflection_result}"
        )

    return answer



def chat(messages):
    goal = get_current_goal(messages)

    # Agent'ın iç çalışma geçmişi.
    # Tool sonuçları ve controller mesajları kalıcı sohbeti kirletmez.
    working_messages = inject_runtime_context(
    messages,
    user_message=goal,
)
    observations = []
    used_action_tool = False
    memory_operation_used = False

    for step in range(MAX_STEPS):

        response = ollama_chat(
            model=MODEL,
            messages=working_messages,
            tools=TOOLS,
            think=False,
        )

        working_messages.append(response.message)

        # --------------------------------
        # MODEL TOOL ÇAĞIRMADI
        # --------------------------------
        if not response.message.tool_calls:

            answer = response.message.content or ""

            # Normal sohbet
            if not used_action_tool:
                return finalize_answer(
                    messages,
                    answer,
                    used_action_tool=used_action_tool,
                    memory_operation_used=memory_operation_used,
                )

            # Tool kullanıldıysa görev gerçekten tamamlandı mı?
            complete = is_task_complete(
                goal=goal,
                answer=answer,
                observations="\n\n".join(observations),
            )

            if complete:
                return finalize_answer(
                    messages,
                    answer,
                    used_action_tool=used_action_tool,
                    memory_operation_used=memory_operation_used,
                )

            # Controller cevabı reddetti.
            working_messages.append({
                "role": "system",
                "content": f"""
TASK CONTROLLER:

Your previous response did NOT correctly complete the user's request.

ORIGINAL USER GOAL:
{goal}

ACTUAL TOOL OBSERVATIONS:
{chr(10).join(observations)}

Continue working until the original goal is actually completed.

Rules:
- Tool observations are the source of truth.
- Do not invent information.
- Do not use placeholders.
- Do not tell the user how to do something if you can do it yourself.
- If another tool is needed, use it.
- If the requested information is already present in the observations,
  answer directly using that information.
- Keep the final response concise and natural.
"""
            })

            continue

        # --------------------------------
        # MODEL TOOL ÇAĞIRDI
        # --------------------------------
        for call in response.message.tool_calls:

            tool_name = call.function.name
            arguments = call.function.arguments

            if tool_name not in NON_ACTION_TOOLS:
                used_action_tool = True

            result = ""

            if tool_name not in TOOL_MAP:
                result = f"TOOL_ERROR: Unknown tool: {tool_name}"

            if tool_name in {
                "save_memory",
                "save_self_memory",
                "forget_memory",
            }:
                memory_operation_used = True

            else:
                target = (
                    arguments.get("path")
                    or arguments.get("app_name")
                    or arguments.get("cwd")
                    or arguments.get("command")
                    or ""
                )

                print(f"\n[TOOL] {tool_name}: {target}")

                if check_permission(tool_name, target):

                    try:
                        result = TOOL_MAP[tool_name](**arguments)

                        if tool_name == "read_file" and not result.startswith("READ_ERROR"):
                            observations.append(
                                f"EXACT_FILE_CONTENT:\n{result}"
                            )
                        else:
                            observations.append(
                                f"{tool_name}: {result}"
                            )

                    except Exception as e:
                        result = f"TOOL_ERROR: {type(e).__name__}: {e}"
                else:
                    result = "Permission denied by user."

            # Şimdilik debug için gösteriyoruz
            print(f"[RESULT] {result[:500]}")

            # Tool sonucunu sadece çalışma geçmişine ekle
            working_messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": result,
            })

    return "Task stopped because maximum agent steps were reached."











