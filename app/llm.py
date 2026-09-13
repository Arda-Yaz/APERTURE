from ollama import chat as ollama_chat

from persona import build_persona_context

from config import (
    ENABLE_DYNAMIC_SELF,
    ENABLE_RELATIONSHIP,
    ENABLE_SELF_CONSOLIDATION,
)

from reflection import (
    maybe_reflect,
    maybe_consolidate_self_state,
)

from self_state import (
    build_self_state_context,
    maybe_update_self_state,
    get_self_state,
)


from relationship import (
    build_relationship_context,
    get_relationship_state,
    maybe_update_relationship,
)


from experience import (
    start_turn,
    record_assistant_message,
    record_tool_call,
    record_tool_result,
    record_self_state_update,
    record_relationship_state_update,
    record_turn_stopped,
)

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
    build_actor_memory_context,
)


MODEL = "qwen3:8b"
MAX_STEPS = 8


ACTOR_OPTIONS = {
    # Qwen3 recommended settings for thinking mode
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,

    # Mild repetition protection
    "presence_penalty": 1.0,

    # Prevent pathological unlimited generations
    "num_predict": 4096,
}

# ============================================================
# TOOL AVAILABILITY
# ============================================================

BASE_TOOLS = [
    list_directory,
    read_file,
    write_file,
    open_app,
    run_terminal,
    save_memory,
    search_memory,
    forget_memory,
]


def should_expose_self_memory_tool(
    user_message: str,
) -> bool:
    """
    Expose save_self_memory only when Arda explicitly asks
    APERTURE to remember/store something about itself.

    Normal self-expression must go through Reflection instead.
    """

    text = " ".join(
        str(user_message)
        .casefold()
        .split()
    )

    if not text:
        return False

    recall_only = (
        "do you remember ",
        "did you remember ",
        "hatırlıyor musun",
        "hatırladın mı",
    )

    if any(
        text.startswith(marker)
        for marker in recall_only
    ):
        return False

    explicit_patterns = (
        # English
        "remember this about yourself",
        "remember that about yourself",
        "remember that you ",
        "remember you ",
        "remember your ",
        "please remember that you ",
        "please remember your ",
        "i want you to remember that you ",
        "i want you to remember your ",
        "save this about yourself",
        "save that about yourself",
        "save that you ",
        "save your ",
        "store this about yourself",
        "store that about yourself",
        "store that you ",
        "store your ",
        "don't forget that you ",
        "don't forget you ",
        "don't forget your ",
        "do not forget that you ",
        "do not forget you ",
        "do not forget your ",

        # Turkish
        "kendin hakkında bunu hatırla",
        "kendin hakkında şunu hatırla",
        "kendinle ilgili bunu hatırla",
        "kendinle ilgili şunu hatırla",
        "kendin hakkında bunu kaydet",
        "kendin hakkında şunu kaydet",
        "kendinle ilgili bunu kaydet",
        "kendinle ilgili şunu kaydet",
    )

    if any(
        pattern in text
        for pattern in explicit_patterns
    ):
        return True

    if (
        "about yourself" in text
        and any(
            marker in text
            for marker in (
                "remember",
                "save",
                "store",
                "don't forget",
                "do not forget",
            )
        )
    ):
        return True

    if (
        any(
            marker in text
            for marker in (
                "kendin",
                "kendinle",
                "kendin hakkında",
            )
        )
        and any(
            marker in text
            for marker in (
                "hatırla",
                "kaydet",
                "unutma",
            )
        )
    ):
        return True

    return False


def get_tools_for_goal(
    goal: str,
):
    tools = list(
        BASE_TOOLS
    )

    if should_expose_self_memory_tool(
        goal
    ):
        tools.append(
            save_self_memory
        )

    return tools


TOOL_MAP = {
    "list_directory":
        list_directory,

    "read_file":
        read_file,

    "write_file":
        write_file,

    "open_app":
        open_app,

    "run_terminal":
        run_terminal,

    "save_memory":
        save_memory,

    "search_memory":
        search_memory,

    "forget_memory":
        forget_memory,

    "save_self_memory":
        save_self_memory,
}


NON_ACTION_TOOLS = {
    "save_memory",
    "save_self_memory",
    "search_memory",
    "forget_memory",
}


# ============================================================
# GOAL / CONTEXT
# ============================================================

def get_current_goal(
    messages,
):
    for message in reversed(
        messages
    ):
        if (
            isinstance(
                message,
                dict,
            )
            and message.get("role")
            == "user"
        ):
            return message.get(
                "content",
                "",
            )

    return ""


def inject_runtime_context(
    messages,
    user_message: str,
):
    runtime_messages = list(
        messages
    )

    persona_context = (
        build_persona_context(
            user_message=user_message,
        )
    )

    actor_memory_context = (
        build_actor_memory_context(
            query=user_message,
        )
    )

    context_parts = [
        persona_context,
    ]

    if ENABLE_DYNAMIC_SELF:

        context_parts.append(
            build_self_state_context()
        )

    if ENABLE_RELATIONSHIP:

        context_parts.append(
            build_relationship_context()
        )

    context_parts.append(
        actor_memory_context
    )

    runtime_context = (
        "\n\n".join(
            context_parts
        )
    )

    if (
        runtime_messages
        and isinstance(
            runtime_messages[0],
            dict,
        )
        and runtime_messages[0].get(
            "role"
        )
        == "system"
    ):
        system_message = dict(
            runtime_messages[0]
        )

        system_message["content"] = (
            system_message.get(
                "content",
                "",
            )
            + "\n\n"
            + runtime_context
        )

        runtime_messages[0] = (
            system_message
        )

    else:
        runtime_messages.insert(
            0,
            {
                "role": "system",
                "content":
                    runtime_context,
            },
        )

    return runtime_messages


# ============================================================
# FINAL ANSWER / COGNITIVE POST-PROCESSING
# ============================================================

def finalize_answer(
    messages,
    answer: str,
    *,
    turn_id: str,
    used_action_tool: bool,
    memory_operation_used: bool,
):
    messages.append({
        "role": "assistant",
        "content": answer,
    })

    # --------------------------------------------------------
    # OBSERVED EXPERIENCE
    # --------------------------------------------------------

    record_assistant_message(
        turn_id,
        answer,
    )

    # --------------------------------------------------------
    # REFLECTION
    # --------------------------------------------------------

    reflection_result = (
        maybe_reflect(
            messages,
            used_action_tool=(
                used_action_tool
            ),
            memory_operation_used=(
                memory_operation_used
            ),
            turn_id=turn_id,
        )
    )

    if reflection_result:
        print(
            f"\n[REFLECTION] "
            f"{reflection_result}"
        )
    # --------------------------------------------------------
    # DYNAMIC SELF — EXPERIMENTAL
    # --------------------------------------------------------

    if ENABLE_DYNAMIC_SELF:

        previous_self_state = (
            get_self_state()
        )

        self_state_result = (
            maybe_update_self_state(
                messages,
                used_action_tool=(
                    used_action_tool
                ),
                memory_operation_used=(
                    memory_operation_used
                ),
            )
        )

        if self_state_result is not None:

            print(
                f"\n[SELF_STATE] "
                f"{self_state_result}"
            )

            record_self_state_update(
                turn_id,
                previous_state=(
                    previous_self_state
                ),
                updated_state=(
                    self_state_result
                ),
            )

            if (
                ENABLE_SELF_CONSOLIDATION
            ):

                self_consolidation_result = (
                    maybe_consolidate_self_state(
                        turn_id=turn_id,
                        state_update=(
                            self_state_result
                        ),
                        assistant_evidence=(
                            answer
                        ),
                    )
                )

                if self_consolidation_result:

                    print(
                        "\n[SELF_CONSOLIDATION] "
                        f"{self_consolidation_result}"
                    )


    # --------------------------------------------------------
    # RELATIONSHIP MODEL — EXPERIMENTAL
    # --------------------------------------------------------

    # --------------------------------------------------------
    # RELATIONSHIP MODEL — EXPERIMENTAL
    # --------------------------------------------------------

    if ENABLE_RELATIONSHIP:

        previous_relationship_state = (
            get_relationship_state()
        )

        relationship_result = (
            maybe_update_relationship(
                messages,
                used_action_tool=(
                    used_action_tool
                ),
                memory_operation_used=(
                    memory_operation_used
                ),
            )
        )

        if relationship_result is not None:

            print(
                "\n[RELATIONSHIP] "
                f"{relationship_result}"
            )

            record_relationship_state_update(
                turn_id,
                previous_state=(
                    previous_relationship_state
                ),
                updated_state=(
                    relationship_result
                ),
            )

    return answer

# ============================================================
# RESULT STATUS
# ============================================================

def _tool_result_status(
    result,
) -> str:

    text = str(
        result
    )

    error_prefixes = (
        "TOOL_ERROR",
        "READ_ERROR",
        "WRITE_ERROR",
        "MEMORY_ERROR",
    )

    if text.startswith(
        error_prefixes
    ):
        return "error"

    return "ok"


# ============================================================
# AGENT LOOP
# ============================================================

def chat(
    messages,
):
    goal = get_current_goal(
        messages
    )

    # One user -> APERTURE interaction
    # receives one stable turn ID.
    turn_id = (
        start_turn(
            goal
        )
    )

    available_tools = (
        get_tools_for_goal(
            goal
        )
    )

    self_memory_tool_allowed = (
        should_expose_self_memory_tool(
            goal
        )
    )

    # Agent'ın iç çalışma geçmişi.
    # Tool sonuçları ve controller mesajları
    # kalıcı sohbeti kirletmez.
    working_messages = (
        inject_runtime_context(
            messages,
            user_message=goal,
        )
    )

    observations = []

    used_action_tool = False
    memory_operation_used = False

    for step in range(
        MAX_STEPS
    ):

        response = ollama_chat(
            model=MODEL,
            messages=(
                working_messages
            ),
            tools=available_tools,
            think=True,
            options=ACTOR_OPTIONS,
        )

        working_messages.append(
            response.message
        )

        # ====================================================
        # MODEL TOOL ÇAĞIRMADI
        # ====================================================

        if (
            not response
            .message
            .tool_calls
        ):

            answer = (
                response.message.content
                or ""
            )

            # Normal sohbet
            if not used_action_tool:

                return finalize_answer(
                    messages,
                    answer,
                    turn_id=turn_id,
                    used_action_tool=(
                        used_action_tool
                    ),
                    memory_operation_used=(
                        memory_operation_used
                    ),
                )

            # Tool kullanıldıysa görev
            # gerçekten tamamlandı mı?
            complete = (
                is_task_complete(
                    goal=goal,
                    answer=answer,
                    observations=(
                        "\n\n".join(
                            observations
                        )
                    ),
                )
            )

            if complete:

                return finalize_answer(
                    messages,
                    answer,
                    turn_id=turn_id,
                    used_action_tool=(
                        used_action_tool
                    ),
                    memory_operation_used=(
                        memory_operation_used
                    ),
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

        # ====================================================
        # MODEL TOOL ÇAĞIRDI
        # ====================================================

        for call in (
            response
            .message
            .tool_calls
        ):

            tool_name = (
                call
                .function
                .name
            )

            arguments = (
                call
                .function
                .arguments
            )

            # -----------------------------------------------
            # EXPERIENCE: TOOL CALL
            # -----------------------------------------------

            record_tool_call(
                turn_id,
                tool_name=tool_name,
                arguments=(
                    arguments
                    if isinstance(
                        arguments,
                        dict,
                    )
                    else {
                        "raw":
                            str(arguments)
                    }
                ),
            )

            # -----------------------------------------------
            # SELF MEMORY DEFENSE
            # -----------------------------------------------

            if (
                tool_name
                == "save_self_memory"
                and not
                self_memory_tool_allowed
            ):

                result = (
                    "TOOL_ERROR: "
                    "save_self_memory is only "
                    "available when Arda "
                    "explicitly asks APERTURE "
                    "to remember something "
                    "about itself."
                )

                print(
                    f"\n[TOOL BLOCKED] "
                    f"{tool_name}"
                )

                print(
                    f"[RESULT] "
                    f"{result}"
                )

                record_tool_result(
                    turn_id,
                    tool_name=tool_name,
                    result=result,
                    status="blocked",
                )

                working_messages.append({
                    "role": "tool",
                    "tool_name":
                        tool_name,
                    "content":
                        result,
                })

                continue

            # -----------------------------------------------
            # TRACK ACTION / MEMORY OPERATIONS
            # -----------------------------------------------

            if (
                tool_name
                not in
                NON_ACTION_TOOLS
            ):
                used_action_tool = True

            if tool_name in {
                "save_memory",
                "save_self_memory",
                "forget_memory",
            }:
                memory_operation_used = True

            # -----------------------------------------------
            # UNKNOWN TOOL
            # -----------------------------------------------

            if (
                tool_name
                not in TOOL_MAP
            ):
                result = (
                    "TOOL_ERROR: "
                    "Unknown tool: "
                    f"{tool_name}"
                )

                result_status = (
                    "error"
                )

            else:

                target = (
                    arguments.get("path")
                    or arguments.get(
                        "app_name"
                    )
                    or arguments.get(
                        "cwd"
                    )
                    or arguments.get(
                        "command"
                    )
                    or arguments.get(
                        "memory_id"
                    )
                    or ""
                )

                print(
                    f"\n[TOOL] "
                    f"{tool_name}: "
                    f"{target}"
                )

                # -------------------------------------------
                # PERMISSION
                # -------------------------------------------

                if check_permission(
                    tool_name,
                    target,
                ):

                    try:
                        result = (
                            TOOL_MAP[
                                tool_name
                            ](
                                **arguments
                            )
                        )

                        result_status = (
                            _tool_result_status(
                                result
                            )
                        )

                        if (
                            tool_name
                            == "read_file"
                            and not
                            str(result)
                            .startswith(
                                "READ_ERROR"
                            )
                        ):
                            observations.append(
                                "EXACT_FILE_CONTENT:"
                                f"\n{result}"
                            )

                        else:
                            observations.append(
                                f"{tool_name}: "
                                f"{result}"
                            )

                    except Exception as e:

                        result = (
                            "TOOL_ERROR: "
                            f"{type(e).__name__}: "
                            f"{e}"
                        )

                        result_status = (
                            "error"
                        )

                else:

                    result = (
                        "Permission denied "
                        "by user."
                    )

                    result_status = (
                        "permission_denied"
                    )

            # -----------------------------------------------
            # DEBUG OUTPUT
            # -----------------------------------------------

            print(
                f"[RESULT] "
                f"{str(result)[:500]}"
            )

            # -----------------------------------------------
            # EXPERIENCE: TOOL RESULT
            # -----------------------------------------------

            record_tool_result(
                turn_id,
                tool_name=tool_name,
                result=result,
                status=result_status,
            )

            # Tool sonucunu sadece
            # çalışma geçmişine ekle.
            working_messages.append({
                "role": "tool",
                "tool_name":
                    tool_name,
                "content":
                    result,
            })

    # ========================================================
    # MAX STEP STOP
    # ========================================================

    stop_reason = (
        "Task stopped because maximum "
        "agent steps were reached."
    )

    record_turn_stopped(
        turn_id,
        reason=stop_reason,
    )

    return stop_reason

