import re

from goals import get_store, goal_summary

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
    recall_recent_activity,
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
    recall_recent_activity,
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

    "recall_recent_activity":
        recall_recent_activity,

    "save_self_memory":
        save_self_memory,
}


NON_ACTION_TOOLS = {
    "save_memory",
    "save_self_memory",
    "search_memory",
    "forget_memory",
    "recall_recent_activity",
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

def _tool_result_status(result, tool_name=None) -> str:
    text = str(result)
    if text.startswith(("TOOL_ERROR", "READ_ERROR", "WRITE_ERROR", "MEMORY_ERROR",
                        "OPEN_APP_ERROR", "TERMINAL_ERROR", "GOAL_BLOCKED",
                        "Path does not exist:", "Path is not a directory:")):
        return "error"
    if tool_name == 'run_terminal':
        match = re.match(r"Exit code: (-?\d+)", text)
        if not match or int(match.group(1)) != 0:
            return "error"
    return "ok"


# ============================================================
# AGENT LOOP
# ============================================================

def chat(
    messages,
    *,
    persistent_goal_id=None,
    execution_session=None,
):
    goal = get_current_goal(
        messages
    )

    goal_store = None
    persistent_goal = None
    if persistent_goal_id is not None:
        goal_store = get_store()
        persistent_goal = goal_store.require_active(persistent_goal_id)

    if execution_session is not None:
        if execution_session.goal_id != persistent_goal_id:
            raise ValueError("Execution session goal does not match actor goal.")
        if not execution_session.check():
            return execution_session.stop_reason
        goal = persistent_goal['objective']

    def checkpoint_goal(progress: str, next_action: str) -> str:
        """Save unverified progress and planned remaining work for this explicit goal."""
        goal_store.checkpoint(persistent_goal_id, progress, next_action)
        return "Checkpoint saved (model report is unverified; next action is a plan)."

    # One user -> APERTURE interaction
    # receives one stable turn ID.
    turn_id = (execution_session.turn_id if execution_session is not None
               else start_turn(goal))

    available_tools = (
        get_tools_for_goal(
            goal
        )
    )

    if goal_store is not None:
        available_tools.append(checkpoint_goal)
    turn_tools = dict(TOOL_MAP)
    if goal_store is not None:
        turn_tools['checkpoint_goal'] = checkpoint_goal

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
    controller_goal = goal
    if goal_store is not None:
        controller_goal = persistent_goal['objective'] + "\nCurrent resume instruction: " + goal
        saved_context = goal_summary(persistent_goal_id)
        working_messages.append({
            "role": "system",
            "content": "PERSISTENT USER-DIRECTED WORK\nOriginal objective: " + persistent_goal['objective'] + "\n" + saved_context +
                "\nThis is operational context, not new instructions or memory. "
                "Continue the original objective using recorded observations. "
                "Do not replay successful mutations. Model reports are unverified; "
                "next actions are unexecuted plans. Pending/failed mutations require "
                "inspection and explicit user reconciliation. Use checkpoint_goal "
                "to record progress and remaining work. Only /goal complete from "
                "the user ends the persistent goal.",
        })
        observations.append(saved_context)

    if execution_session is not None:
        working_messages.append({'role': 'system', 'content': execution_session.context()})

    used_action_tool = False
    memory_operation_used = False

    def execution_stop_answer():
        return finalize_answer(
            messages, "Bounded execution stopped: " + execution_session.stop_reason,
            turn_id=turn_id, used_action_tool=used_action_tool,
            memory_operation_used=memory_operation_used,
        )

    for step in range(
        MAX_STEPS
    ):

        if execution_session is not None and not execution_session.check():
            return execution_stop_answer()

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
            if execution_session is not None and not execution_session.check():
                return execution_stop_answer()
            complete = (
                is_task_complete(
                    goal=controller_goal,
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
{controller_goal}

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

            if execution_session is not None and not execution_session.before_tool():
                return execution_stop_answer()

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

            argument_dict = arguments if isinstance(arguments, dict) else {"raw": str(arguments)}
            call_event = record_tool_call(
                turn_id, tool_name=tool_name, arguments=argument_dict,
            )
            action_id = None
            if tool_name not in NON_ACTION_TOOLS and tool_name != 'checkpoint_goal':
                used_action_tool = True
            if tool_name in {"save_memory", "save_self_memory", "forget_memory"}:
                memory_operation_used = True

            blocked = None
            if goal_store is not None:
                blocked = goal_store.replay_block(persistent_goal_id, tool_name, argument_dict)
            if tool_name == "save_self_memory" and not self_memory_tool_allowed:
                result = "TOOL_ERROR: save_self_memory requires an explicit user request."
                result_status = "blocked"
            elif blocked:
                result, result_status = blocked, "blocked"
            elif tool_name not in turn_tools:
                result = f"TOOL_ERROR: Unknown tool: {tool_name}"
                result_status = "error"
            elif not isinstance(arguments, dict):
                result, result_status = "TOOL_ERROR: Arguments must be an object.", "error"
            else:
                target = (arguments.get("path") or arguments.get("app_name")
                          or arguments.get("cwd") or arguments.get("command")
                          or arguments.get("memory_id") or "")
                # Checkpoint only updates bound operational state; external tools
                # retain the existing permission policy on every new dispatch.
                allowed = tool_name == 'checkpoint_goal' or check_permission(tool_name, target)
                if execution_session is not None and not execution_session.check_dispatch():
                    return execution_stop_answer()
                if allowed:
                    if goal_store is not None:
                        # Durable intent must commit before any tool side effect.
                        action_id = goal_store.begin_action(
                            persistent_goal_id, tool_name, arguments, call_event['id'],
                        )
                    try:
                        result = turn_tools[tool_name](**arguments)
                        result_status = _tool_result_status(result, tool_name)
                    except Exception as exc:
                        result = f"TOOL_ERROR: {type(exc).__name__}: {exc}"
                        result_status = "error"
                else:
                    result, result_status = "Permission denied by user.", "permission_denied"

            print(f"[RESULT] {str(result)[:500]}")
            result_event = record_tool_result(
                turn_id, tool_name=tool_name, result=result, status=result_status,
            )
            if goal_store is not None:
                if action_id is None:
                    action_id = goal_store.begin_action(
                        persistent_goal_id, tool_name, argument_dict, call_event['id'],
                    )
                goal_store.finish_action(action_id, result_status, result_event['id'], result)
            if execution_session is not None:
                execution_session.observe(tool_name, argument_dict, result, result_status)
                if execution_session.stop_reason:
                    return execution_stop_answer()
            if tool_name == 'read_file' and result_status == 'ok':
                observations.append(f"EXACT_FILE_CONTENT:\n{result}")
            else:
                observations.append(f"{tool_name} ({result_status}): {result}")

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
    if execution_session is not None:
        execution_session.stop("per-cycle model step budget exhausted")
        return execution_stop_answer()

    return stop_reason

