"""Ephemeral, explicitly authorized bounded execution over the existing actor."""
import hashlib

from experience import record_event, start_turn
from goals import MUTATING_TOOLS, action_key, get_store, goal_summary

MAX_CYCLES = 3
MAX_TOOL_ATTEMPTS = 12


def validate_cycles(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_CYCLES:
        raise ValueError("Usage: /goal run <cycles>; cycles must be 1–3.")
    return value


class ExecutionSession:
    def __init__(self, goal_id, cycles, turn_id):
        self.goal_id = goal_id
        self.cycles = validate_cycles(cycles)
        self.turn_id = turn_id
        self.cycle = 0
        self.tool_attempts = 0
        self.stop_reason = None
        self.observations = []
        self.seen = set()

    def stop(self, reason):
        if self.stop_reason is None:
            self.stop_reason = reason
        return self.stop_reason

    def check_dispatch(self):
        if self.stop_reason:
            return False
        store = get_store()
        goal = store.get(self.goal_id)
        if not goal or goal['status'] != 'active':
            self.stop('goal is inactive')
        elif any(a['tool_name'] in MUTATING_TOOLS and a['status'] in {'pending', 'error'}
                 and not a['resolution'] for a in store.actions(self.goal_id)):
            self.stop('mutation effects are uncertain; explicit reconciliation required')
        return self.stop_reason is None

    def check(self):
        if not self.check_dispatch():
            return False
        if self.tool_attempts >= MAX_TOOL_ATTEMPTS:
            self.stop('tool attempt budget exhausted')
        return self.stop_reason is None

    def before_tool(self):
        if not self.check():
            return False
        self.tool_attempts += 1
        return True

    def observe(self, tool_name, arguments, result, status):
        if status != 'ok':
            self.stop({'permission_denied': 'permission denied', 'blocked': 'action blocked; reconciliation may be required'}
                      .get(status, 'tool error; mutation effects may require reconciliation'))
            return
        if tool_name in {'checkpoint_goal', 'recall_recent_activity', 'search_memory'}:
            return
        digest = hashlib.sha256(str(result).encode('utf-8')).hexdigest()
        key = (action_key(tool_name, arguments), digest)
        if key not in self.seen:
            self.seen.add(key)
            self.observations.append(f"{tool_name} arguments_fingerprint={key[0]} result={str(result)[:1000]}")

    def context(self):
        return (f"BOUNDED EXECUTION: cycle {self.cycle}/{self.cycles}; "
                f"remaining tool attempts {MAX_TOOL_ATTEMPTS - self.tool_attempts}.\n"
                "Internal continuation under the original user authorization. "
                "Checkpoints and remaining work are unverified reports and unexecuted plans.\n"
                "Successful tool observations from this session:\n" + '\n'.join(self.observations[-12:]))


def run_goal(messages, *, persistent_goal_id, cycles):
    """Execute synchronously; restarting APERTURE never restores this authorization."""
    validate_cycles(cycles)
    get_store().require_active(persistent_goal_id)
    from llm import chat, get_current_goal
    session = ExecutionSession(persistent_goal_id, cycles, start_turn(get_current_goal(messages)))
    print('[GOAL] Starting bounded execution')
    try:
        for cycle in range(1, cycles + 1):
            if not session.check():
                break
            session.cycle = cycle
            print(f'[GOAL] Cycle {cycle}/{cycles}')
            record_event(event_type='goal_execution_cycle', event_class='system', actor='system',
                         turn_id=session.turn_id, content=f'Cycle {cycle}/{cycles}',
                         metadata={'goal_id': persistent_goal_id})
            previous = len(session.observations)
            answer = chat(messages, persistent_goal_id=persistent_goal_id, execution_session=session)
            print(answer)
            print(f'[GOAL] New observations: {len(session.observations) - previous}')
            if not session.check():
                break
            if len(session.observations) == previous:
                session.stop('no new meaningful tool observations')
                break
        session.stop('cycle budget exhausted')
    except KeyboardInterrupt:
        session.stop('user interrupted execution; goal remains recoverable')
    except Exception as exc:
        session.stop(f'execution error ({type(exc).__name__}); inspect recorded actions before continuing')
    record_event(event_type='goal_execution_stopped', event_class='system', actor='system',
                 turn_id=session.turn_id, content=session.stop_reason,
                 metadata={'goal_id': persistent_goal_id, 'cycles': session.cycle,
                           'tool_attempts': session.tool_attempts})
    return f"[GOAL] Stopped: {session.stop_reason}\n" + goal_summary(persistent_goal_id)
