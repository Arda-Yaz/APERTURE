"""User-directed operational work. Imports never initialize storage."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "aperture_goals.db"
_STORE = None
MUTATING_TOOLS = {"write_file", "run_terminal", "open_app", "save_memory",
                  "save_self_memory", "forget_memory"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def action_key(tool_name, arguments):
    encoded = json.dumps([tool_name, arguments], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _operation_detail(arguments):
    """Keep the inspection target visible even when payloads are large."""
    if not isinstance(arguments, dict):
        return str(arguments)[:400]
    parts = []
    priority = {'path':400, 'command':500, 'app_name':80, 'memory_id':80, 'cwd':200}
    for key, budget in priority.items():
        if key in arguments:
            value = (arguments[key] if isinstance(arguments[key], str)
                     else json.dumps(arguments[key], ensure_ascii=False))
            parts.append(f"{key}={value[:budget]}" + (' [truncated]' if len(value) > budget else ''))
    for key, value in list((k, v) for k, v in arguments.items() if k not in priority)[:3]:
        encoded = json.dumps(value, ensure_ascii=False)
        parts.append(f"{str(key)[:80]}={encoded[:80]}" +
                     (f" [payload truncated; {len(encoded)} serialized characters]" if len(encoded) > 80 else ''))
    return ', '.join(parts)


class GoalStore:
    def __init__(self, db_path=DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY, objective TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active','paused','completed','cancelled')),
                    progress TEXT NOT NULL DEFAULT '', next_action TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS one_unfinished_goal ON goals ((1))
                    WHERE status IN ('active','paused');
                CREATE TABLE IF NOT EXISTS actions (
                    id TEXT PRIMARY KEY, goal_id TEXT NOT NULL REFERENCES goals(id),
                    tool_name TEXT NOT NULL, fingerprint TEXT NOT NULL, status TEXT NOT NULL,
                    call_event_id TEXT, result_event_id TEXT, result TEXT NOT NULL DEFAULT '',
                    resolution TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def unfinished(self):
        with self._connect() as db:
            row = db.execute("SELECT * FROM goals WHERE status IN ('active','paused')").fetchone()
        return dict(row) if row else None

    def get(self, goal_id):
        with self._connect() as db:
            row = db.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
        return dict(row) if row else None

    def require_active(self, goal_id):
        goal = self.get(goal_id)
        if not goal or goal['status'] != 'active':
            raise ValueError("Goal must be active; completed/cancelled goals cannot resume.")
        return goal

    def create(self, objective):
        objective = objective.strip()
        if not objective or len(objective) > 8000:
            raise ValueError("Provide an objective of 1–8000 characters.")
        goal_id, now = uuid.uuid4().hex, _now()
        try:
            with self._connect() as db:
                db.execute("INSERT INTO goals(id,objective,status,created_at,updated_at) VALUES (?,?,'active',?,?)",
                           (goal_id, objective, now, now))
        except sqlite3.IntegrityError as exc:
            raise ValueError("An unfinished goal already exists; complete or cancel it first.") from exc
        return self.get(goal_id)

    def transition(self, goal_id, status):
        if status not in {'active', 'paused', 'completed', 'cancelled'}:
            raise ValueError("Invalid goal status.")
        with self._connect() as db:
            cursor = db.execute("UPDATE goals SET status=?,updated_at=? WHERE id=? AND status IN ('active','paused')",
                                (status, _now(), goal_id))
            if cursor.rowcount != 1:
                raise ValueError("No unfinished goal; terminal goals cannot resume.")
        return self.get(goal_id)

    def checkpoint(self, goal_id, progress, next_action):
        self.require_active(goal_id)
        if not isinstance(progress, str) or not isinstance(next_action, str):
            raise ValueError("Checkpoint fields must be strings.")
        with self._connect() as db:
            db.execute("UPDATE goals SET progress=?,next_action=?,updated_at=? WHERE id=?",
                       (progress[:4000], next_action[:4000], _now(), goal_id))

    def actions(self, goal_id):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM actions WHERE goal_id=? ORDER BY rowid", (goal_id,)).fetchall()
        return [dict(row) for row in rows]

    def replay_block(self, goal_id, tool_name, arguments):
        self.require_active(goal_id)
        if tool_name not in MUTATING_TOOLS:
            return None
        actions = self.actions(goal_id)
        uncertain = [a for a in actions if a['tool_name'] in MUTATING_TOOLS
                     and a['status'] in {'pending', 'error'} and not a['resolution']]
        if uncertain:
            return ("GOAL_BLOCKED: Interrupted or failed mutation has uncertain effects. "
                    "Inspect its effects, then use /goal resolve " + uncertain[0]['id'] + " succeeded|retry.")
        fingerprint = action_key(tool_name, arguments)
        if any(a['fingerprint'] == fingerprint and (a['status'] == 'ok' or a['resolution'] == 'succeeded') for a in actions):
            return "GOAL_BLOCKED: This mutation already succeeded or was user-confirmed; do not replay it."
        return None

    def begin_action(self, goal_id, tool_name, arguments, call_event_id):
        self.require_active(goal_id)
        action_id, now = uuid.uuid4().hex, _now()
        with self._connect() as db:
            db.execute("INSERT INTO actions(id,goal_id,tool_name,fingerprint,status,call_event_id,created_at,updated_at) VALUES (?,?,?,?,'pending',?,?,?)",
                       (action_id, goal_id, tool_name, action_key(tool_name, arguments), call_event_id, now, now))
        return action_id

    def finish_action(self, action_id, status, result_event_id, result):
        with self._connect() as db:
            db.execute("UPDATE actions SET status=?,result_event_id=?,result=?,updated_at=? WHERE id=?",
                       (status, result_event_id, str(result)[:4000], _now(), action_id))
            db.execute("UPDATE goals SET updated_at=? WHERE id=(SELECT goal_id FROM actions WHERE id=?)",
                       (_now(), action_id))

    def resolve(self, goal_id, action_id, resolution):
        if resolution not in {'succeeded', 'retry'}:
            raise ValueError("Resolution must be succeeded or retry after inspecting effects.")
        with self._connect() as db:
            cursor = db.execute("UPDATE actions SET resolution=?,updated_at=? WHERE id=? AND goal_id=? AND status IN ('pending','error') AND resolution IS NULL",
                                (resolution, _now(), action_id, goal_id))
            if cursor.rowcount != 1:
                raise ValueError("No unresolved action with that identifier in this goal.")

    def summary_actions(self, goal_id):
        actions = self.actions(goal_id)
        unresolved = [a for a in actions if a['tool_name'] in MUTATING_TOOLS
                      and a['status'] in {'pending', 'error'} and not a['resolution']]
        succeeded = [a for a in actions if a['tool_name'] in MUTATING_TOOLS
                     and (a['status'] == 'ok' or a['resolution'] == 'succeeded')]
        return list({a['id']: a for a in unresolved[:8] + succeeded[-8:] + actions[-8:]}.values())

    def summary(self, goal_id, evidence_events=()):
        goal = self.get(goal_id)
        if not goal:
            return "No goal found."
        lines = [f"Goal {goal['id']} ({goal['status']}): {goal['objective'][:800]}",
                 f"Unverified model progress: {goal['progress'][:600]}",
                 f"Planned next action: {goal['next_action'][:600]}"]
        actions = self.actions(goal_id)
        lines.append(f"Recorded tool attempts: {len(actions)} (latest 8 plus up to 8 unresolved and 8 successful mutations; replay protection covers all).")
        evidence = {event['id']: event for event in evidence_events}
        for action in self.summary_actions(goal_id):
            call = evidence.get(action['call_event_id'], {})
            arguments = call.get('metadata', {}).get('arguments')
            detail = (_operation_detail(arguments)
                      if arguments is not None else 'Operation detail unavailable; inspect referenced Experience event.')
            lines.append(f"{action['id']} {action['tool_name']} {action['status']} "
                         f"call={action['call_event_id']} result={action['result_event_id']} "
                         f"user_resolution={action['resolution']} arguments={detail}: {action['result'][:250]}")
        return '\n'.join(lines)


def get_store():
    global _STORE
    if _STORE is None:
        _STORE = GoalStore()
    return _STORE


def goal_summary(goal_id):
    """Hydrate operation detail from observed, sanitized Experience by ID.

    Reading old references works across episodes; the operational store itself
    never initializes or owns Experience. Missing evidence remains explicit.
    """
    from experience import get_events_by_ids
    store = get_store()
    event_ids = [action['call_event_id'] for action in store.summary_actions(goal_id)
                 if action['call_event_id']]
    events = get_events_by_ids(event_ids) if event_ids else []
    return store.summary(goal_id, evidence_events=events)


def handle_goal_command(text):
    """Return (display text, explicit execution goal id or None)."""
    parts = text.strip().split(maxsplit=2)
    command = parts[1] if len(parts) > 1 else 'status'
    argument = parts[2] if len(parts) > 2 else ''
    store = get_store()
    if command == 'start':
        goal = store.create(argument)
        return goal_summary(goal['id']) + '\nUse /goal resume to begin work.', None
    goal = store.unfinished()
    if command == 'status':
        return goal_summary(goal['id']) if goal else 'No unfinished goal.', None
    if not goal:
        raise ValueError("No unfinished goal.")
    if command == 'run':
        from goal_execution import validate_cycles
        if not argument.isascii() or not argument.isdecimal():
            raise ValueError("Usage: /goal run <cycles>; cycles must be 1–3.")
        validate_cycles(int(argument))
        store.require_active(goal['id'])
        return goal_summary(goal['id']), goal['id']
    if command == 'resolve':
        values = argument.split()
        if len(values) != 2:
            raise ValueError("Usage: /goal resolve <action-id> succeeded|retry")
        store.resolve(goal['id'], *values)
    elif command in {'pause', 'resume', 'complete', 'cancel'}:
        store.transition(goal['id'], {'pause':'paused', 'resume':'active', 'complete':'completed', 'cancel':'cancelled'}[command])
        if command == 'resume':
            return goal_summary(goal['id']), goal['id']
    else:
        raise ValueError("Commands: /goal start <objective>, status, pause, resume [instruction], run <cycles>, complete, cancel, resolve <action-id> succeeded|retry")
    return goal_summary(goal['id']), None
