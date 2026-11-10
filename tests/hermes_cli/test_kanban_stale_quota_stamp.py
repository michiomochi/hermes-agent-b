"""A stale quota stamp must not outlive a review handoff.

The dispatcher's rate-limit reaper requeues a quota-walled card and stamps a quota-flavoured
``tasks.last_failure_error`` WITHOUT counting a failure.  While the latest ended run is that
``rate_limited`` run, ``check_respawn_guard`` returns before ``blocker_auth``.  A handoff
(``request_review`` / ``request_changes``) ends the latest run as ``review_requested`` /
``changes_requested``, so the early return stops applying; a stamp that survives the handoff then
matches ``_RESPAWN_BLOCKER_RE`` and parks the card on ``blocker_auth`` on every tick until an
operator clears the column by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

# What the reaper writes for a rate-limited worker (kanban_db_dispatch.py, dead.rate_limited).
QUOTA_STAMP = "pid 1 exited rate-limited (quota wall) — requeued without counting a failure"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # No rate-limit cooldown: the guard outcome then depends only on the stamp and the latest run.
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def clock(monkeypatch):
    """A controllable wall clock, so run ordering (``ended_at DESC``) never depends on real time."""

    class Clock:
        now = 5_000_000

        def __call__(self) -> float:
            return self.now

        def tick(self, seconds: int = 10) -> None:
            self.now += seconds

    c = Clock()
    monkeypatch.setattr(kb.time, "time", c)
    return c


def _rate_limited_requeue(conn, tid, lane, clock) -> None:
    """One claimed run that ends ``rate_limited`` and is requeued with the reaper's stamp."""
    claim = kb.claim_review_task if lane == "review" else kb.claim_task
    assert claim(conn, tid) is not None
    run_id = kb.get_task(conn, tid).current_run_id
    clock.tick()
    conn.execute(
        "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', ended_at=? WHERE id=?",
        (clock.now, run_id),
    )
    conn.execute(
        "UPDATE tasks SET status=?, current_run_id=NULL, claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, last_failure_error=? WHERE id=?",
        (lane, QUOTA_STAMP, tid),
    )
    conn.commit()
    clock.tick()


def _stamp_and_failures(conn, tid):
    row = conn.execute(
        "SELECT last_failure_error, consecutive_failures FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    return row["last_failure_error"], row["consecutive_failures"]


def test_request_review_after_rate_limited_runs_is_dispatchable(kanban_home, clock):
    """The production shape: ~19 rate-limited requeues, then a successful ``request_review``."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rate-limited then review", assignee="executor")
        for _ in range(19):
            _rate_limited_requeue(conn, tid, "ready", clock)
        conn.execute("UPDATE tasks SET consecutive_failures = 2 WHERE id = ?", (tid,))
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None  # precondition

        assert kb.claim_task(conn, tid) is not None
        run_id = kb.get_task(conn, tid).current_run_id
        clock.tick()
        assert kb.request_review(conn, tid, reviewer="reviewer", expected_run_id=run_id)

        stamp, failures = _stamp_and_failures(conn, tid)
        assert stamp is None
        assert failures == 2  # only complete_task resets the breaker counter
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None


def test_changes_requested_after_reviewer_rate_limit_is_dispatchable(kanban_home, clock):
    """The reviewer is rate-limited too: its requeue re-stamps the card, and the stamp must not
    survive the reviewer's verdict back to the implementer."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reviewer rate-limited", assignee="executor")
        assert kb.claim_task(conn, tid) is not None
        run_id = kb.get_task(conn, tid).current_run_id
        clock.tick()
        assert kb.request_review(conn, tid, reviewer="reviewer", expected_run_id=run_id)
        clock.tick()
        for _ in range(3):
            _rate_limited_requeue(conn, tid, "review", clock)
        assert kb.claim_review_task(conn, tid) is not None
        review_run = kb.get_task(conn, tid).current_run_id
        clock.tick()
        ok, implementer = kb.request_changes(
            conn, tid, reason="needs another pass", expected_run_id=review_run,
        )
        assert ok and implementer == "executor"

        assert kb.get_task(conn, tid).status == "ready"
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


def test_genuine_auth_failure_still_blocks_after_a_handoff(kanban_home, clock):
    """Negative control: a handoff must not whitewash a LATER real failure.  The reviewer run that
    follows the handoff dies on a provider auth error; that newest run decides."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="auth failure after handoff", assignee="executor")
        assert kb.claim_task(conn, tid) is not None
        run_id = kb.get_task(conn, tid).current_run_id
        clock.tick()
        assert kb.request_review(conn, tid, reviewer="reviewer", expected_run_id=run_id)
        clock.tick()

        assert kb.claim_review_task(conn, tid) is not None
        review_run = kb.get_task(conn, tid).current_run_id
        clock.tick()
        conn.execute(
            "UPDATE task_runs SET outcome='spawn_failed', status='failed', ended_at=? WHERE id=?",
            (clock.now, review_run),
        )
        conn.execute(
            "UPDATE tasks SET status='review', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL, last_failure_error=? WHERE id=?",
            ("provider authentication failed", tid),
        )
        conn.commit()

        assert kbd.check_respawn_guard(conn, tid, lane="review") == "blocker_auth"
