"""Regression tests for issues found in the security/loop review."""
from __future__ import annotations

import asyncio

from typer.testing import CliRunner

from src import cli, coordinator
from src.state import Phase, RoundState


def test_accumulate_usage_ignores_non_object_json(tmp_path):
    """A corrupt/crafted usage sidecar (valid JSON, not an object) must not crash."""
    st = coordinator.create_debate("q")
    st.active_models = ["gpt4o"]
    st.rounds = [RoundState(round_num=1, phase=Phase.PROPOSING)]
    st.current_round = 1
    rd = tmp_path / "round-001"
    rd.mkdir()
    (rd / "gpt4o.proposal.usage.json").write_text("[1, 2, 3]")
    coordinator.accumulate_usage(tmp_path, st)  # must not raise
    assert "gpt4o" not in st.usage


def test_debate_id_rejects_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DEBATES_DIR", tmp_path)
    r = CliRunner()
    for bad in ["../../etc", "/etc", "a/b", "..", "foo/../bar", "."]:
        for cmd in ("show", "status", "resume"):
            res = r.invoke(cli.app, [cmd, bad])
            assert res.exit_code == 2, f"{cmd} {bad!r} not rejected (exit {res.exit_code})"


def test_valid_debate_id_not_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DEBATES_DIR", tmp_path)
    res = CliRunner().invoke(cli.app, ["show", "20260525-181144-ecc843"])
    assert res.exit_code == 1  # not found, but NOT rejected as invalid (which is 2)


def test_coordinator_deadlocks_when_state_never_appears(tmp_path):
    status = asyncio.run(
        asyncio.wait_for(
            coordinator.coordinator_loop(tmp_path, poll_interval=0.01, stall_timeout=0.05),
            timeout=5,
        )
    )
    assert status == "deadlocked"
