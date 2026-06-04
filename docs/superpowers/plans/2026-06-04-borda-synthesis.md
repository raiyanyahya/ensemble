# Borda Ranking + Synthesis-as-Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add (1) an additive Borda ranking signal piggybacked on the VOTING phase and (2) a synthesis-as-candidate step (endorsed author drafts a merged answer, participants confirm by vote) that runs only on consensus and always falls back to today's verbatim output.

**Architecture:** Both features are filesystem phases consistent with ensemble's existing propose→review→rebut→vote model. SYNTHESIS and CONFIRM are two new phases entered *only* after a FINALIZE consensus; any failure/stall in them finalizes verbatim instead of undoing the reached consensus. Borda never alters the finalize path — it is a recorded signal plus a tiebreaker for the already-arbitrary deadlock-plurality-tie case.

**Tech Stack:** Python 3.10+, pydantic models, pytest, asyncio file-polling coordinator/agents.

---

## File structure

- `src/state.py` — `Phase.SYNTHESIS`/`Phase.CONFIRM`; `ModelOutput` gains `synthesis`, `confirm`, `ranking`; `PHASE_FILE_SUFFIX` entries; `RoundState` gains `rankings`, `borda_scores`, `synthesis_used`, `synthesis_author`, `confirm_tally`.
- `src/coordinator.py` — ranking/synthesis/confirm parsing; Borda tally + tiebreak; per-phase participant set; finalize→synthesis→confirm orchestration with verbatim fallback; `final.md` rendering.
- `src/agent.py` — participation gating (winner-only synthesis); SYNTHESIS/CONFIRM prompt text.
- `tests/` — `test_state.py`, `test_parsing.py`, `test_consensus.py`, `test_flow.py`.

Each task is TDD: failing test → run (fail) → implement → run (pass) → commit.

---

### Task 1: State scaffolding for new phases and fields

**Files:**
- Modify: `src/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write failing test**

```python
def test_new_phases_and_output_fields():
    from src.state import Phase, PHASE_FILE_SUFFIX, ModelOutput
    assert Phase.SYNTHESIS == "synthesis"
    assert Phase.CONFIRM == "confirm"
    assert PHASE_FILE_SUFFIX[Phase.SYNTHESIS] == "synthesis"
    assert PHASE_FILE_SUFFIX[Phase.CONFIRM] == "confirm"

    o = ModelOutput(synthesis="merged", confirm="APPROVE", ranking=["B", "A"])
    assert o.has_phase(Phase.SYNTHESIS)
    assert o.has_phase(Phase.CONFIRM)
    # ranking rides along the vote phase, doesn't gate it
    assert ModelOutput(vote=None, ranking=["A"]).has_phase(Phase.VOTING) is False

    merged = ModelOutput()
    merged.merge_from(o)
    assert merged.synthesis == "merged" and merged.confirm == "APPROVE"
    assert merged.ranking == ["B", "A"]
    # for_phase isolates the field a phase owns
    assert ModelOutput(synthesis="x", confirm="y").for_phase(Phase.SYNTHESIS).confirm == ""
```

- [ ] **Step 2: Run — expect FAIL** `pytest tests/test_state.py::test_new_phases_and_output_fields -v`

- [ ] **Step 3: Implement.** In `Phase`: add `SYNTHESIS = "synthesis"` and `CONFIRM = "confirm"` (before `DONE`). In `PHASE_FILE_SUFFIX` add `Phase.SYNTHESIS: "synthesis", Phase.CONFIRM: "confirm"`. In `ModelOutput` add fields `synthesis: str = ""`, `confirm: str = ""`, `ranking: list[str] = Field(default_factory=list)`. Extend `has_phase` (`SYNTHESIS → bool(self.synthesis)`, `CONFIRM → bool(self.confirm)`), `for_phase` (`SYNTHESIS → ModelOutput(synthesis=self.synthesis)`, `CONFIRM → ModelOutput(confirm=self.confirm)`, and add `ranking=self.ranking` to the `VOTING` copy), and `merge_from` (copy `synthesis`, `confirm`, and `ranking` when non-empty). In `RoundState` add: `rankings: dict[str, list[str]] = Field(default_factory=dict)`, `borda_scores: dict[str, int] = Field(default_factory=dict)`, `synthesis_used: bool = False`, `synthesis_author: str = ""`, `confirm_tally: dict[str, int] = Field(default_factory=dict)`.

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: state scaffolding for synthesis/confirm phases + ranking"`

---

### Task 2: Parse ranking, synthesis, and confirm directives

**Files:**
- Modify: `src/coordinator.py` (parsing section)
- Test: `tests/test_parsing.py`

- [ ] **Step 1: Write failing tests**

```python
def test_parse_ranking_section():
    from src.coordinator import parse_output
    o = parse_output("## Vote\nFINALIZE: Participant A\n\n## Ranking\nB > C > A\n")
    assert o.vote is not None
    assert o.ranking == ["B", "C", "A"]

def test_parse_synthesis_and_confirm():
    from src.coordinator import parse_output
    assert parse_output("## Synthesis\nMerged answer here.\n").synthesis.startswith("Merged")
    assert parse_output("## Confirm\nAPPROVE\n").confirm == "APPROVE"
    assert parse_output("REJECT — the merge drops the minority view\n").confirm == "REJECT"
    assert parse_output("## Confirm\n(no directive)\n").confirm == ""
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.**
  - Add `Ranking|Synthesis|Confirm` to the `_SECTION_END` alternation and to the boundary regex inside `_headerless_vote_section`.
  - Add a confirm directive regex + detector mirroring the vote one:
    ```python
    _CONFIRM_DIRECTIVE = re.compile(r"(?im)^[\s>*_`-]*(APPROVE|REJECT)\b")
    def detect_confirm(text: str) -> str:
        m = _CONFIRM_DIRECTIVE.search(text or "")
        return m.group(1).upper() if m else ""
    ```
  - Add a ranking parser:
    ```python
    def parse_ranking(section: str) -> list[str]:
        line = next((ln.strip() for ln in section.splitlines() if ln.strip()), "")
        toks = [t.strip() for t in re.split(r"[>›]", line) if t.strip()]
        return toks
    ```
  - In `parse_output`: after the vote block, parse `## Ranking` via `_section(...)` → `output.ranking = parse_ranking(rank_sec)`. Parse `## Synthesis` → `output.synthesis`. For confirm, try `## Confirm` section first (`detect_confirm`), else run `detect_confirm(text)` on the whole text so a bare `APPROVE`/`REJECT` is recovered.

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: parse ranking, synthesis, and confirm directives"`

---

### Task 3: Borda tally and storage

**Files:**
- Modify: `src/coordinator.py` (consensus section)
- Test: `tests/test_consensus.py`

- [ ] **Step 1: Write failing test**

```python
def test_borda_tally_resolves_labels_and_scores():
    from src.coordinator import compute_borda
    from src.state import DebateState, RoundState, ModelOutput, Phase
    st = DebateState(debate_id="d", prompt="p")
    st.active_models = ["gpt4o", "claude", "deepseek"]
    st.participant_aliases = {"gpt4o": "Participant A", "claude": "Participant B", "deepseek": "Participant C"}
    rs = RoundState(round_num=1, phase=Phase.VOTING)
    rs.model_outputs = {
        "gpt4o": ModelOutput(ranking=["B", "C", "A"]),
        "claude": ModelOutput(ranking=["B", "A", "C"]),
        "deepseek": ModelOutput(ranking=["A", "B"]),  # partial ballot ok
    }
    st.rounds = [rs]
    scores = compute_borda(st)
    # B: 2+2+0 = 4 ; A: 0+1+1 = 2 ; C: 1+0+0 = 1
    assert scores == {"claude": 4, "gpt4o": 2, "deepseek": 1}
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** in `coordinator.py`:

```python
def _resolve_label(token: str, state: DebateState) -> str | None:
    aliases = state.participant_aliases or assign_aliases(state.active_models)
    up = token.upper()
    for name in state.active_models:
        if aliases.get(name, name).upper() in up:
            return name
    for name in state.active_models:
        letter = aliases.get(name, name).split()[-1].upper()
        if re.search(rf"\b{re.escape(letter)}\b", up):
            return name
    return None

def _resolve_ranking(tokens: list[str], state: DebateState) -> list[str]:
    out: list[str] = []
    for t in tokens:
        m = _resolve_label(t, state)
        if m and m not in out:
            out.append(m)
    return out

def compute_borda(state: DebateState) -> dict[str, int]:
    rs = state.round_state
    scores = {m: 0 for m in state.active_models}
    for out in rs.model_outputs.values():
        ballot = _resolve_ranking(out.ranking, state)
        k = len(ballot)
        for i, m in enumerate(ballot):
            scores[m] += (k - 1 - i)
    return scores
```

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: Borda tally over ranking ballots"`

---

### Task 4: Record Borda during voting + deadlock tiebreak

**Files:**
- Modify: `src/coordinator.py`
- Test: `tests/test_consensus.py`

- [ ] **Step 1: Write failing test**

```python
def test_endorsement_tiebreak_uses_borda():
    from src.coordinator import _endorsement_tally
    from src.state import DebateState, RoundState, Phase
    st = DebateState(debate_id="d", prompt="p")
    st.active_models = ["gpt4o", "claude"]
    rs = RoundState(round_num=1, phase=Phase.VOTING)
    rs.endorsements = {"gpt4o": "gpt4o", "claude": "claude"}  # 1–1 plurality tie
    rs.borda_scores = {"gpt4o": 1, "claude": 5}
    st.rounds = [rs]
    winner, top = _endorsement_tally(st)
    assert winner == "claude" and top == 1  # Borda breaks the tie
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.** Make `_endorsement_tally` break ties by `rs.borda_scores` (then by stable model order):

```python
def _endorsement_tally(state: DebateState) -> tuple[str | None, int]:
    rs = state.round_state
    counts: dict[str, int] = {}
    for target in rs.endorsements.values():
        counts[target] = counts.get(target, 0) + 1
    if not counts:
        return None, 0
    top = max(counts.values())
    tied = [k for k, c in counts.items() if c == top]
    winner = max(tied, key=lambda k: (rs.borda_scores.get(k, 0), -state.active_models.index(k)))
    return winner, top
```

  Then, in the `VOTING` branch of `coordinator_loop` (right after `_record_endorsements(state)`), record rankings + Borda before `determine_consensus`:

```python
            for name, out in rs.model_outputs.items():
                if out.ranking:
                    rs.rankings[name] = out.ranking
            rs.borda_scores = compute_borda(state)
```

- [ ] **Step 4: Run — expect PASS** (and `pytest tests/test_consensus.py -q` stays green — unique-majority finalize is unaffected because no tie occurs there).
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: record Borda each vote; break deadlock plurality ties by Borda"`

---

### Task 5: Agent participation gating + synthesis/confirm prompts

**Files:**
- Modify: `src/agent.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing test**

```python
def test_agent_participation_and_prompt_phases():
    from src.agent import _agent_participates, build_agent_prompt
    from src.state import DebateState, RoundState, Phase
    st = DebateState(debate_id="d", prompt="p")
    st.active_models = ["gpt4o", "claude"]
    rs = RoundState(round_num=1, phase=Phase.SYNTHESIS)
    rs.consensus_winner = "gpt4o"
    st.rounds = [rs]; st.current_round = 1; st.current_phase = Phase.SYNTHESIS
    assert _agent_participates("gpt4o", st) is True     # winner authors synthesis
    assert _agent_participates("claude", st) is False    # others sit it out
    rs.phase = Phase.CONFIRM; st.current_phase = Phase.CONFIRM
    assert _agent_participates("claude", st) is True      # everyone confirms
    _, user = build_agent_prompt("claude", st)
    assert "CONFIRM" in user
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.**
  - Add helper:
    ```python
    def _agent_participates(model_name: str, state: DebateState) -> bool:
        rs = state.round_state
        if rs.phase == Phase.SYNTHESIS:
            return model_name == rs.consensus_winner
        return True
    ```
    (import `Phase` in agent.py.)
  - In `agent_loop`, after the `rs.phase not in PHASE_FILE_SUFFIX` guard, add:
    ```python
        if not _agent_participates(model_name, state):
            await asyncio.sleep(poll_interval)
            continue
    ```
  - Extend `SYSTEM_PROMPT`: add to THE PROCESS a line for the optional post-consensus phases, and add two output-format blocks:
    ```
    ## Ranking
    [Optional, VOTING only: rank ALL participant labels best-to-worst, e.g. "B > C > A".]

    ## Synthesis
    [SYNTHESIS phase only: write the single best merged answer, integrating the
     strongest points across proposals and explicitly preserving any minority view.]

    ## Confirm
    [CONFIRM phase only: first line exactly APPROVE or REJECT — does the synthesis
     faithfully capture the group's conclusion?]
    ```
    Also append to the VOTING format note: "You may also add a `## Ranking` line."
  - In `build_agent_prompt`, when `rs.phase == Phase.SYNTHESIS`, append the winner's own task context, and when `rs.phase == Phase.CONFIRM`, surface the synthesis text:
    ```python
        if rs.phase == Phase.CONFIRM:
            syn = rs.model_outputs.get(rs.consensus_winner)
            if syn and syn.synthesis:
                context_parts.append(f"\n\nPROPOSED SYNTHESIS TO CONFIRM:\n{syn.synthesis}")
    ```
    (The existing final `YOUR TASK: You are in the {phase} phase` line already names SYNTHESIS/CONFIRM in uppercase, satisfying the test's `"CONFIRM" in user`.)

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: agent gating for winner-only synthesis + confirm prompts"`

---

### Task 6: Coordinator orchestration — finalize → synthesis → confirm, with verbatim fallback

**Files:**
- Modify: `src/coordinator.py`
- Test: `tests/test_consensus.py` (unit) + covered e2e in Task 8

- [ ] **Step 1: Write failing test** (participant-set helper + fallback semantics)

```python
def test_phase_participants_synthesis_is_winner_only():
    from src.coordinator import _phase_participants
    from src.state import DebateState, RoundState, Phase
    st = DebateState(debate_id="d", prompt="p")
    st.active_models = ["gpt4o", "claude", "deepseek"]
    rs = RoundState(round_num=1, phase=Phase.SYNTHESIS)
    rs.consensus_winner = "claude"
    st.rounds = [rs]
    assert _phase_participants(st) == ["claude"]
    rs.phase = Phase.CONFIRM
    assert _phase_participants(st) == st.active_models
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.**
  - Add:
    ```python
    def _phase_participants(state: DebateState) -> list[str]:
        rs = state.round_state
        if rs.phase == Phase.SYNTHESIS and rs.consensus_winner:
            return [rs.consensus_winner]
        return state.active_models
    ```
  - Change `all_phase_complete` to iterate `_phase_participants(state)` instead of `state.active_models`.
  - Add a verbatim-finalize helper:
    ```python
    def _finalize(state, debate_dir, *, synthesis: bool) -> str:
        rs = state.round_state
        rs.synthesis_used = synthesis
        if synthesis:
            rs.synthesis_author = rs.consensus_winner
        state.status = "done"
        state.current_phase = Phase.DONE
        write_final_answer(state, debate_dir)
        save_state(state, debate_dir / "state.json")
        return "done"
    ```
  - In `coordinator_loop`, VOTING branch: replace the `if action == "finalize":` body so it *starts* synthesis instead of finishing:
    ```python
            if action == "finalize":
                _set_phase(state, Phase.SYNTHESIS)
                save_state(state, state_path)
                # fall through; loop drives SYNTHESIS next
    ```
  - Add two new `elif` branches after the VOTING branch:
    ```python
        elif rs.phase == Phase.SYNTHESIS:
            _set_phase(state, Phase.CONFIRM)
            save_state(state, state_path)

        elif rs.phase == Phase.CONFIRM:
            approve = reject = 0
            for out in rs.model_outputs.values():
                if out.confirm == "APPROVE": approve += 1
                elif out.confirm == "REJECT": reject += 1
            rs.confirm_tally = {"APPROVE": approve, "REJECT": reject}
            return _finalize(state, debate_dir, synthesis=approve >= state.majority())
    ```
  - **Fallback wiring (critical):** consensus is already reached once we leave VOTING, so a failure/stall in SYNTHESIS/CONFIRM must finalize verbatim, never deadlock. In the stall handler and the `failed`-models handler, guard with the phase:
    ```python
        if rs.phase in (Phase.SYNTHESIS, Phase.CONFIRM):
            salvage_phase_outputs(debate_dir, state)
            return _finalize(state, debate_dir, synthesis=False)
    ```
    Place this check at the top of both the stall branch and the `if failed:` branch (before the generic drop/deadlock logic), so a winner that can't synthesize, or a confirmer that hangs, ships today's verbatim answer.

- [ ] **Step 4: Run — expect PASS** + `pytest tests/test_consensus.py tests/test_state.py -q` green.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: coordinator drives synthesis→confirm with verbatim fallback"`

---

### Task 7: Render synthesis + Borda in final.md

**Files:**
- Modify: `src/coordinator.py` (`render_final_answer`)
- Test: `tests/test_consensus.py`

- [ ] **Step 1: Write failing test**

```python
def test_final_renders_synthesis_and_ranking():
    from src.coordinator import render_final_answer
    from src.state import DebateState, RoundState, ModelOutput, Phase
    st = DebateState(debate_id="d", prompt="p"); st.status = "done"
    st.active_models = ["gpt4o", "claude"]
    rs = RoundState(round_num=1, phase=Phase.CONFIRM)
    rs.consensus_winner = "gpt4o"
    rs.endorsements = {"gpt4o": "gpt4o", "claude": "gpt4o"}
    rs.model_outputs = {
        "gpt4o": ModelOutput(proposal="raw winner proposal", synthesis="MERGED + minority"),
        "claude": ModelOutput(proposal="other"),
    }
    rs.synthesis_used = True; rs.synthesis_author = "gpt4o"
    rs.borda_scores = {"gpt4o": 2, "claude": 1}
    st.rounds = [rs]; st.current_round = 1
    out = render_final_answer(st)
    assert "MERGED + minority" in out and "Synthesis" in out
    assert "Ranking" in out  # Borda block present
    # verbatim still available below
    assert "raw winner proposal" in out
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.** In `render_final_answer`, where the consensus answer is rendered (the `win_out` block): if `state.status == "done" and rs.synthesis_used and rs.model_outputs.get(rs.consensus_winner, ModelOutput()).synthesis`, lead with a `## Synthesis (group-confirmed)` section containing the synthesis text and a note `*Merged from all proposals; confirmed {confirm_tally}*`, and keep the verbatim winner proposal under its existing `## Consensus Answer`/`## All Proposals` sections. Add a Borda block before `## Cost` when `rs.borda_scores` is non-empty:

```python
    if rs.borda_scores:
        ranked = sorted(rs.borda_scores.items(), key=lambda kv: kv[1], reverse=True)
        lines += ["## Ranking (Borda)", ""]
        for name, score in ranked:
            lines.append(f"- {provider_name(name)}: {score}")
        lines.append("")
```

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: render confirmed synthesis + Borda ranking in final.md"`

---

### Task 8: End-to-end — synthesis approved and rejected paths

**Files:**
- Modify: `tests/test_flow.py`
- Test: same

- [ ] **Step 1: Extend `make_fake`** so phases beyond voting are handled. Add params and branches:

```python
def make_fake(name, vote_line="FINALIZE", confirm="APPROVE"):
    async def fake(system, user, model_id):
        if "PHASE: proposing" in user:
            return f"## Proposal\nAnswer from {name}: the result is 42.\n"
        if "PHASE: reviewing" in user:
            return f"## Reviews\n{name} notes the others are sound.\n"
        if "PHASE: rebuttal" in user:
            return f"## Rebuttal\n{name} stands by the proposal.\n"
        if "PHASE: voting" in user:
            return (f"## Vote\n{vote_line}\n\n## Ranking\nA > B > C\n"
                    f"\n## Reasoning\n{name} reasoning.\n")
        if "PHASE: synthesis" in user:
            return f"## Synthesis\nMerged answer from {name}; minority preserved.\n"
        if "PHASE: confirm" in user:
            return f"## Confirm\n{confirm}\n"
        return ""
    return fake
```

  Thread an optional `confirms: dict[str,str]` through `_setup` (default all `"APPROVE"`).

- [ ] **Step 2: Add tests**

```python
def test_synthesis_used_when_confirmed(monkeypatch, tmp_path):
    state, debate_dir = _setup(monkeypatch, tmp_path, _all_finalize_a())
    status = asyncio.run(asyncio.wait_for(_drive(debate_dir, state.active_models), timeout=20))
    assert status == "done"
    saved = load_state(debate_dir / "state.json")
    rs = saved.round_state
    assert rs.synthesis_used is True
    assert rs.confirm_tally["APPROVE"] == len(MODELS)
    final = (debate_dir / "final.md").read_text()
    assert "Synthesis" in final and "minority preserved" in final
    assert "Ranking (Borda)" in final
    # winner-only synthesis artifact
    winner = rs.consensus_winner
    assert (debate_dir / "round-001" / f"{winner}.synthesis.md").exists()
    for n in MODELS:
        assert (debate_dir / "round-001" / f"{n}.confirm.md").exists()

def test_synthesis_rejected_falls_back_to_verbatim(monkeypatch, tmp_path):
    state, debate_dir = _setup(monkeypatch, tmp_path, _all_finalize_a(),
                               confirms={n: "REJECT" for n in MODELS})
    status = asyncio.run(asyncio.wait_for(_drive(debate_dir, state.active_models), timeout=20))
    assert status == "done"
    saved = load_state(debate_dir / "state.json")
    assert saved.round_state.synthesis_used is False
    final = (debate_dir / "final.md").read_text()
    # today's behavior preserved: every proposal still present
    for n in MODELS:
        assert f"Answer from {n}" in final
```

- [ ] **Step 3: Run — expect FAIL** (until `_setup` threads `confirms`).
- [ ] **Step 4: Implement `_setup` change**, run full suite `pytest -q`.
- [ ] **Step 5: Run — expect PASS** for the whole suite, then `ruff check .`.
- [ ] **Step 6: Commit** `git add -A && git commit -m "test: e2e synthesis-approved and rejected-fallback paths"`

---

## Self-review

- **Spec coverage:** Borda source/tally/recording (T2–4,7), narrow deadlock tiebreak (T4), synthesis author+confirm gate (T5,6), consensus-only trigger + deadlock unchanged (T6 guards), verbatim fallback on reject/error/stall (T6), artifacts + state fields (T1,6), `final.md` rendering (T7), tests incl. fallback (T8). README update is intentionally deferred to a final docs pass after the suite is green.
- **Type consistency:** `compute_borda`, `_resolve_ranking`, `_resolve_label`, `_phase_participants`, `_finalize`, `_agent_participates`, `detect_confirm`, `parse_ranking` are defined once and referenced consistently; `RoundState` field names match across tasks (`rankings`, `borda_scores`, `synthesis_used`, `synthesis_author`, `confirm_tally`).
- **Placeholder scan:** none — every code step shows real code.
- **Docs:** add a README "Synthesis & ranking" note after Task 8 passes (non-blocking).
</content>
