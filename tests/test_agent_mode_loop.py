"""Agent mode plan/tool co-occurrence: a reply carrying BOTH the re-emitted
plan block and the next tool block must have the tool block EXECUTED, not
swallowed by the plan branch (live bug 2026-10-15: agent mode showed a plan,
then stopped - the plan branch `continue`d and dropped the tool block that
came in the same reply). The prompt and the frontend must agree on the
single-round-trip rhythm: progress line + plan + one tool block per reply."""
import os
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("NEON_DATABASE_URL", None)
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")

import re

import app as appmod
import llm

JS = open(os.path.join(os.path.dirname(appmod.__file__), "app.js")).read()


def _plan_branch():
    start = JS.index("const pm = agentMode")
    end = JS.index("const gm =", start)
    return JS[start:end]


def test_plan_branch_keeps_tool_blocks():
    branch = _plan_branch()
    # The plan is stripped from the reply and the branch only `continue`s on
    # the plan-ONLY path; a co-occurring tool block falls through to the tool
    # handlers below.
    assert "planCard(pm[1])" in branch
    assert re.search(r"!\/```\(gmail\|web-fetch\|weather\|fx\|stock\|web-search\|image-gen\|python-run\)", branch), \
        "plan branch must detect a co-occurring tool block"
    assert branch.count("continue;") == 1, "only the plan-only path may continue"


def test_plan_regex_tolerates_tag_whitespace():
    assert re.search(r"```plan\\s\*\\n", JS), "plan regex must allow ```plan <spaces> newline"


def test_plan_only_stall_is_capped():
    branch = _plan_branch()
    assert "planStalls" in branch and "planStalls > 2" in branch, \
        "repeated plan-only replies must break instead of burning the tool budget"


def test_prompt_defines_single_round_trip_rhythm():
    p = llm.AGENT_PROMPT
    assert "all three together" in p, "prompt must require progress line + plan + tool block in ONE reply"
    assert "Never reply with only a plan block after the first" in p


def test_prompt_and_frontend_nudge_agree():
    # The frontend's plan-only nudge must ask for exactly what the prompt's
    # working rhythm describes, or the model gets contradictory instructions.
    m = re.search(r"msg = '(\[Plan shown[^\']*)'", JS)
    assert m, "frontend plan-only nudge missing"
    nudge = m.group(1)
    assert "exactly one tool block" in nudge
    assert "[x]" in nudge and "->" in nudge
