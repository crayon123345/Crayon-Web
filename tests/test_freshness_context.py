"""Freshness grounding: the model must know today's date and the freshness rule,
or it cannot judge what "yesterday"/"latest" means (live failure 2026-10-15:
a Nov-2024 cricket Test was narrated as "yesterday's" result)."""
import datetime

import llm


def test_system_message_carries_todays_utc_date():
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    msgs = llm.to_messages([], "hi")
    assert f"Today's date: {today} (UTC)." in msgs[0].content


def test_system_prompt_has_freshness_rule():
    msgs = llm.to_messages([], "hi")
    assert "Never present old information as if it were current" in msgs[0].content
    assert "could not find it" in msgs[0].content
