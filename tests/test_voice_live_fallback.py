"""Live voice fallback must not die silently. When the Gemini Live websocket
errored or closed mid-session, the handlers called stopLive(); startSrLive();
- but stopLive() clears the live flag and startSrLive's recognition loop
exits immediately on `if (!live) return`, leaving the user with NO voice at
all (matches the Android report: "not hearing"). startSrLive must arm itself.
Also: the mic processing chain must tap through a zero-gain node - connecting
it straight to the destination plays the mic through the speaker (feedback)."""
import os
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("NEON_DATABASE_URL", None)
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")

import re

import app as appmod

JS = open(os.path.join(os.path.dirname(appmod.__file__), "app.js")).read()


def _fn(name):
    start = JS.index("function " + name + "(")
    # next top-level (2-space indent) function declaration closes the body
    m = re.search(r"\n  function ", JS[start + 1:])
    return JS[start:start + 1 + (m.start() if m else len(JS))]


def test_sr_fallback_arms_itself():
    body = _fn("startSrLive")
    assert re.search(r"live = true;\s*liveBtn\.classList\.add\('listening'\);", body), \
        "startSrLive must set live=true itself - callers reach it after stopLive() cleared the flag"


def test_ws_drop_paths_fall_back_to_sr():
    onclose = re.search(r"liveWs\.onclose = \(\) => \{([^}]*)\}", JS)
    onerror = re.search(r"liveWs\.onerror = \(\) => \{([^}]*)\}", JS)
    assert onclose and "startSrLive()" in onclose.group(1)
    assert onerror and "startSrLive()" in onerror.group(1)


def test_sr_loop_survives_ws_close_path():
    # The regression lock for the silent-death chain: after stopLive() runs
    # (live=false), calling startSrLive() must still start the recognition
    # loop - guaranteed only by the self-arming line tested above plus the
    # loop's live check.
    body = _fn("startSrLive")
    arm_at = body.index("live = true")
    loop_at = body.index("if (!live) return;")
    assert arm_at < loop_at, "arming must happen before the loop's live check"


def test_mic_chain_uses_zero_gain_tap():
    assert "tap.gain.value = 0" in JS, "mic chain must tap through a zero-gain node"
    assert "liveNode.connect(liveCtx.destination)" not in JS, \
        "direct mic->destination connection plays the mic through the speaker"
