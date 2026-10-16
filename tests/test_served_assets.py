"""The served web app must BE the repo-root sources. A tracked static/ directory
once shadowed them: app.py only copies root assets into static/ when static/ is
missing, so the stale committed copy was what Render actually served (verified
live 2026-10-15: /static/app.js was a 498-line stale file, no tz payload)."""
import os
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("NEON_DATABASE_URL", None)
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")

from fastapi.testclient import TestClient
import app as appmod


def test_served_app_js_is_root_source_with_tz():
    root_js = open(os.path.join(os.path.dirname(appmod.__file__), "app.js")).read()
    r = TestClient(appmod.app).get("/static/app.js")
    assert r.status_code == 200
    assert r.text == root_js
    assert "timeZone" in r.text, "served JS must post the visitor timezone"


def test_served_index_is_root_source():
    root_html = open(os.path.join(os.path.dirname(appmod.__file__), "index.html")).read()
    r = TestClient(appmod.app).get("/")
    assert r.status_code == 200
    assert r.text == root_html


def test_no_tracked_static_dir():
    import subprocess
    out = subprocess.run(["git", "ls-files", "static"], capture_output=True, text=True,
                         cwd=os.path.dirname(appmod.__file__)).stdout.strip()
    assert out == "", f"static/ must not be tracked: {out[:200]}"
