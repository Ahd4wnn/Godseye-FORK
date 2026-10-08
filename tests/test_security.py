"""Security audit as executable checks: secrets in the tree, SQL construction, dangerous calls, credential leakage through
GStreamer error text, filesystem traversal. (Prompt injection / LLM tool execution: test_query_*; web/API/media: test_web.py;
privacy masks + retention: test_storage.py.)"""
import ast
import logging
import re
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = [p for p in (ROOT / "godseye").rglob("*.py")] + list((ROOT / "tools").glob("*.py")) + list((ROOT / "godseye_remote").glob("*.py"))


def tree_files():
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    return [ROOT / f for f in out if f and (ROOT / f).is_file() and (ROOT / f).suffix in {".py", ".yaml", ".yml", ".md", ".txt", ".lock", ".toml", ".json", ""}]


SECRET = re.compile(r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|sk-[A-Za-z0-9]{32,}"
                    r"|(?i:password|passwd|secret|api[_-]?key)\s*[:=]\s*['\"][^'\"$\s{][^'\"]{5,}['\"]")


def test_no_secrets_in_the_tree():
    hits = []
    for f in tree_files():
        if f.name == "test_security.py":
            continue
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if SECRET.search(line):
                hits.append(f"{f.relative_to(ROOT)}:{i}")
    assert not hits, hits


def test_gitignore_keeps_models_data_and_venvs_out():
    gi = (ROOT / ".gitignore").read_text()
    for entry in (".venv/", ".venv-ray/", "models/", "data/", "third_party/"):
        assert entry in gi
    assert not any("web_token" in str(f) or f.suffix == ".db" for f in tree_files())


# ---------------------------------------------------------------- SQL construction
SAFE_FSTRING = {"_COLS", "_ORDER[plan.ranking_strategy]", "where"}


def sql_sites():
    for f in PKG:
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("execute", "executemany", "executescript") and node.args:
                yield f, node, node.args[0]


def _placeholders(n) -> bool:
    """`",".join("?" * k)` -- a run of bound-parameter markers, nothing else."""
    return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "join" and isinstance(n.func.value, ast.Constant)
            and n.func.value.value == "," and len(n.args) == 1 and isinstance(n.args[0], ast.BinOp) and isinstance(n.args[0].op, ast.Mult)
            and isinstance(n.args[0].left, ast.Constant) and n.args[0].left.value == "?")


def _reviewed(arg) -> bool:
    if isinstance(arg, ast.Constant):
        return True
    if isinstance(arg, ast.Name):                                             # module constants / variables built from reviewed forms
        return arg.id in {"SCHEMA", "sql", "q", "query", "where"}   # `where` = _where(): reviewed fragments + bound parameters
    if isinstance(arg, ast.JoinedStr):                                        # f"...": only reviewed constants may be interpolated
        return {ast.unparse(v.value) for v in arg.values if isinstance(v, ast.FormattedValue)} <= SAFE_FSTRING | {"SCHEMA_VERSION", "v"}
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):            # "... IN (%s)" % ",".join("?" * n)
        return isinstance(arg.left, (ast.Constant, ast.Name)) and (_placeholders(arg.right) or isinstance(arg.right, ast.Name) and arg.right.id == "ph")
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):            # "BEGIN;" + SCHEMA + f"PRAGMA user_version={N};COMMIT;"
        return _reviewed(arg.left) and _reviewed(arg.right)
    if isinstance(arg, ast.IfExp):                                            # 'ROLLBACK' if error else 'COMMIT'
        return _reviewed(arg.body) and _reviewed(arg.orelse)
    if isinstance(arg, ast.Subscript):                                        # MIGRATIONS[v]
        return isinstance(arg.value, ast.Name) and arg.value.id == "MIGRATIONS"
    return False


def test_sql_is_constant_or_built_only_from_reviewed_fragments():
    bad = [f"{f.relative_to(ROOT)}:{call.lineno}: {ast.unparse(arg)[:90]}" for f, call, arg in sql_sites() if not _reviewed(arg)]
    assert not bad, bad


def test_sql_variables_are_only_assembled_from_constants():
    """`sql`/`q` variables passed to execute must be built from constants / reviewed fragments in the same function."""
    for f in PKG:
        src = f.read_text()
        for m in re.finditer(r"\b(?:sql|q)\s*=\s*(.+)", src):
            expr = m.group(1)
            if "execute" in src and ("f\"" in expr or "f'" in expr):
                assert all(x in SAFE_FSTRING or x.startswith("_ORDER") for x in re.findall(r"\{([^}]+)\}", expr)), (f.name, expr[:80])


# ---------------------------------------------------------------- dangerous calls
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}
FORBIDDEN_ATTRS = {("os", "system"), ("os", "popen"), ("pickle", "loads"), ("pickle", "load"), ("yaml", "load"), ("yaml", "unsafe_load"),
                   ("marshal", "loads"), ("shutil", "rmtree")}
SUBPROCESS_OK = {"clips.py", "metrics.py", "_paths.py"}            # fixed argv, shell=False (reviewed)


def test_no_dangerous_calls_and_subprocess_is_argv_only():
    bad = []
    for f in PKG:
        for node in ast.walk(ast.parse(f.read_text())):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_CALLS:
                bad.append(f"{f.name}:{node.lineno} {fn.id}")
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                if (fn.value.id, fn.attr) in FORBIDDEN_ATTRS:
                    bad.append(f"{f.name}:{node.lineno} {fn.value.id}.{fn.attr}")
                if fn.value.id == "subprocess":
                    if f.name not in SUBPROCESS_OK and f.parent.name != "tools":
                        bad.append(f"{f.name}:{node.lineno} subprocess outside the reviewed files")
                    if any(k.arg == "shell" and not (isinstance(k.value, ast.Constant) and k.value.value is False) for k in node.keywords):
                        bad.append(f"{f.name}:{node.lineno} shell=True")
                    if node.args and not isinstance(node.args[0], (ast.List, ast.Name)):
                        bad.append(f"{f.name}:{node.lineno} subprocess argv is not a list")
            for k in node.keywords:
                if k.arg == "verify" and isinstance(k.value, ast.Constant) and k.value.value is False:
                    bad.append(f"{f.name}:{node.lineno} TLS verify disabled")
    assert not bad, bad


def test_torch_load_is_limited_to_the_pinned_checkpoint_loader():
    users = [f.name for f in PKG if "torch.load(" in f.read_text()]
    assert users == ["reid.py"], users
    assert "weights_only=False" in (ROOT / "godseye/perception/reid.py").read_text()      # documented: trusted, pinned release file


# ---------------------------------------------------------------- credentials never reach logs / health
def test_gstreamer_errors_do_not_leak_camera_passwords(gst, make_cfg, caplog):
    from godseye.ingest import IngestManager
    caplog.set_level(logging.DEBUG)
    m = IngestManager(make_cfg({"camera_id": "c", "url": "rtsp://admin:topsecret123@127.0.0.1:9/stream",
                                "reconnect": {"initial_backoff_s": 0.2, "max_backoff_s": 0.4, "connect_timeout_s": 2, "max_attempts": 2}}))
    m.start()
    try:
        t = time.time()
        while time.time() - t < 20 and m.stats()["c"].state.value != "failed":
            time.sleep(0.2)
        st = m.stats()["c"]
        assert st.last_error                                               # something failed, and it is described
        assert "topsecret123" not in st.last_error
        assert "topsecret123" not in caplog.text
    finally:
        m.stop()


def test_redact_text_covers_urls_in_free_text():
    from godseye.ingest.config import redact_text
    for raw in ("Could not open rtsp://u:p@h:554/x", "x http://a:b@c/d y https://tok@h/z", "rtmp://user:pw@10.0.0.1/live/stream: refused"):
        out = redact_text(raw)
        assert "@" in out and not re.search(r"://[^*/\s]+@", out), out


# ---------------------------------------------------------------- filesystem
def test_camera_ids_are_path_safe_everywhere():
    from godseye.ingest.config import ConfigError, parse_camera
    for bad in ("../x", "a/b", "a b", "", "x" * 65 + "/", "c;rm -rf"):
        with pytest.raises(ConfigError):
            parse_camera({"camera_id": bad, "url": "file:///tmp/x.mp4"}, {}, 0)
