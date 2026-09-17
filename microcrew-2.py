#!/usr/bin/env python3
"""
microcrew.py — a small multi-role coding crew on a LiteLLM / OpenAI-compatible gateway.
Standard library only. Works with or without git: in a git repo it uses branches, worktrees
(parallel items) and merges; in a plain directory it uses file snapshots under .crew/ and works
items one at a time. Two modes:

  campaign (default)  The crew reads the latest run output and the repo, writes its OWN backlog
                      of independent fixes, then works the items — in parallel git worktrees when
                      their file sets don't overlap — each as plan -> implement -> review, merges
                      every PASSed item into the campaign branch with the test suite as the gate,
                      and packages a zip + CAMPAIGN.md hand-off.
  single              One pre-specified change: triage -> plan -> implement -> review -> package.

Roles (each phase is a fresh conversation with its own system prompt and tool set):
  senior  read-only   survey/backlog, review gate, hand-off
  ml      read-only   plans anything the runtime LLM sees or returns (prompts, schema, parsing, retries)
  fuzz    read-only   plans fuzzer/seed/harness/health-checker changes
  swe     read/write  implements plans, writes tests, runs the suite, commits, resolves merges

Phases hand off through files in <workspace>/.crew/ (BACKLOG.json, PLAN.md, IMPLEMENT.md, REVIEW.md,
HANDOFF.md). Every turn is printed and saved to .crew/transcript-*.jsonl. Nothing under the run
output can be modified. .crew/ and the zip never enter git (.git/info/exclude).

Env (same as microagent.py):
  LITELLM_BASE_URL  LITELLM_API_KEY  AGENT_MODEL  AGENT_TLS_VERIFY=false
Optional:
  CREW_MODEL_SENIOR / CREW_MODEL_ML / CREW_MODEL_FUZZ / CREW_MODEL_SWE   per-role models
  CREW_MAX_TURNS (per phase, default 80)  CREW_LLM_TIMEOUT (s, 300)  CREW_MAX_TOKENS (4096)
  CREW_TEST_CMD (default: python -m py_compile src/*.py && python -m pytest -q tests)

Usage:
  python microcrew.py --repo . --verify
  python microcrew.py --repo . --run discver --task TASK-discver.md                # campaign, sequential
  python microcrew.py --repo . --run discver --task TASK-discver.md --parallel 2 --max-items 8
  python microcrew.py --repo . --run discver --task TASK-discver.md --mode single
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import difflib
import json
import os
import shutil
import pathlib
import re
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

# ----------------------------------------------------------------------------- config
BASE_URL = os.environ.get("LITELLM_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("LITELLM_API_KEY", "")
MODEL = os.environ.get("AGENT_MODEL", "")
VERIFY_TLS = os.environ.get("AGENT_TLS_VERIFY", "true").strip().lower() not in ("0", "false", "no")
MAX_TURNS = int(os.environ.get("CREW_MAX_TURNS", "80"))
LLM_TIMEOUT = int(os.environ.get("CREW_LLM_TIMEOUT", "300"))
MAX_TOKENS = int(os.environ.get("CREW_MAX_TOKENS", "4096"))
TEST_CMD = os.environ.get("CREW_TEST_CMD", "python -m py_compile src/*.py && python -m pytest -q tests")
TOOL_OUT_LIMIT = 12_000
CTX_LIMIT_CHARS = 350_000
READ_MAX_LINES = 300
ROLES = ("senior", "ml", "fuzz", "swe")
ROLE_MODEL = {r: os.environ.get(f"CREW_MODEL_{r.upper()}") or MODEL for r in ROLES}
_TOKENS_KEY = "max_tokens"
_PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def now() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# ----------------------------------------------------------------------------- LLM
def _post(payload: dict) -> dict:
    req = urllib.request.Request(BASE_URL + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
                                 method="POST")
    ctx = None if VERIFY_TLS else ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT, context=ctx) as r:
        return json.loads(r.read().decode())


def chat(model: str, messages: list, tools: list | None, tag: str = "") -> dict:
    """One completion with retries; returns the assistant message dict."""
    global _TOKENS_KEY
    last_err = None
    for attempt in range(1, 4):
        payload = {"model": model, "messages": messages, _TOKENS_KEY: MAX_TOKENS}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        t0 = time.time()
        try:
            resp = _post(payload)
            choice = resp["choices"][0]
            usage = resp.get("usage", {})
            log(f"      [{tag}llm] {model} {time.time()-t0:.1f}s finish={choice.get('finish_reason')} "
                f"in={usage.get('prompt_tokens','?')} out={usage.get('completion_tokens','?')}")
            return choice["message"]
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:600]
            last_err = f"HTTP {e.code}: {body}"
            if e.code == 400 and "max_tokens" in body and _TOKENS_KEY == "max_tokens":
                _TOKENS_KEY = "max_completion_tokens"
                log(f"      [{tag}llm] gateway wants max_completion_tokens; switching")
                continue
            if e.code in (400, 401, 403, 404):
                break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        log(f"      [{tag}llm] attempt {attempt} failed after {time.time()-t0:.0f}s: {last_err}")
        time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"LLM call failed: {last_err}")


# ----------------------------------------------------------------------------- workspace + tools
_DENY = re.compile(r"(\brm\s+-rf\s+/(\s|$)|git\s+push\b.*--force|\bmkfs\b|\bshutdown\b|\breboot\b|>\s*/dev/sd)")
_MUTATING = re.compile(r"(\bsed\s+-i\b|\brm\s|\bmv\s|\bcp\s|(?<![0-9])>>?(?!&)|\btee\b|"
                       r"\bgit\s+(add|commit|checkout|reset|clean|stash|rebase|merge|worktree)\b|\bpip\s+install\b|\bchmod\b|\btouch\b)")


def git(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


IGNORE_DIRS = {".crew", ".git", "__pycache__", ".pytest_cache", "output", "oss-crs", "test-targets", "node_modules", ".venv", "venv"}
IGNORE_SUFFIXES = (".pyc", ".zip", ".log", ".jar", ".class")


def _tree_files(root: pathlib.Path, run: pathlib.Path | None) -> dict:
    """rel path -> absolute path for every file that counts as 'the code' (build junk, run output, framework checkouts excluded)."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dp = pathlib.Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not (run and (dp / d).resolve() == run)]
        for f in filenames:
            if f.endswith(IGNORE_SUFFIXES):
                continue
            out[str((dp / f).relative_to(root))] = dp / f
    return out


def snapshot(root: pathlib.Path, run: pathlib.Path | None, dest: pathlib.Path) -> int:
    if dest.exists():
        shutil.rmtree(dest)
    files = _tree_files(root, run)
    for rel, src in files.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / rel)
    return len(files)


def restore(dest: pathlib.Path, root: pathlib.Path, run: pathlib.Path | None) -> None:
    """Put the tree back exactly as the snapshot had it: overwrite changed files, delete files that did not exist."""
    snap = _tree_files(dest, None)
    cur = _tree_files(root, run)
    for rel, src in snap.items():
        target = root / rel
        if not target.exists() or target.read_bytes() != src.read_bytes():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    for rel in set(cur) - set(snap):
        (root / rel).unlink(missing_ok=True)


def changed_files(dest: pathlib.Path, root: pathlib.Path, run: pathlib.Path | None) -> list:
    snap, cur = _tree_files(dest, None), _tree_files(root, run)
    out = [rel for rel in cur if rel not in snap or cur[rel].read_bytes() != snap[rel].read_bytes()]
    out += [rel for rel in snap if rel not in cur]
    return sorted(out)


def make_patch(dest: pathlib.Path, root: pathlib.Path, run: pathlib.Path | None) -> str:
    """Unified diff of the tree against a snapshot (text files only)."""
    snap, cur = _tree_files(dest, None), _tree_files(root, run)
    chunks = []
    for rel in changed_files(dest, root, run):
        a = snap[rel].read_bytes() if rel in snap else b""
        b = cur[rel].read_bytes() if rel in cur else b""
        if b"\x00" in a or b"\x00" in b:
            chunks.append(f"Binary file {rel} changed\n")
            continue
        chunks.extend(difflib.unified_diff(a.decode(errors="replace").splitlines(True), b.decode(errors="replace").splitlines(True),
                                           fromfile=f"a/{rel}", tofile=f"b/{rel}"))
    return "".join(chunks) or "(no changes)\n"


def _clip(s: str, limit: int = TOOL_OUT_LIMIT) -> str:
    if len(s) <= limit:
        return s
    return f"{s[: limit*2//3]}\n... [{len(s)-limit} chars elided — narrow the request] ...\n{s[-limit//3:]}"


class Workspace:
    """A repo checkout (main repo or a git worktree) plus the shared read-only run output."""

    def __init__(self, repo: pathlib.Path, run: pathlib.Path | None, tag: str, git_mode: bool = True):
        self.repo = pathlib.Path(repo).resolve()
        self.run = pathlib.Path(run).resolve() if run else None
        self.crew = self.repo / ".crew"
        self.crew.mkdir(exist_ok=True)
        self.tag = tag
        self.git_mode = git_mode

    def resolve(self, p: str) -> pathlib.Path:
        path = pathlib.Path(p)
        if not path.is_absolute():
            path = self.repo / path
        path = path.resolve()
        for root in [self.repo] + ([self.run] if self.run else []):
            if path == root or root in path.parents:
                return path
        raise ValueError(f"path outside repo/run dirs: {p}")

    def in_run(self, path: pathlib.Path) -> bool:
        return bool(self.run) and (path == self.run or self.run in path.parents)

    def note(self, name: str, default: str = "MISSING") -> str:
        p = self.crew / name
        return p.read_text() if p.exists() else default

    # --- tools
    def t_list_dir(self, path: str = ".") -> str:
        p = self.resolve(path)
        if p.is_file():
            return f"{p} is a file ({p.stat().st_size} bytes)"
        rows = [f"{'d' if c.is_dir() else 'f'} {c.stat().st_size:>10} {c.name}"
                for c in sorted(p.iterdir()) if c.name != ".git"]
        return "\n".join(rows) or "(empty)"

    def t_read_file(self, path: str, start: int = 1, end: int | None = None) -> str:
        p = self.resolve(path)
        lines = p.read_text(errors="replace").splitlines()
        total = len(lines)
        start = max(1, int(start))
        end = min(total, int(end) if end else start + READ_MAX_LINES - 1, start + READ_MAX_LINES - 1)
        body = "\n".join(f"{i:6d}\t{lines[i-1]}" for i in range(start, end + 1))
        return f"[{p.name}: lines {start}-{end} of {total}]\n{body}"

    def t_search(self, pattern: str, path: str = ".", max_hits: int = 60) -> str:
        p = self.resolve(path)
        r = subprocess.run(["grep", "-rnIE", "--exclude-dir=.git", "--exclude-dir=.crew", "--exclude-dir=__pycache__",
                            "-e", pattern, str(p)], capture_output=True, text=True, timeout=120)
        hits = [h[:400] for h in r.stdout.splitlines()]
        shown = hits[: int(max_hits)]
        more = f"\n... {len(hits)-len(shown)} more hits" if len(hits) > len(shown) else ""
        return ("\n".join(shown) or "(no matches)") + more

    def t_run(self, cmd: str, timeout: int = 600, read_only: bool = False) -> str:
        if _DENY.search(cmd):
            return "REFUSED: destructive command"
        if read_only and _MUTATING.search(cmd):
            return "REFUSED: this phase is read-only (grep/cat/ls/git diff/pytest only)"
        try:
            r = subprocess.run(cmd, shell=True, cwd=self.repo, capture_output=True, text=True, timeout=int(timeout))
        except subprocess.TimeoutExpired:
            return f"TIMEOUT after {timeout}s: {cmd}"
        out = f"exit={r.returncode}\n"
        if r.stdout:
            out += "--- stdout ---\n" + r.stdout
        if r.stderr:
            out += "\n--- stderr ---\n" + r.stderr
        return out

    def t_write_file(self, path: str, content: str) -> str:
        p = self.resolve(path)
        if self.in_run(p):
            return "REFUSED: the run output is read-only evidence"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} chars to {p.relative_to(self.repo)}"

    def t_str_replace(self, path: str, old: str, new: str) -> str:
        p = self.resolve(path)
        if self.in_run(p):
            return "REFUSED: the run output is read-only evidence"
        text = p.read_text()
        n = text.count(old)
        if n != 1:
            return f"REFUSED: old text matches {n} times (must be exactly 1). Widen or narrow it."
        p.write_text(text.replace(old, new, 1))
        return f"replaced 1 occurrence in {p.relative_to(self.repo)}"

    def t_write_note(self, name: str, content: str) -> str:
        if "/" in name or name.startswith("."):
            return "REFUSED: note name must be a bare filename like PLAN.md"
        (self.crew / name).write_text(content)
        return f"note saved: .crew/{name} ({len(content)} chars)"

    def t_write_backlog(self, items: list) -> str:
        errs, ids = [], set()
        for i, it in enumerate(items):
            for k in ("id", "title", "area", "priority", "evidence", "change", "acceptance", "files", "role"):
                if k not in it:
                    errs.append(f"item {i}: missing {k}")
            if it.get("role") not in ("ml", "fuzz", "swe"):
                errs.append(f"item {i}: role must be ml|fuzz|swe")
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,40}", str(it.get("id", ""))):
                errs.append(f"item {i}: id must be a short kebab-case slug")
            if it.get("id") in ids:
                errs.append(f"item {i}: duplicate id")
            ids.add(it.get("id"))
            if not isinstance(it.get("files"), list):
                errs.append(f"item {i}: files must be a list of repo paths")
        if errs:
            return "REFUSED:\n" + "\n".join(errs)
        items = sorted(items, key=lambda x: int(x["priority"]))
        (self.crew / "BACKLOG.json").write_text(json.dumps(items, indent=2))
        md = ["# BACKLOG", ""]
        for it in items:
            md += [f"## {it['priority']}. {it['id']} — {it['title']}  [{it['area']} / planner: {it['role']}]",
                   f"Files: {', '.join(it['files']) or '(unspecified)'}", "Evidence:"]
            md += [f"- {e}" for e in (it["evidence"] if isinstance(it["evidence"], list) else [it["evidence"]])]
            md += ["Change: " + it["change"], "Acceptance: " + it["acceptance"], ""]
        (self.crew / "BACKLOG.md").write_text("\n".join(md))
        return f"backlog saved: {len(items)} items -> .crew/BACKLOG.json"


def _schema(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


ITEM_PROPS = {"id": {"type": "string"}, "title": {"type": "string"},
              "area": {"type": "string", "enum": ["patching", "fuzzing", "observability", "dedup", "other"]},
              "priority": {"type": "integer"}, "evidence": {"type": "array", "items": {"type": "string"}},
              "change": {"type": "string"}, "acceptance": {"type": "string"},
              "files": {"type": "array", "items": {"type": "string"}},
              "role": {"type": "string", "enum": ["ml", "fuzz", "swe"]}}
TOOL_SCHEMAS = {
    "list_dir": _schema("list_dir", "List a directory (repo or run output).", {"path": {"type": "string"}}, []),
    "read_file": _schema("read_file", f"Read a file with line numbers, at most {READ_MAX_LINES} lines per call; page with start/end.",
                         {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path"]),
    "search": _schema("search", "Recursive grep -nE over a path (repo or run output). Use before reading; use it on big logs instead of reading them.",
                      {"pattern": {"type": "string"}, "path": {"type": "string"}, "max_hits": {"type": "integer"}}, ["pattern"]),
    "run": _schema("run", "Run a shell command in the workspace (py_compile, pytest, git diff, grep). Output is truncated; keep commands specific.",
                   {"cmd": {"type": "string"}, "timeout": {"type": "integer"}}, ["cmd"]),
    "write_file": _schema("write_file", "Create or overwrite a repo file (new tests/modules). Prefer str_replace for existing files.",
                          {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    "str_replace": _schema("str_replace", "Surgical edit: replace exactly one occurrence of old with new in a repo file.",
                           {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, ["path", "old", "new"]),
    "write_note": _schema("write_note", "Save a deliverable to .crew/<name> (PLAN.md, IMPLEMENT.md, REVIEW.md, HANDOFF.md).",
                          {"name": {"type": "string"}, "content": {"type": "string"}}, ["name", "content"]),
    "write_backlog": _schema("write_backlog", "Save the prioritized backlog (survey phase only). Each item is ONE behavioural change with evidence, acceptance test, files it will touch, and which planner role owns it.",
                             {"items": {"type": "array", "items": {"type": "object", "properties": ITEM_PROPS,
                                                                    "required": list(ITEM_PROPS)}}}, ["items"]),
    "done": _schema("done", "Finish this phase with a short summary.", {"summary": {"type": "string"}}, ["summary"]),
}
READ_TOOLS = ["list_dir", "read_file", "search", "run", "write_note", "done"]
SURVEY_TOOLS = READ_TOOLS + ["write_backlog"]
ALL_TOOLS = ["list_dir", "read_file", "search", "run", "write_file", "str_replace", "write_note", "done"]


def call_tool(ws: Workspace, name: str, args: dict, read_only: bool) -> str:
    if name == "done":
        return args.get("summary", "")
    if read_only and name in ("write_file", "str_replace"):
        return "REFUSED: this phase is read-only"
    fn = getattr(ws, f"t_{name}", None)
    if fn is None:
        return f"unknown tool {name}"
    try:
        if name == "run":
            return _clip(ws.t_run(args.get("cmd", ""), int(args.get("timeout", 600)), read_only=read_only))
        return _clip(fn(**{k: v for k, v in args.items() if v is not None}))
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e}"


# ----------------------------------------------------------------------------- role prompts
SENIOR_SYS = """You are the senior engineer leading a small crew fixing discver, a Java cyber-reasoning system (Jazzer fuzzing + LLM analysis + automated patching) that runs in a colleague's container you cannot execute. You work from the latest run output on disk and the repo.
Principles: evidence before hypothesis; each backlog item is ONE behavioural change; fail loud with a specific cause; no feature flags; never claim an outcome you did not run; distinguish "the fuzzer/patcher could not" from "our plumbing never let it try" — every failure so far has been the latter.
Tool discipline: search before read; never read more than 300 lines in one call; grep large logs, never read them whole. Save deliverables with write_note / write_backlog, then call done. When evidence is missing, write MISSING plus the exact command that would produce it — never invent a log line."""

ML_SYS = """You are the ML engineer. You own everything the runtime LLM sees and returns: prompt construction, the action schema, reply parsing, repair prompts, retries, token and timeout caps, and the per-turn accounting.
The task brief names the runtime model. Design for the weakest model that could be used (lenient parsing, explicit schema with a worked example, counted and logged failures, caps) — and if the brief says the runtime is a frontier model and the loop still never reaches test_patch, the defect is in discver's own protocol or plumbing; find that, don't blame the model.
You are read-only: you write PLAN.md, you do not edit code. Never read more than 300 lines per call. Call done when PLAN.md is saved."""

FUZZ_SYS = """You are the fuzzing engineer. You own the Jazzer/libFuzzer runner invocation, seed generation and wiring, harness generation and selection, corpus handling, and the run health/liveness checker.
You know: a harness SATURATED by a shallow crash (a few hundred execs, throws on nearly every input, fork restarts dominate) needs the fuzzer to keep going past known crashes (Jazzer keep-going with stack-trace dedup) and/or a corpus seeded past the crash; a harness whose coverage never moves from its first stats line is a no-op harness or an uninstrumented target and should be flagged and dropped, not fuzzed for millions of execs; an exec rate shown as 0.0/s next to millions of execs is a broken calculation; health verdicts are made more accurate, never softened. Everything auto-detects — no flags.
You are read-only: you write PLAN.md, you do not edit code. Never read more than 300 lines per call. Call done when PLAN.md is saved."""

SWE_SYS = """You are the software engineer. You implement PLAN.md exactly, with surgical str_replace edits (no rewrites, no reformatting of untouched code), add the tests it specifies under tests/, and prove the change yourself with the test command given in the prompt; iterate until green; add a CHANGELOG.md entry; then, ONLY if the prompt says this workspace is a git repo, git add -A && git commit -m "<area>: <change>" — otherwise never run git; the script snapshots your changes.
Rules: one behavioural change; no flags or env switches; fail loud with specific causes; never touch crash_dedup._get_crash_signature or the sidecar builder wiring; never modify anything under the run output; never claim a result you did not see in a tool output.
Finish by saving IMPLEMENT.md with the actual last lines of the test command and git log -1, then call done."""

ROLE_SYS = {"senior": SENIOR_SYS, "ml": ML_SYS, "fuzz": FUZZ_SYS, "swe": SWE_SYS}

DEFAULT_TASK = """# TASK — discver: decide the fixes from the run output and the code as they are NOW

discver is an in-house Java cyber-reasoning system: Jazzer/libFuzzer fuzzing + LLM analysis + automated patching, run under OSS-CRS in a colleague's container that you cannot execute. You have the repo (src/, tests/) and the latest real run's output (run output path). Nothing else about the code's current state is given to you on purpose: the code has changed a lot recently, so read the run output and the code and decide what to fix yourself.

The two outcomes that matter, in order:
(a) a `validated` patch is emitted for a PoV-backed crash — meaning: patch applied, target rebuilt through the builder sidecar, the specific PoV re-run and no longer crashing, existing tests passing;
(b) the primary target harness is reported HEALTHY by the run's own health checker.

How to decide:
- Evidence first. Everything you claim about a failure must point at a log line, a patch_index field, a health block, or a file:line.
- Establish what actually runs. Where several implementations of the same stage exist (patchers, parsers, seed generators, health checks), find which one the orchestrator calls and which one the log shows executing; never edit a path that does not run; dead or duplicate paths that confuse the live one are a legitimate backlog item.
- Distinguish "the fuzzer/patcher could not" from "our plumbing never let it try". Look for: iteration budgets exhausted with zero tool/test calls; replies dropped or counted silently; harnesses saturated by a shallow crash or with coverage that never moves; seeds not reaching runners; verdicts or reasons that hide the cause.
- Each backlog item is ONE behavioural change with an acceptance test that runs here (fakes, no network, no container), the files it touches, and what the next real run's log/patch_index/health block will show if it worked and if it didn't.

Invariants — the reviewer blocks any violation:
- No feature flags, no env switches; everything auto-detects and logs which path it chose at startup.
- Fail loud with a specific cause; no silent skip, no bare except, no continue on a failure without a log line carrying the raw data.
- `validated` means exactly the definition in (a); anything less is UNVALIDATED with a prominent label; never tests_pass=True when no test command ran.
- Never modify real target source; patches are emitted under output/patches/.
- Do not modify the PoV replay oracle (crash_dedup._get_crash_signature) or the builder-sidecar wiring.
- One behavioural change per item; observability and caps may ride along; health verdicts are made more accurate, never more lenient.
- Never claim an outcome you did not run and see in a tool output.
"""


def _paths(ws: Workspace) -> str:
    kind = "git checkout — commit your work" if ws.git_mode else "plain directory, NOT a git repo — never run git; the script snapshots and diffs your changes"
    return f"PATHS\nworkspace ({kind}): {ws.repo}\nrun output (read-only evidence): {ws.run or 'NONE PROVIDED'}\ntest command: {TEST_CMD}\n"


def survey_prompt(ws: Workspace, task: str, max_items: int) -> str:
    return f"""PHASE: survey

TASK BRIEF
{task}

{_paths(ws)}
DO, IN ORDER
1. list_dir the run output. Find the run report (report_run_*.md / REPORT.md), orchestrator.log, patches/patch_index.json, health.json, unique_bugs/. Search them for: "FINAL RESULTS", "run health", "\\[(DEAD|WEAK|HEALTHY)\\]", "SATURATED", "coverage never", "peak coverage 0", "LLM-generated seeds", "patch-turn", "llm-call", "timed out|timeout", "invalid JSON", "build oracle path|build_ok", "model". Read patch_index.json. Establish which runtime model the run used.
2. Survey the repo read-only: src/ layout; where the fuzz runner is launched (Jazzer args), where seeds are generated and wired, where harnesses are selected, the health checker, the patch loop (prompt builder, reply parser, parse-failure handling, iteration accounting, static-* handling), dedup. Cite file:line. Where more than one implementation of a stage exists, determine from the orchestrator's imports/calls and from the log which one actually ran; list the others as dead or duplicate.
3. Decide, on evidence, what to fix. write_backlog with 3 to {max_items} items, priority 1 = highest expected impact on the two outcomes that matter: (a) a validated patch emitted for a PoV-backed crash, (b) the primary target harness reported HEALTHY. Each item is ONE behavioural change with: evidence (log or code lines), the change, an acceptance test (unit test with fakes, no network), the files it will touch (be specific — items with disjoint files run in parallel), and the planner role (ml = LLM-facing protocol, fuzz = fuzzer/seeds/harness/health, swe = plain plumbing). Do not include an item the evidence shows is already fixed. Prefer several small items over one large one.
4. write_note SURVEY.md: what the run shows (one paragraph each for fuzzing and patching, with the exact lines), what is MISSING and the command to get it, and why the backlog is ordered as it is.
5. done.
"""


def plan_prompt(ws: Workspace, task: str, item: dict) -> str:
    return f"""PHASE: plan  ITEM: {item['id']}

TASK BRIEF (context)
{task}

BACKLOG ITEM (your scope — nothing outside it)
{json.dumps(item, indent=2)}

SURVEY.md (senior engineer)
{ws.note('SURVEY.md')}

{_paths(ws)}
Read the code the item touches (page with read_file; search first), then write PLAN.md with exactly these sections:
  EDITS — per edit: file, function, what the current lines do (quote 1-3 lines), what they become (sketch, not a rewrite). Surgical; no reformatting.
  BEHAVIOUR — the one behavioural change in two sentences, and what the next real run's log/patch_index/health block will show if it works and if it doesn't.
  LOGGING — the log line(s) added, with their exact format; every new failure path names its cause.
  TESTS — file name; for each test: setup (fakes), the input, the expected behaviour, the assertion. No network, no container.
  OUT OF SCOPE — things you noticed but must not touch here (they go in HANDOFF.md).
Then done.
"""


def implement_prompt(ws: Workspace, task: str, item: dict | None, feedback: str | None) -> str:
    fb = f"\n\nREVIEW FEEDBACK TO ADDRESS (previous attempt was BLOCKED)\n{feedback}\n" if feedback else ""
    scope = f"ITEM: {item['id']} — {item['title']}" if item else "single task"
    return f"""PHASE: implement  {scope}

TASK BRIEF (context; PLAN.md is your spec)
{task}

PLAN.md
{ws.note('PLAN.md')}{fb}

{_paths(ws)}
Implement PLAN.md. Before each edit, read the target region with read_file so your str_replace old-text is exact. Add the tests, run the test command, fix until green, add the CHANGELOG.md entry{', commit' if ws.git_mode else ' (do not run git)'}. Save IMPLEMENT.md with the actual outputs, then done.
"""


def review_prompt(ws: Workspace, task: str, base: str, item: dict | None) -> str:
    scope = f"ITEM: {item['id']} — {item['title']}" if item else "single task"
    return f"""PHASE: review  {scope}

TASK BRIEF (invariants are at the end)
{task}

PLAN.md
{ws.note('PLAN.md')}

IMPLEMENT.md (engineer's report)
{ws.note('IMPLEMENT.md')}

{_paths(ws)}
Review the change: the complete diff of this change is in .crew/DIFF.patch (changed files listed at the top) — read it with read_file, paging through all of it{f"; `git diff {base} --stat` and `git diff {base} -- <file>` are also available" if ws.git_mode else ""}. Then run the test command yourself.
BLOCK if any of: a feature flag or env switch was added; a silent path (bare except, continue/return on failure without a specific log line); a patch can be marked validated without PoV replay + tests; crash_dedup._get_crash_signature or the sidecar wiring was touched; the change goes beyond the plan's one behavioural change; no tests, or the suite is not green when YOU run it; an identical LLM request is resent after a timeout; anything under the run output was modified; the report claims something the diff does not show; a health verdict was made more lenient.
Write REVIEW.md: first line exactly `VERDICT: PASS` or `VERDICT: BLOCK`, then numbered findings with file:line.
If PASS, also write HANDOFF.md: HYPOTHESIS (one line), CHANGE (files, one paragraph), NEXT RUN WILL SHOW (if right / if wrong: the exact log lines, patch_index fields or health block), THE ONE GREP for the colleague to run, and OUT OF SCOPE items noticed. Then done.
"""


def merge_fix_prompt(ws: Workspace, task: str, item: dict, branch: str) -> str:
    return f"""PHASE: merge-fix  ITEM: {item['id']}

Branch `{branch}` (a reviewed, PASSed change: {item['title']}) conflicts with the current campaign branch in this workspace, which already contains other merged items.
Run `git merge --no-ff --no-edit {branch}`, resolve every conflict so that BOTH sides' intents survive (never drop either change silently), remove conflict markers, run the test command until green, then `git add -A && git commit --no-edit` (or `git commit -m "merge {branch}"`). Verify with `git status` (clean) and `git log -1`.
{_paths(ws)}
Save IMPLEMENT.md with the last lines of the test command and git log -1, then done.
"""


def triage_prompt(ws: Workspace, task: str) -> str:
    return f"""PHASE: triage (single mode)

TASK BRIEF
{task}

{_paths(ws)}
1. list_dir the run output; search it for "FINAL RESULTS", "patch-turn", "llm-call", "timed out|timeout", "seed", "invalid JSON", "build oracle path|build_ok"; read patches/patch_index.json if present.
2. Survey src/patch_generator.py read-only: prompt/brief builder and the JSON action example; the reply parser; parse-failure handling; what increments attempts; max iterations; static-* handling; what fills root_cause. Cite file:line.
3. write_note TRIAGE.md: VERDICT (one line + the proving line, or MISSING + command) / EVIDENCE (numbered, file:line) / THE ONE CHANGE / ACCEPTANCE TESTS / DO NOT TOUCH / ASK THE COLLEAGUE (one grep).
4. done.
"""


# ----------------------------------------------------------------------------- phase runner
_JSON_ACTION = re.compile(r"\{[^{}]*\"tool\"\s*:\s*\"([a-z_]+)\"[^{}]*\}", re.S)


def run_phase(ws: Workspace, phase: str, role: str, user: str, tools: list[str], expected_note: str | None) -> str:
    model = ROLE_MODEL[role]
    read_only = "write_file" not in tools
    schemas = [TOOL_SCHEMAS[t] for t in tools]
    messages = [{"role": "system", "content": ROLE_SYS[role]}, {"role": "user", "content": user}]
    tag = f"{ws.tag}:{phase}"
    transcript = ws.crew / f"transcript-{phase}-{now()}.jsonl"
    log(f"\n=== [{tag}] {role} on {model}, {'read-only' if read_only else 'read/write'}, max {MAX_TURNS} turns ===")
    empty = 0
    for turn in range(1, MAX_TURNS + 1):
        _compact(messages, tag)
        msg = chat(model, messages, schemas, tag=f"{ws.tag}:")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            if isinstance(tc["function"].get("arguments"), dict):
                tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
        assistant = {"role": "assistant", "content": content}
        if tool_calls:
            assistant["tool_calls"] = [{"id": tc["id"], "type": "function",
                                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                                       for tc in tool_calls]
        messages.append(assistant)
        with transcript.open("a") as f:
            f.write(json.dumps({"turn": turn, "assistant": assistant}) + "\n")

        if tool_calls:
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError as e:
                    args, result = {}, f"ERROR: tool arguments were not valid JSON ({e}). Resend the call."
                else:
                    result = call_tool(ws, name, args, read_only) if name in tools else f"tool {name} not available in this phase"
                brief = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in args.items() if k not in ("content", "new", "old", "items"))
                log(f"[{tag} t{turn:02d}] {name}({brief}) -> {len(result)} chars")
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                with transcript.open("a") as f:
                    f.write(json.dumps({"turn": turn, "tool": name, "args": args, "result": result[:2000]}) + "\n")
                if name == "done":
                    log(f"[{tag}] done: {result[:300]}")
                    return result
            continue

        m = _JSON_ACTION.search(content)  # fallback: JSON action typed as text
        if m:
            try:
                action = json.loads(m.group(0))
                name = action.get("tool")
                args = action.get("args", {k: v for k, v in action.items() if k != "tool"})
                result = call_tool(ws, name, args, read_only) if name in tools else f"tool {name} not available"
                log(f"[{tag} t{turn:02d}] (text-action) {name} -> {len(result)} chars")
                messages.append({"role": "user", "content": f"[result of {name}]\n{result}"})
                if name == "done":
                    return result
                continue
            except Exception as e:
                messages.append({"role": "user", "content": f"Could not execute that action ({e}). Use the tools."})
                continue
        if content.strip():
            if expected_note and not (ws.crew / expected_note).exists():
                (ws.crew / expected_note).write_text(content)
                log(f"[{tag} t{turn:02d}] plain-text reply saved as .crew/{expected_note}")
            else:
                log(f"[{tag} t{turn:02d}] plain-text reply ({len(content)} chars) — ending phase")
            return content
        empty += 1
        if empty >= 3:
            log(f"[{tag}] three empty replies — ending phase")
            return "EMPTY"
        messages.append({"role": "user", "content": "Your reply was empty. Continue: call a tool, or call done."})
    log(f"[{tag}] hit MAX_TURNS={MAX_TURNS}")
    return "MAX_TURNS"


def _compact(messages: list, tag: str) -> None:
    total = sum(len(json.dumps(m)) for m in messages)
    if total < CTX_LIMIT_CHARS:
        return
    idx = [i for i, m in enumerate(messages) if m.get("role") == "tool" and len(m.get("content", "")) > 500]
    for i in idx[:-8]:
        messages[i]["content"] = "[earlier tool output elided to save context — re-run the tool if needed]"
    log(f"      [{tag}] compacted transcript ({total} chars)")


# ----------------------------------------------------------------------------- build / test gate
def run_tests(ws: Workspace) -> tuple[bool, str]:
    try:
        r = subprocess.run(TEST_CMD, shell=True, cwd=ws.repo, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-8:])
    return r.returncode == 0, tail


def write_diff(ws: Workspace, base: str, snap: pathlib.Path | None) -> str:
    """Write .crew/DIFF.patch for the reviewer: git diff in git mode, snapshot diff otherwise."""
    if ws.git_mode:
        stat = git(ws.repo, "diff", base, "--stat").stdout
        body = git(ws.repo, "diff", base).stdout
        untracked = git(ws.repo, "ls-files", "--others", "--exclude-standard").stdout
        text = f"# git diff {base} --stat\n{stat}\n# untracked (uncommitted new files)\n{untracked}\n{body}"
    else:
        files = changed_files(snap, ws.repo, ws.run)
        text = "# changed files vs snapshot\n" + "\n".join(files) + "\n\n" + make_patch(snap, ws.repo, ws.run)
    (ws.crew / "DIFF.patch").write_text(text)
    return text


def verdict(ws: Workspace) -> str:
    return "PASS" if ws.note("REVIEW.md", "").lstrip().upper().startswith("VERDICT: PASS") else "BLOCK"


# ----------------------------------------------------------------------------- campaign
def work_item(main: Workspace, item: dict, task: str, base_ref: str) -> dict:
    """plan -> implement -> review (max 3 loops) in a dedicated worktree on branch crew/<id>."""
    branch = f"crew/{item['id']}"
    wt = main.repo.parent / f"{main.repo.name}-crew" / item["id"]
    git(main.repo, "worktree", "remove", "--force", str(wt))
    git(main.repo, "worktree", "prune")
    r = git(main.repo, "worktree", "add", "-B", branch, str(wt), base_ref)
    if r.returncode != 0:
        return {"item": item, "status": "ERROR", "detail": r.stderr.strip(), "branch": branch, "worktree": str(wt)}
    ws = Workspace(wt, main.run, tag=item["id"])
    (ws.crew / "SURVEY.md").write_text(main.note("SURVEY.md"))
    try:
        run_phase(ws, "plan", item["role"], plan_prompt(ws, task, item), READ_TOOLS, "PLAN.md")
        feedback = None
        for loop in range(1, 4):
            run_phase(ws, "implement", "swe", implement_prompt(ws, task, item, feedback), ALL_TOOLS, "IMPLEMENT.md")
            for stale in ("REVIEW.md", "HANDOFF.md"):
                (ws.crew / stale).unlink(missing_ok=True)
            write_diff(ws, base_ref, None)
            run_phase(ws, "review", "senior", review_prompt(ws, task, base_ref, item), READ_TOOLS, "REVIEW.md")
            if verdict(ws) == "PASS":
                ok, tail = run_tests(ws)
                if ok:
                    return {"item": item, "status": "PASS", "branch": branch, "worktree": str(wt), "handoff": ws.note("HANDOFF.md", "")}
                feedback = f"Reviewer passed it, but the test command fails when run by the script:\n{tail}"
                log(f"[{item['id']}] tests fail after PASS — sending back (loop {loop})")
            else:
                feedback = ws.note("REVIEW.md")
                log(f"[{item['id']}] review BLOCK (loop {loop}/3)")
        return {"item": item, "status": "BLOCK", "branch": branch, "worktree": str(wt), "detail": feedback or ""}
    except Exception as e:
        return {"item": item, "status": "ERROR", "branch": branch, "worktree": str(wt), "detail": f"{type(e).__name__}: {e}"}


def work_item_nogit(main: Workspace, item: dict, task: str) -> dict:
    """No-git mode: plan -> implement -> review in place, one item at a time; snapshot before, restore on failure."""
    snap = main.crew / "snap" / item["id"]
    n = snapshot(main.repo, main.run, snap)
    for stale in ("PLAN.md", "IMPLEMENT.md", "REVIEW.md", "HANDOFF.md", "DIFF.patch"):
        (main.crew / stale).unlink(missing_ok=True)
    main.tag = item["id"]
    log(f"[{item['id']}] snapshot of {n} files taken")
    res = {"item": item, "status": "ERROR", "branch": "(no git)", "worktree": str(snap), "detail": ""}
    try:
        run_phase(main, "plan", item["role"], plan_prompt(main, task, item), READ_TOOLS, "PLAN.md")
        feedback = None
        for loop in range(1, 4):
            run_phase(main, "implement", "swe", implement_prompt(main, task, item, feedback), ALL_TOOLS, "IMPLEMENT.md")
            for stale in ("REVIEW.md", "HANDOFF.md"):
                (main.crew / stale).unlink(missing_ok=True)
            write_diff(main, "snapshot", snap)
            run_phase(main, "review", "senior", review_prompt(main, task, "snapshot", item), READ_TOOLS, "REVIEW.md")
            if verdict(main) == "PASS":
                ok, tail = run_tests(main)
                if ok:
                    res.update(status="MERGED", handoff=main.note("HANDOFF.md", ""), detail="applied in place")
                    log(f"[{item['id']}] applied; suite green")
                    break
                feedback = f"Reviewer passed it, but the test command fails when run by the script:\n{tail}"
                log(f"[{item['id']}] tests fail after PASS — sending back (loop {loop})")
            else:
                feedback = main.note("REVIEW.md")
                log(f"[{item['id']}] review BLOCK (loop {loop}/3)")
        else:
            res.update(status="BLOCK", detail=feedback or "")
    except Exception as e:
        res.update(status="ERROR", detail=f"{type(e).__name__}: {e}")
    finally:
        main.tag = "main"
        keep = main.crew / "items" / item["id"]
        keep.mkdir(parents=True, exist_ok=True)
        for name in ("PLAN.md", "IMPLEMENT.md", "REVIEW.md", "HANDOFF.md", "DIFF.patch"):
            if (main.crew / name).exists():
                shutil.copy2(main.crew / name, keep / name)
        if res["status"] != "MERGED":
            restore(snap, main.repo, main.run)
            log(f"[{item['id']}] {res['status']} — workspace restored from snapshot; notes kept in .crew/items/{item['id']}/")
        shutil.rmtree(snap, ignore_errors=True)
        res["worktree"] = str(keep)
    return res


def integrate(main: Workspace, res: dict, task: str) -> bool:
    """Merge a PASSed item branch into the campaign branch; suite must stay green; else roll back."""
    item, branch = res["item"], res["branch"]
    pre = git(main.repo, "rev-parse", "HEAD").stdout.strip()
    r = git(main.repo, "merge", "--no-ff", "--no-edit", branch)
    if r.returncode != 0:
        conflicts = git(main.repo, "diff", "--name-only", "--diff-filter=U").stdout.split()
        log(f"[integrate {item['id']}] merge conflict in {conflicts} — merge-fix phase")
        git(main.repo, "merge", "--abort")
        try:
            run_phase(main, "merge-fix", "swe", merge_fix_prompt(main, task, item, branch), ALL_TOOLS, "IMPLEMENT.md")
        except Exception as e:
            log(f"[integrate {item['id']}] merge-fix phase failed: {e}")
        clean = git(main.repo, "status", "--short").stdout.strip() == ""
        advanced = git(main.repo, "rev-parse", "HEAD").stdout.strip() != pre
        if not (clean and advanced):
            git(main.repo, "merge", "--abort")
            git(main.repo, "reset", "--hard", pre)
            res["status"], res["detail"] = "FAILED_MERGE", f"conflicts in {conflicts} not resolved"
            return False
    ok, tail = run_tests(main)
    if not ok:
        git(main.repo, "reset", "--hard", pre)
        res["status"], res["detail"] = "FAILED_INTEGRATION", f"suite red after merge — rolled back:\n{tail}"
        log(f"[integrate {item['id']}] suite red after merge — rolled back")
        return False
    res["status"] = "MERGED"
    log(f"[integrate {item['id']}] merged; suite green")
    return True


def pick_wave(remaining: list, k: int) -> list:
    wave, used = [], set()
    for it in remaining:
        files = set(it.get("files") or ["*"])
        if (files & used) or ("*" in used) or ("*" in files and wave):
            continue
        wave.append(it)
        used |= files
        if len(wave) >= k:
            break
    return wave or remaining[:1]


def campaign(main: Workspace, task: str, base: str, max_items: int, parallel: int, skip_survey: bool) -> list:
    if not skip_survey or not (main.crew / "BACKLOG.json").exists():
        for stale in ("BACKLOG.json", "BACKLOG.md", "SURVEY.md"):
            (main.crew / stale).unlink(missing_ok=True)
        try:
            run_phase(main, "survey", "senior", survey_prompt(main, task, max_items), SURVEY_TOOLS, "SURVEY.md")
        except Exception as e:
            log(f"[campaign] survey phase failed: {e}")
    if not (main.crew / "BACKLOG.json").exists():
        log("[campaign] no BACKLOG.json written — stopping (read .crew/SURVEY.md and the survey transcript)")
        return []
    items = json.loads((main.crew / "BACKLOG.json").read_text())[:max_items]
    log("[campaign] backlog: " + ", ".join(f"{i['priority']}:{i['id']}({i['role']})" for i in items))
    results, remaining = [], list(items)
    if not main.git_mode:
        for it in items:
            results.append(work_item_nogit(main, it, task))
        return results
    while remaining:
        wave = pick_wave(remaining, parallel)
        remaining = [i for i in remaining if i not in wave]
        base_ref = git(main.repo, "rev-parse", "HEAD").stdout.strip()
        log(f"\n[campaign] wave: {[i['id'] for i in wave]} from {base_ref[:10]}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
            wave_results = list(ex.map(lambda it: work_item(main, it, task, base_ref), wave))
        for res in wave_results:
            if res["status"] == "PASS":
                try:
                    integrate(main, res, task)
                except Exception as e:
                    res["status"], res["detail"] = "ERROR", f"integrate: {type(e).__name__}: {e}"
                    git(main.repo, "merge", "--abort")
            else:
                log(f"[campaign] {res['item']['id']}: {res['status']} — {res.get('detail','')[:300]}")
            results.append(res)
            if res["status"] == "MERGED":
                git(main.repo, "worktree", "remove", "--force", res["worktree"])
    return results


def write_campaign_md(main: Workspace, results: list, base: str) -> None:
    lines = [f"# CAMPAIGN — {now()}", "", f"Base: {base}",
             f"Branch: {git(main.repo, 'rev-parse', '--abbrev-ref', 'HEAD').stdout.strip() if main.git_mode else '(no git — changes applied in place; baseline snapshot in .crew/baseline/)'}", "",
             "| # | item | area | status | branch |", "|---|---|---|---|---|"]
    for r in results:
        it = r["item"]
        lines.append(f"| {it['priority']} | {it['id']} — {it['title']} | {it['area']} | {r['status']} | {r['branch']} |")
    lines += ["", "## Not merged", ""]
    lines += [f"- {r['item']['id']}: {r['status']} — {r.get('detail','')[:500]} (see {r['worktree']})"
              for r in results if r["status"] != "MERGED"] or ["(none)"]
    lines += ["", "## Hand-off per merged item", ""]
    for r in results:
        if r["status"] == "MERGED":
            lines += [f"### {r['item']['id']} — {r['item']['title']}", "", r.get("handoff") or "(no HANDOFF.md written)", ""]
    lines += ["", "## Survey", "", main.note("SURVEY.md")]
    (main.crew / "CAMPAIGN.md").write_text("\n".join(lines))


def package(main: Workspace, base: str, handoff_name: str) -> None:
    if main.git_mode:
        names = git(main.repo, "diff", "--name-only", base).stdout.split()
    else:
        names = changed_files(main.crew / "baseline", main.repo, main.run)
    files = [f for f in names if re.match(r"^(src|tests)/.*\.py$", f) and (main.repo / f).exists()]
    if not files:
        log(f"[package] NO changed .py files under src/ or tests/ since {base} — nothing to send")
        return
    out = main.repo / "discver-patch-fix.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(main.repo / f, f)
        if (main.crew / handoff_name).exists():
            z.write(main.crew / handoff_name, "HANDOFF.md")
    log(f"[package] {out.name}: " + ", ".join(files) + " + HANDOFF.md")


# ----------------------------------------------------------------------------- main
def verify() -> None:
    log(f"[verify] base_url={BASE_URL or 'MISSING'} model={MODEL or 'MISSING'} tls_verify={VERIFY_TLS}")
    if not (BASE_URL and API_KEY and MODEL):
        sys.exit("[verify] set LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL")
    for role in ROLES:
        t0 = time.time()
        msg = chat(ROLE_MODEL[role], [{"role": "user", "content": "Reply with the single word: ready"}], None)
        log(f"[verify] {role:6s} {ROLE_MODEL[role]}: {time.time()-t0:.1f}s -> {str(msg.get('content'))[:40]!r}")
    log("[verify] OK")


def _git_local_excludes(repo: pathlib.Path, run: pathlib.Path | None) -> None:
    common = git(repo, "rev-parse", "--git-common-dir").stdout.strip()
    info = (repo / common).resolve() / "info" if common else repo / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    excl = info / "exclude"
    cur = excl.read_text() if excl.exists() else ""
    want = [".crew/", "discver-patch-fix.zip", "__pycache__/", "*.pyc", ".pytest_cache/"]
    if run and (run == repo or repo in run.parents):
        want.append(str(run.relative_to(repo)) + "/")
    add = [w for w in want if w not in cur]
    if add:
        excl.write_text(cur.rstrip("\n") + "\n" + "\n".join(add) + "\n")
    attrs = info / "attributes"
    if "CHANGELOG.md merge=union" not in (attrs.read_text() if attrs.exists() else ""):
        with attrs.open("a") as f:
            f.write("CHANGELOG.md merge=union\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="multi-role coding crew on LiteLLM")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run", help="run output dir (default: <repo>/discver if it exists) — read-only evidence")
    ap.add_argument("--task", help="task brief (markdown); default: built-in evidence-only brief")
    ap.add_argument("--mode", choices=["campaign", "single"], default="campaign")
    ap.add_argument("--max-items", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=1, help="items worked concurrently (only items with disjoint files)")
    ap.add_argument("--skip-survey", action="store_true", help="reuse .crew/BACKLOG.json from a previous run")
    ap.add_argument("--base", help="base commit for diff/zip (default: HEAD at start)")
    ap.add_argument("--branch", help="campaign branch (default: crew/campaign-<timestamp>)")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    repo = pathlib.Path(a.repo).resolve()
    if a.verify:
        verify()
        return
    if not (BASE_URL and API_KEY and MODEL):
        sys.exit("set LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL (and AGENT_TLS_VERIFY=false if needed)")
    task = pathlib.Path(a.task).read_text() if a.task else DEFAULT_TASK
    run = pathlib.Path(a.run).resolve() if a.run else (repo / "discver" if (repo / "discver").exists() else None)
    if run and run.is_file():
        run = run.parent
    top = git(repo, "rev-parse", "--show-toplevel").stdout.strip()
    git_mode = bool(top) and pathlib.Path(top).resolve() == repo
    if git_mode:
        _git_local_excludes(repo, run)
        branch = a.branch or (f"crew/campaign-{now()}" if a.mode == "campaign" else None)
        if branch:
            exists = git(repo, "rev-parse", "--verify", "-q", branch).returncode == 0
            git(repo, "checkout", *([] if exists else ["-b"]), branch)
        base = a.base or git(repo, "rev-parse", "HEAD").stdout.strip()
        where = f"base={base} branch={git(repo, 'rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()}"
        dirty = git(repo, "status", "--short").stdout.strip()
    else:
        base = "snapshot"
        dirty = ""
        if top:
            log(f"[crew] {repo} is inside the git repo {top} but is not its root — using no-git mode")
    main_ws = Workspace(repo, run, tag="main", git_mode=git_mode)
    (main_ws.crew / "TASK.md").write_text(task)
    if not git_mode:
        n = snapshot(repo, run, main_ws.crew / "baseline")
        where = f"no-git mode: baseline snapshot of {n} files in .crew/baseline/ (changes are applied in place, one item at a time)"
        if a.parallel > 1:
            log("[crew] --parallel needs git worktrees; running items sequentially")

    log(f"[crew] repo={repo}\n[crew] run={run}\n[crew] {where}")
    log("[crew] models: " + " ".join(f"{r}={ROLE_MODEL[r]}" for r in ROLES) + f"  tls_verify={VERIFY_TLS}")
    log(f"[crew] mode={a.mode} max_items={a.max_items} parallel={a.parallel} max_turns/phase={MAX_TURNS} test_cmd={TEST_CMD!r}")
    log(f"[crew] task brief: {a.task or 'built-in (evidence-only)'} -> .crew/TASK.md")
    if dirty:
        log(f"[crew] WARNING working tree is dirty (commit or stash first):\n{dirty}")

    if a.mode == "single":
        run_phase(main_ws, "triage", "senior", triage_prompt(main_ws, task), READ_TOOLS, "TRIAGE.md")
        (main_ws.crew / "SURVEY.md").write_text(main_ws.note("TRIAGE.md"))
        item = {"id": "single", "title": "the one change from TRIAGE.md", "area": "patching", "priority": 1,
                "evidence": [], "change": "see TRIAGE.md", "acceptance": "see TRIAGE.md", "files": [], "role": "ml"}
        run_phase(main_ws, "plan", "ml", plan_prompt(main_ws, task, item), READ_TOOLS, "PLAN.md")
        feedback = None
        for loop in range(1, 4):
            run_phase(main_ws, "implement", "swe", implement_prompt(main_ws, task, None, feedback), ALL_TOOLS, "IMPLEMENT.md")
            for stale in ("REVIEW.md", "HANDOFF.md"):
                (main_ws.crew / stale).unlink(missing_ok=True)
            run_phase(main_ws, "review", "senior", review_prompt(main_ws, task, base, None), READ_TOOLS, "REVIEW.md")
            if verdict(main_ws) == "PASS":
                log("[crew] review PASS")
                break
            feedback = main_ws.note("REVIEW.md")
            log(f"[crew] review BLOCK (loop {loop}/3)")
        package(main_ws, base, "HANDOFF.md")
    else:
        results = campaign(main_ws, task, base, a.max_items, a.parallel, a.skip_survey)
        write_campaign_md(main_ws, results, base)
        package(main_ws, base, "CAMPAIGN.md")
        merged = [r["item"]["id"] for r in results if r["status"] == "MERGED"]
        log(f"[crew] campaign finished: merged {merged}; details in .crew/CAMPAIGN.md")
    log("[crew] done")


if __name__ == "__main__":
    main()
