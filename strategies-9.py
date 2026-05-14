"""
strategies.py — LLM strategy engine (v4).

Simplified: prescan + codebase map + cross-file audit + coverage seeds.
All project-agnostic. Works on any C/C++ codebase.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from llm_client import VLLMClient

logger = logging.getLogger("gemma-fuzzer.strategies")





@dataclass
class StrategyResult:
    strategy_name: str
    findings: int
    details: list
    elapsed: float


class StrategyOrchestrator:
    def __init__(self, llm, src_dir, build_dir, output_dir, harness_name):
        self.llm = llm
        self.src_dir = src_dir
        self.build_dir = build_dir
        self.output_dir = output_dir
        self.harness_name = harness_name
        self.crash_summaries = []
        self.crash_reports = []
        self.generated_harnesses = []
        self.round_number = 0
        self.results_log = []
        self.codebase_map = None
        self.risky_files = []

    def run_round(self, corpus_dir):
        self.round_number += 1
        results = []
        logger.info("═══ Strategy Round %d ═══", self.round_number)

        if self.round_number == 1:
            results.append(self._run("prescan", self._prescan))
            results.append(self._run("codebase_map", self._codebase_map))

        if self.round_number <= 3 or self.round_number % 3 == 0:
            results.append(self._run("cross_file_audit",
                                     lambda: self._cross_file_audit()))

        results.append(self._run("coverage_seeds",
                                 lambda: self._coverage_seeds(corpus_dir)))

        self.results_log.extend(results)
        total = sum(r.findings for r in results)
        logger.info("═══ Round %d: %d findings ═══", self.round_number, total)
        return results

    def add_crash(self, summary, report):
        self.crash_summaries.append(summary)
        if report:
            self.crash_reports.append(report)

    def get_generated_harnesses(self):
        return list(self.generated_harnesses)

    # ── Prescan (fast regex, no LLM) ─────────────────────────────

    def _prescan(self):
        PATTERNS = [
            (r'\bmemcpy\s*\(', "memcpy", 3), (r'\bstrcpy\s*\(', "strcpy", 4),
            (r'\bsprintf\s*\(', "sprintf", 4), (r'\bmalloc\s*\(', "malloc", 2),
            (r'\brealloc\s*\(', "realloc", 3), (r'\bfree\s*\(', "free", 2),
            (r'\bstrcat\s*\(', "strcat", 4), (r'\batoi\s*\(', "atoi", 3),
            (r'\bsscanf\s*\(', "sscanf", 3), (r'\bgets\s*\(', "gets", 5),
        ]

        src_path = Path(self.src_dir)
        file_risks = []

        for f in src_path.rglob("*.c"):
            if any(s in str(f).lower() for s in [
                "test", ".git", "example", "python", "CMakeFiles", "aflplusplus", "afl-", "honggfuzz", "libqasan", "qemu_mode", "libfuzzer", "centipede"
            ]):
                continue
            try:
                content = f.read_text(errors="replace")
            except Exception:
                continue

            risk = 0
            found = []
            for pattern, name, weight in PATTERNS:
                count = len(re.findall(pattern, content))
                if count > 0:
                    risk += count * weight
                    found.append(f"{name}({count})")

            if risk > 0:
                rel = str(f.relative_to(src_path))
                file_risks.append((rel, content, risk, found))

        file_risks.sort(key=lambda x: -x[2])
        self.risky_files = [(n, c, s) for n, c, s, _ in file_risks[:15]]

        results = []
        for n, _, s, pats in file_risks[:10]:
            logger.info("[prescan] %s — risk=%d [%s]", n, s, ", ".join(pats[:5]))
            results.append({"file": n, "risk": s})

        logger.info("[prescan] Scanned %d files, %d with patterns.",
                   sum(1 for _ in src_path.rglob("*.c")), len(file_risks))
        return results

    # ── Codebase map (LLM identifies audit targets) ──────────────

    def _codebase_map(self):
        if not self.llm.is_available() or not self.risky_files:
            return []

        summary = "Source files ranked by dangerous-pattern density:\n\n"
        for name, content, score in self.risky_files[:10]:
            funcs = re.findall(r'\b([a-zA-Z_]\w+)\s*\([^)]*\)\s*\{', content)
            funcs = [f for f in funcs if f not in {"if", "for", "while", "switch"}]
            summary += f"## {name} (risk={score})\nFunctions: {', '.join(funcs[:15])}\n\n"

        _default_prompt = """\
Identify the TOP 5 most likely vulnerable functions. For each specify
file, function, likely bug type, and how external input reaches it.
Respond ONLY with a JSON array. Start with [ end with ].
[{"file":"f.c","function":"func","bug_type":"overflow",
  "data_flow":"input→parse→func(buf,controlled_size)",
  "audit_priority":"critical"}]"""

        response = self.llm.chat(
            system=_default_prompt,
            user=summary, max_tokens=2000, temperature=0.2,
        )
        if not response:
            return []

        targets = _parse_json_array(response)
        self.codebase_map = {"targets": targets}
        for t in targets:
            logger.info("[codebase-map] TARGET: %s:%s — %s",
                       t.get("file", "?"), t.get("function", "?"),
                       t.get("bug_type", "?"))

        (Path(self.output_dir) / "codebase_map.json").write_text(
            json.dumps(targets, indent=2))
        return targets

    # ── Cross-file audit ─────────────────────────────────────────

    def _cross_file_audit(self):
        if not self.llm.is_available():
            return []

        src_path = Path(self.src_dir)
        bugs_dir = Path(self.output_dir) / "bugs"
        bugs_dir.mkdir(parents=True, exist_ok=True)

        targets = (self.codebase_map or {}).get("targets", [])
        if not targets:
            targets = [{"file": n} for n, _, _ in self.risky_files[:5]]

        all_findings = []
        for target in targets[:5]:
            fname = target.get("file", "")
            matches = list(src_path.rglob(Path(fname).name)) if fname else []
            if not matches:
                continue

            try:
                content = matches[0].read_text(errors="replace")
                rel = str(matches[0].relative_to(src_path))
            except Exception:
                continue

            user_msg = f"File: {rel}\n"
            if target.get("function"):
                user_msg += f"Focus on: {target['function']}\n"
            if target.get("bug_type"):
                user_msg += f"Suspected: {target['bug_type']}\n"
            user_msg += f"\n```c\n{content[:8000]}\n```"

            logger.info("[cross-audit] Auditing %s", rel)
            response = self.llm.chat(
                system="""\
Analyze this C source code for security vulnerabilities. Be specific about
the exact operation that's vulnerable and what input triggers it.
Respond ONLY with a JSON array. Start with [ end with ].
[{"function":"f","file":"f.c","bug_type":"overflow",
  "description":"specific explanation","severity":"high"}]""",
                user=user_msg, max_tokens=2000, temperature=0.2,
            )
            if not response:
                continue

            findings = _parse_json_array(response)
            for f in findings:
                f["strategy"] = "cross_file_audit"
                f["timestamp"] = time.time()
                fh = hashlib.sha256(
                    json.dumps(f, sort_keys=True).encode()).hexdigest()[:12]
                (bugs_dir / f"xaudit-{fh}.json").write_text(json.dumps(f, indent=2))
                logger.info("[cross-audit] FINDING: %s in %s",
                           f.get("bug_type", "?"), f.get("function", "?"))
                all_findings.append(f)

        return all_findings

    # ── Coverage seeds ───────────────────────────────────────────

    def _coverage_seeds(self, corpus_dir):
        if not self.llm.is_available():
            return 0

        seeds_dir = Path(self.output_dir) / "seeds"
        scripts_dir = Path(self.output_dir) / "seed_scripts"
        seeds_dir.mkdir(parents=True, exist_ok=True)
        scripts_dir.mkdir(parents=True, exist_ok=True)

        source_files = _collect_source_files(Path(self.src_dir), self.harness_name)

        user_msg = f"Target: {self.harness_name}\n"
        if self.crash_summaries:
            user_msg += "Crashes found (target DIFFERENT paths):\n"
            for s in self.crash_summaries[:5]:
                user_msg += f"- {s}\n"
        if source_files:
            user_msg += "\nSource (understand the input format):\n"
            for sf, content in source_files[:2]:
                user_msg += f"\n// {sf}\n{content[:3000]}\n"

        response = self.llm.chat(
            system="""\
Generate a Python script that creates a test input file for fuzzing.
The script must write the input to "/tmp/poc_input".
Use only Python standard library.
Think about what input format the target expects based on the source code.
Generate inputs that exercise edge cases and error-handling paths.
Output ONLY the Python script. No markdown. Start with a comment.""",
            user=user_msg, max_tokens=2000, temperature=0.6,
        )
        if not response:
            return 0

        script = _clean_code(response)
        sh = hashlib.sha256(script.encode()).hexdigest()[:8]
        script_path = scripts_dir / f"seed_{sh}.py"
        script_path.write_text(script)

        data = _run_script(str(script_path))
        if data is None:
            return 0

        seed_path = seeds_dir / f"seed-{hashlib.sha256(data).hexdigest()[:12]}"
        seed_path.write_bytes(data)
        logger.info("[cov-seed] Generated %d byte seed.", len(data))
        return 1

    # ── Runner ───────────────────────────────────────────────────

    def _run(self, name, func):
        logger.info("[strategy] Running: %s", name)
        t0 = time.monotonic()
        try:
            result = func()
            findings = result if isinstance(result, int) else len(result) if isinstance(result, list) else 0
            details = result if isinstance(result, list) else []
        except Exception as exc:
            logger.error("[strategy] %s failed: %s", name, exc, exc_info=True)
            findings, details = 0, []
        elapsed = time.monotonic() - t0
        logger.info("[strategy] %s → %d findings (%.1fs)", name, findings, elapsed)
        return StrategyResult(name, findings, details, elapsed)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _collect_source_files(src_path, harness_name, max_files=8):
    files = []
    for f in src_path.rglob("*.c"):
        if any(s in str(f).lower() for s in ["test", ".git", "example", "python", "CMakeFiles", "aflplusplus", "afl-", "honggfuzz", "libqasan", "qemu_mode", "libfuzzer", "centipede"]):
            continue
        try:
            content = f.read_text(errors="replace")
            priority = 10
            fname = f.name.lower()
            hname = (harness_name or "").lower()
            if hname and hname in fname: priority += 20
            for kw in ["parse", "read", "dict", "buf", "string", "encoding",
                        "decode", "header", "cert", "key", "alloc", "memory"]:
                if kw in fname:
                    priority += 15
                    break
            for kw in ["memcpy", "malloc", "strcpy", "realloc", "sprintf"]:
                if kw in content.lower(): priority += 3
            files.append((str(f.relative_to(src_path)), content, priority))
        except Exception:
            continue
    files.sort(key=lambda x: -x[2])
    return [(n, c) for n, c, _ in files[:max_files]]


def _clean_code(response):
    code = response.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1] if "\n" in code else code[3:]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


def _parse_json_array(response):
    clean = response.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()
    s, e = clean.find("["), clean.rfind("]")
    if s != -1 and e != -1 and e > s:
        clean = clean[s:e + 1]
    else:
        return []
    try:
        result = json.loads(clean)
        return [i for i in result if isinstance(i, dict)] if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def _run_script(script_path):
    try:
        os.unlink("/tmp/poc_input")
    except FileNotFoundError:
        pass
    try:
        result = subprocess.run(["python3", script_path],
                                capture_output=True, timeout=30, text=True)
        if result.returncode != 0:
            logger.warning("[script] Error: %s", result.stderr[:200])
            return None
        if not os.path.exists("/tmp/poc_input"):
            return None
        data = Path("/tmp/poc_input").read_bytes()
        return data if data else None
    except Exception:
        return None
