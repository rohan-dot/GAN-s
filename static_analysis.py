"""
static_analysis.py — Real static analysis tool integration.

Runs actual security tools (not LLM guessing) and produces structured
findings that feed into the LLM Scanner for prioritization.

Tools (used if available, gracefully skipped if not):
1. Flawfinder — C/C++ vulnerability scanner (pip install flawfinder)
2. Cppcheck — static analyzer for C/C++ (conda install cppcheck)
3. Semgrep — pattern-based security scanner (pip install semgrep)

The LLM Scanner gets these findings as EVIDENCE, not replacement.
"Flawfinder flagged CWE-120 at line 571. Is this exploitable?"
is far more useful than "find bugs in this function."
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("gemma-fuzzer.static")


@dataclass
class StaticFinding:
    """Unified finding from any static analysis tool."""
    tool: str           # "flawfinder", "cppcheck", "semgrep"
    file: str           # relative path
    line: int
    severity: str       # "low", "medium", "high", "critical"
    cwe: str            # "CWE-120" or ""
    function: str       # function name if known
    category: str       # "buffer", "format", "race", etc.
    message: str        # human-readable description
    context: str        # source code line


def run_all(src_dir: str) -> list[StaticFinding]:
    """Run all available static analysis tools and return unified findings."""
    findings = []

    findings.extend(_run_flawfinder(src_dir))
    findings.extend(_run_cppcheck(src_dir))

    # Sort by severity (high first) then by file/line
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings.sort(key=lambda f: (severity_order.get(f.severity, 9), f.file, f.line))

    logger.info("[static] Total: %d findings from %d tools.",
                len(findings),
                len(set(f.tool for f in findings)))

    return findings


def findings_to_context(findings: list[StaticFinding], max_findings: int = 20) -> str:
    """Format findings as context string for LLM prompts."""
    if not findings:
        return ""

    lines = [f"## Static Analysis Tool Findings ({len(findings)} total)\n"]
    for f in findings[:max_findings]:
        lines.append(
            f"[{f.tool}] {f.file}:{f.line} — {f.severity.upper()} "
            f"{f.cwe} {f.category}: {f.message}"
        )
        if f.context:
            lines.append(f"  Code: {f.context.strip()}")

    if len(findings) > max_findings:
        lines.append(f"\n... and {len(findings) - max_findings} more findings.")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# FLAWFINDER
# ══════════════════════════════════════════════════════════════════

def _run_flawfinder(src_dir: str) -> list[StaticFinding]:
    """Run flawfinder on C/C++ source code."""
    if not shutil.which("flawfinder"):
        logger.info("[static] flawfinder not found. Install: pip install flawfinder")
        return []

    logger.info("[static] Running flawfinder on %s...", src_dir)
    try:
        result = subprocess.run(
            ["flawfinder", "--csv", "--columns", src_dir],
            capture_output=True, text=True, timeout=120,
        )

        findings = []
        reader = csv.DictReader(io.StringIO(result.stdout))
        for row in reader:
            # Skip fuzzing infrastructure
            filepath = row.get("File", "")
            if any(s in filepath.lower() for s in [
                "aflplusplus", "honggfuzz", "libfuzzer", "qemu_mode", "/test"
            ]):
                continue

            try:
                rel_path = str(Path(filepath).relative_to(src_dir))
            except ValueError:
                rel_path = filepath

            level = int(row.get("Level", "1"))
            severity = "critical" if level >= 4 else "high" if level >= 3 else "medium" if level >= 2 else "low"

            cwes = row.get("CWEs", "")
            cwe = cwes.split("!")[0].strip() if cwes else ""

            findings.append(StaticFinding(
                tool="flawfinder",
                file=rel_path,
                line=int(row.get("Line", 0)),
                severity=severity,
                cwe=cwe,
                function=row.get("Name", ""),
                category=row.get("Category", ""),
                message=row.get("Warning", ""),
                context=row.get("Context", ""),
            ))

        logger.info("[static] flawfinder: %d findings.", len(findings))
        return findings

    except subprocess.TimeoutExpired:
        logger.warning("[static] flawfinder timed out.")
        return []
    except Exception as exc:
        logger.warning("[static] flawfinder failed: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════
# CPPCHECK
# ══════════════════════════════════════════════════════════════════

def _run_cppcheck(src_dir: str) -> list[StaticFinding]:
    """Run cppcheck on C/C++ source code."""
    if not shutil.which("cppcheck"):
        logger.info("[static] cppcheck not found. Install: conda install -c conda-forge cppcheck")
        return []

    logger.info("[static] Running cppcheck on %s...", src_dir)
    try:
        # Use template output for easy parsing
        result = subprocess.run(
            [
                "cppcheck",
                "--enable=warning,style,performance,portability",
                "--template={file}|||{line}|||{severity}|||{id}|||{message}",
                "--quiet",
                src_dir,
            ],
            capture_output=True, text=True, timeout=120,
        )

        findings = []
        for line in result.stderr.split("\n"):
            parts = line.strip().split("|||")
            if len(parts) != 5:
                continue

            filepath, line_no, severity, check_id, message = parts

            # Skip fuzzing infrastructure
            if any(s in filepath.lower() for s in [
                "aflplusplus", "honggfuzz", "libfuzzer", "qemu_mode", "/test"
            ]):
                continue

            try:
                rel_path = str(Path(filepath).relative_to(src_dir))
            except ValueError:
                rel_path = filepath

            # Map cppcheck severity to our scale
            sev_map = {"error": "high", "warning": "medium",
                       "style": "low", "performance": "low",
                       "portability": "low"}

            # Map common cppcheck IDs to CWEs
            cwe_map = {
                "nullPointer": "CWE-476", "bufferAccessOutOfBounds": "CWE-119",
                "arrayIndexOutOfBounds": "CWE-119", "memleak": "CWE-401",
                "doubleFree": "CWE-415", "useAfterFree": "CWE-416",
                "uninitvar": "CWE-457", "integerOverflow": "CWE-190",
            }

            findings.append(StaticFinding(
                tool="cppcheck",
                file=rel_path,
                line=int(line_no) if line_no.isdigit() else 0,
                severity=sev_map.get(severity, "medium"),
                cwe=cwe_map.get(check_id, ""),
                function="",
                category=check_id,
                message=message,
                context="",
            ))

        logger.info("[static] cppcheck: %d findings.", len(findings))
        return findings

    except subprocess.TimeoutExpired:
        logger.warning("[static] cppcheck timed out.")
        return []
    except Exception as exc:
        logger.warning("[static] cppcheck failed: %s", exc)
        return []
