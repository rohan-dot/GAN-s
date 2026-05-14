"""
agents.py — Parallel multi-agent vulnerability discovery (v4).

Modeled after Buttercup (Trail of Bits, AIxCC 2nd place):
- Tree-sitter AST for code understanding
- Parallel agent execution via ThreadPoolExecutor
- Reflexion loops with structured feedback at every level
- Fully project-agnostic (reads actual project headers/types)
- Direct harness generation (#include .c for internal functions)

Pipeline: Scanner → Exploiter → Verifier (parallel where possible)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from code_analysis import (
    CallGraph, FunctionInfo, ParamInfo,
    get_file_includes, find_include_dirs, find_static_lib,
)
from llm_client import VLLMClient
from triage import skeptical_triage, generate_context

logger = logging.getLogger("gemma-fuzzer.agents")



@dataclass
class ScanFinding:
    file: str
    function: str
    bug_type: str
    confidence: float
    description: str
    data_flow: str
    trigger_hint: str
    params: list[ParamInfo] = None  # from tree-sitter


@dataclass
class ExploitResult:
    finding: ScanFinding
    crashed: bool
    poc_path: str | None
    crash_output: str
    attempts: int
    harness_path: str | None = None


# ══════════════════════════════════════════════════════════════════
# COMPILATION & FUZZING HELPERS
# ══════════════════════════════════════════════════════════════════

def compile_harness(code: str, name: str, src_dir: str, output_dir: str) -> tuple[str | None, str]:
    """Compile a C harness with auto-retry for missing macros.
    
    When internal headers use project-specific macros (XML_HIDDEN, 
    OPENSSL_EXPORT, CURL_EXTERN, etc.), the first compile fails.
    We parse the error, define those macros as empty, and retry.
    This works for ANY project without hardcoding macro names.
    """
    harness_dir = Path(output_dir) / "generated_harnesses"
    harness_dir.mkdir(parents=True, exist_ok=True)

    h = hashlib.sha256(code.encode()).hexdigest()[:8]
    src_path = harness_dir / f"{name}_{h}.c"
    bin_path = harness_dir / f"{name}_{h}"
    src_path.write_text(code)

    inc_dirs = find_include_dirs(src_dir)
    lib = find_static_lib(src_dir)

    def _build_cmd(extra_defines: list[str] = None, extra_libs: list[str] = None) -> list[str]:
        cmd = ["clang", "-g", "-O1", "-fsanitize=address,fuzzer"]

        # Auto-include config.h if it exists (autotools/cmake projects)
        config_h = Path(src_dir) / "config.h"
        if config_h.exists():
            cmd.extend(["-DHAVE_CONFIG_H", "-include", str(config_h)])

        # Extra defines (from error parsing)
        if extra_defines:
            cmd.extend(extra_defines)

        for d in inc_dirs:
            cmd.extend(["-I", d])
        cmd.append(str(src_path))
        if lib:
            cmd.append(lib)
        # Only add libraries that are actually needed
        cmd.extend(["-lm", "-Wl,--allow-multiple-definition"])
        if extra_libs:
            cmd.extend(extra_libs)
        cmd.extend(["-o", str(bin_path)])
        return cmd

    # Attempt 1: compile with minimal flags (no -lz -llzma)
    try:
        cmd = _build_cmd()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("[compile] OK: %s", bin_path.name)
            return str(bin_path), ""

        stderr = result.stderr

        # If linker can't find symbols, try adding common libraries
        if "undefined reference" in stderr or "cannot find" in stderr:
            extra_libs = []
            if "lzma" in stderr or "LZMA" in stderr:
                extra_libs.append("-llzma")
            if "compress" in stderr or "inflate" in stderr or "zlibVersion" in stderr:
                extra_libs.append("-lz")
            if "pthread" in stderr:
                extra_libs.append("-lpthread")
            if extra_libs:
                logger.info("[compile] Retrying with libs: %s", " ".join(extra_libs))
                cmd = _build_cmd(extra_libs=extra_libs)
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    logger.info("[compile] OK (with extra libs): %s", bin_path.name)
                    return str(bin_path), ""
                stderr = result.stderr

        # Attempt 2: parse errors for unknown type names / undeclared identifiers
        # and define them as empty macros
        stderr = result.stderr
        unknown_macros = set()
        for pattern in [
            r"unknown type name '(\w+)'",
            r"undeclared identifier '(\w+)'",
            r"use of undeclared identifier '(\w+)'",
        ]:
            for match in re.finditer(pattern, stderr):
                macro = match.group(1)
                # Only auto-define things that look like macros (ALL_CAPS or
                # mixed with underscores, typically visibility/export attributes)
                if (macro.isupper() or 
                    (any(c.isupper() for c in macro) and '_' in macro) or
                    macro.endswith("_HIDDEN") or
                    macro.endswith("_EXPORT") or
                    macro.endswith("_EXTERN") or
                    macro.endswith("_PUBLIC") or
                    macro.endswith("_API") or
                    macro.startswith("__")):
                    unknown_macros.add(macro)

        if unknown_macros:
            logger.info("[compile] Retrying with auto-defines: %s",
                       ", ".join(unknown_macros))
            extra = [f"-D{m}=" for m in unknown_macros]
            cmd = _build_cmd(extra_defines=extra)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info("[compile] OK (with auto-defines): %s", bin_path.name)
                return str(bin_path), ""

        return None, result.stderr

    except Exception as exc:
        return None, str(exc)


def run_libfuzzer(binary: str, output_dir: str, timeout: int) -> tuple[bool, str | None, str]:
    """Run LibFuzzer. Returns (crashed, poc_path, output_summary)."""
    crash_dir = Path(output_dir) / "exploit_crashes"
    corpus_dir = Path(output_dir) / "exploit_corpus"
    crash_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        binary, str(corpus_dir),
        f"-artifact_prefix={crash_dir}/",
        f"-max_total_time={timeout}",
        "-max_len=65536", "-detect_leaks=0", "-timeout=10",
    ]

    env = os.environ.copy()
    env["ASAN_OPTIONS"] = "abort_on_error=1:symbolize=1:detect_leaks=0"

    output = ""
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout + 30, env=env)
        output = (result.stdout or b"").decode("utf-8", errors="replace")
        output += (result.stderr or b"").decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        output = "timeout (expected)"

    # Check for crash files
    pov_dir = Path(output_dir) / "povs"
    pov_dir.mkdir(parents=True, exist_ok=True)
    crashes = sorted(crash_dir.glob("crash-*"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if crashes:
        dest = pov_dir / f"exploit-{crashes[0].name}"
        shutil.copy2(crashes[0], dest)
        return True, str(dest), output[-2000:]

    return False, None, _summarize_fuzzer_output(output)


def _summarize_fuzzer_output(output: str) -> str:
    """Extract useful feedback from LibFuzzer output."""
    lines = output.split("\n")
    summary_parts = []
    for line in lines[-50:]:
        if any(kw in line for kw in ["execs_per_sec", "cov:", "NEW", "REDUCE",
                                      "Total execs", "BINGO", "ERROR"]):
            summary_parts.append(line.strip())
    return "\n".join(summary_parts[-10:]) if summary_parts else "no useful output"


# ══════════════════════════════════════════════════════════════════
# AGENT 1: SCANNER (parallel)
# ══════════════════════════════════════════════════════════════════

SCANNER_PROMPT = """\
You are a security vulnerability scanner analyzing C/C++ source code.
Identify specific code locations that contain vulnerabilities.

You receive a function's source code, its parameter types, and call graph context.

For each vulnerability, output:
- file, function, bug_type, confidence (0.0-1.0)
- description: the EXACT operation that's vulnerable (cite the line)
- data_flow: how external input reaches this point
- trigger_hint: what parameter values trigger the bug

ONLY report findings with confidence >= 0.6.

Respond ONLY with a JSON array. Start with [ end with ].
[{"file":"f.c","function":"func","bug_type":"integer-overflow",
  "confidence":0.85,"description":"plen+nlen+2 overflows when both near UINT_MAX",
  "data_flow":"parse_input→process→func(buf,controlled_plen,controlled_nlen)",
  "trigger_hint":"plen=0xFFFFFFF0 and nlen=0x20 causes sum to wrap to small value"}]"""


class ScannerAgent:
    def __init__(self, llm: VLLMClient, max_workers: int = 3):
        self.llm = llm
        self.max_workers = max_workers

    def scan_top_targets(self, call_graph, src_dir, max_targets=8, static_context=""):
        """Scan the most dangerous functions, in parallel."""
        if not self.llm.is_available():
            return []

        dangerous = call_graph.find_dangerous_functions(min_score=10)
        targets = dangerous[:max_targets]

        logger.info("[scanner] Scanning %d targets in parallel...", len(targets))

        all_findings = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._scan_one, name, call_graph, src_dir, static_context): name
                for name, score in targets
            }
            for future in as_completed(futures):
                fname = futures[future]
                try:
                    findings = future.result()
                    for f in findings:
                        logger.info("[scanner] FINDING: %s in %s (%.2f) — %s",
                                   f.bug_type, f.function, f.confidence,
                                   f.description[:60])
                    all_findings.extend(findings)
                except Exception as exc:
                    logger.error("[scanner] Error scanning %s: %s", fname, exc)

        all_findings.sort(key=lambda f: -f.confidence)
        return all_findings

    def _scan_one(self, func_name, call_graph, src_dir, static_context=""):
        """Scan a single function with security context + static tool evidence."""
        context = call_graph.get_function_context(func_name, src_dir)
        if not context:
            return []

        fdef = call_graph.functions.get(func_name)
        callers = call_graph.get_callers(func_name, depth=3)

        # Generate security briefing for this file (cached per-file)
        file_briefing = ""
        if fdef:
            try:
                file_content = (Path(src_dir) / fdef.file).read_text(errors="replace")
                file_briefing = generate_context(
                    self.llm, fdef.file, file_content, src_dir,
                )
            except Exception:
                pass

        user_msg = f"Function: {func_name}\n"
        if fdef:
            user_msg += f"File: {fdef.file}:{fdef.line}\n"
            user_msg += f"Signature: {fdef.signature}\n"
            user_msg += f"Static: {fdef.is_static}\n"
            user_msg += f"Called by: {', '.join(fdef.called_by[:5])}\n"
            user_msg += f"Calls: {', '.join(fdef.calls[:5])}\n"

        # Add security context briefing (from AISLE-style context gen)
        if file_briefing:
            user_msg += f"\n\nSecurity context for this file:\n{file_briefing}\n"

        if callers:
            user_msg += "\nCall chains from entry points:\n"
            for path in callers[:3]:
                user_msg += f"  {' → '.join(path)}\n"

        # Add static analysis tool findings as evidence
        if static_context:
            # Filter to findings relevant to this function's file
            relevant = []
            for line in static_context.split("\n"):
                if fdef and fdef.file in line:
                    relevant.append(line)
                elif func_name in line:
                    relevant.append(line)
            if relevant:
                user_msg += "\n\nStatic analysis tool findings for this file:\n"
                user_msg += "\n".join(relevant[:10])
                user_msg += "\n\nEvaluate whether these tool findings represent real vulnerabilities.\n"

        user_msg += f"\n```c\n{context[:6000]}\n```"

        response = self.llm.chat(
            system=SCANNER_PROMPT, user=user_msg,
            max_tokens=1500, temperature=0.1,
        )
        if not response:
            return []

        # Parse and enrich with tree-sitter parameter info
        findings = []
        for f in _parse_json_array(response):
            if f.get("confidence", 0) < 0.6:
                continue
            finding = ScanFinding(
                file=f.get("file", fdef.file if fdef else ""),
                function=f.get("function", func_name),
                bug_type=f.get("bug_type", "unknown"),
                confidence=f.get("confidence", 0.6),
                description=f.get("description", ""),
                data_flow=f.get("data_flow", ""),
                trigger_hint=f.get("trigger_hint", ""),
                params=fdef.params if fdef else None,
            )
            findings.append(finding)
        return findings


# ══════════════════════════════════════════════════════════════════
# AGENT 2: EXPLOITER (reflexion loop)
# ══════════════════════════════════════════════════════════════════

HARNESS_PROMPT = """\
Write a LibFuzzer harness in C that calls `{func_name}` DIRECTLY to
trigger a {bug_type}.

The function is in file `{src_file}`. Access it with:
    #include "{src_file}"

That file uses these includes (use the same ones before #include "{src_file}"):
{file_includes}

Function signature: {signature}

RULES:
1. Include <stdint.h>, <stddef.h>, <string.h> first
2. Then the file's own includes (listed above)
3. Then #include "{src_file}"
4. Implement: int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
5. For integer parameters (int, unsigned int, size_t): read via memcpy from
   fuzz data. Do NOT clamp or limit them — let overflow happen.
6. For pointer/buffer parameters: point into the fuzz data buffer
7. For struct/context parameters: create using the library's own init functions
   (look at the source code to find the right constructor)
8. Clean up and return 0

Vulnerability: {description}
Trigger: {trigger_hint}

Source code:
```c
{source_code}
```

Output ONLY C code. No markdown fences. Start with #include."""

HARNESS_FIX_PROMPT = """\
Fix this harness that failed to compile.

Code:
```c
{code}
```

Error:
```
{error}
```

The file {src_file} uses these includes:
{file_includes}

Common fixes:
- Add missing #include for undefined types
- Don't redefine types already in the includes
- Make sure #include "{src_file}" comes AFTER the library includes
- Check parameter types match the function signature: {signature}

Output ONLY fixed C code. No markdown. Start with #include."""

HARNESS_NOCASH_PROMPT = """\
The harness compiled and ran but found NO crashes in {seconds}s ({execs} executions).

LibFuzzer output:
{fuzzer_output}

The function signature is: {signature}
The bug type is: {bug_type}
Trigger hint: {trigger_hint}

Possible issues:
- The function setup might be wrong (wrong init, missing state)
- The fuzz input might not be reaching the vulnerable parameters
- Integer parameters might need specific value ranges

Write a DIFFERENT harness with a different approach. Maybe:
- Initialize the context differently
- Map fuzz bytes to parameters in a different order
- Use a different entry point that still reaches the vulnerable function

Output ONLY C code. No markdown. Start with #include."""


class ExploiterAgent:
    def __init__(self, llm, src_dir, output_dir, fuzz_seconds=120):
        self.llm = llm
        self.src_dir = src_dir
        self.output_dir = output_dir
        self.fuzz_seconds = fuzz_seconds

    def exploit(self, finding, call_graph) -> ExploitResult:
        """Full reflexion loop: generate → compile → fuzz → reflect → retry."""
        if not self.llm.is_available():
            return ExploitResult(finding=finding, crashed=False,
                                poc_path=None, crash_output="", attempts=0)

        fdef = call_graph.functions.get(finding.function)
        context = call_graph.get_function_context(finding.function, self.src_dir)
        file_includes = get_file_includes(self.src_dir, finding.file)
        includes_str = "\n".join(file_includes) if file_includes else "// (check source for includes)"
        signature = fdef.signature if fdef else finding.function

        code = None
        compile_error = ""
        last_fuzzer_output = ""

        # ── Up to 4 attempts with different feedback each time ──
        for attempt in range(4):
            logger.info("[exploiter] Attempt %d/4 for %s (%s)",
                       attempt + 1, finding.function, finding.bug_type)

            if attempt == 0:
                # First attempt: generate from description
                response = self.llm.chat(
                    system="You are a security fuzzing engineer. Output ONLY C code.",
                    user=HARNESS_PROMPT.format(
                        func_name=finding.function,
                        bug_type=finding.bug_type,
                        src_file=finding.file,
                        file_includes=includes_str,
                        signature=signature,
                        description=finding.description,
                        trigger_hint=finding.trigger_hint,
                        source_code=context[:5000],
                    ),
                    max_tokens=2000, temperature=0.2,
                )
            elif compile_error:
                # Attempt 2-3: fix compilation error
                response = self.llm.chat(
                    system="You are a C programmer. Fix the error. Output ONLY C code.",
                    user=HARNESS_FIX_PROMPT.format(
                        code=code, error=compile_error[:800],
                        src_file=finding.file,
                        file_includes=includes_str,
                        signature=signature,
                    ),
                    max_tokens=2000, temperature=0.1 + attempt * 0.1,
                )
            elif last_fuzzer_output:
                # Compiled but no crash: try different approach
                response = self.llm.chat(
                    system="You are a security fuzzing engineer. Output ONLY C code.",
                    user=HARNESS_NOCASH_PROMPT.format(
                        seconds=self.fuzz_seconds,
                        execs=_extract_exec_count(last_fuzzer_output),
                        fuzzer_output=last_fuzzer_output[:500],
                        signature=signature,
                        bug_type=finding.bug_type,
                        trigger_hint=finding.trigger_hint,
                    ),
                    max_tokens=2000, temperature=0.3 + attempt * 0.1,
                )
            else:
                break

            if not response:
                continue

            code = _clean_code(response)

            # Compile
            bin_path, compile_error = compile_harness(
                code, f"exploit_{finding.function}", self.src_dir, self.output_dir,
            )

            if not bin_path:
                logger.warning("[exploiter] Compile fail (attempt %d): %s",
                              attempt + 1, compile_error[:150])
                continue

            # Run LibFuzzer
            logger.info("[exploiter] Compiled! Fuzzing for %ds...", self.fuzz_seconds)
            compile_error = ""  # clear so next iteration knows to use no-crash prompt
            crashed, poc_path, output = run_libfuzzer(
                bin_path, self.output_dir, self.fuzz_seconds,
            )

            if crashed:
                logger.info("[exploiter] *** CRASH *** %s in %s",
                           finding.bug_type, finding.function)
                return ExploitResult(
                    finding=finding, crashed=True, poc_path=poc_path,
                    crash_output=output, attempts=attempt + 1,
                    harness_path=bin_path,
                )

            last_fuzzer_output = output
            logger.info("[exploiter] No crash (attempt %d). Execs: %s",
                       attempt + 1, _extract_exec_count(output))

        # ── Template fallback (no LLM, uses tree-sitter param types) ──
        logger.info("[exploiter] Trying template harness for %s...", finding.function)
        template_result = self._try_template(finding, call_graph)
        if template_result:
            return template_result

        return ExploitResult(
            finding=finding, crashed=False, poc_path=None,
            crash_output="all attempts exhausted", attempts=4,
        )

    def _try_template(self, finding, call_graph) -> ExploitResult | None:
        """Generate a mechanical harness from tree-sitter parameter types."""
        fdef = call_graph.functions.get(finding.function)
        if not fdef or not fdef.params:
            return None

        includes = get_file_includes(self.src_dir, finding.file)
        includes_code = "\n".join(includes) if includes else ""

        # Map parameters to fuzz input mechanically
        param_setup = []
        param_args = []
        offset = 0
        needs_cleanup = []

        for i, param in enumerate(fdef.params):
            pname = param.name or f"p{i}"

            if param.is_size or param.is_unsigned_int:
                # Integer: read 4 bytes from fuzz data, DO NOT CLAMP
                param_setup.append(f"    unsigned int {pname};")
                param_setup.append(f"    memcpy(&{pname}, data + {offset}, 4);")
                param_args.append(pname)
                offset += 4

            elif param.is_pointer:
                # Pointer: point into remaining fuzz data
                cast = param.type_text.strip() if param.type_text else "const void *"
                if not cast.endswith("*"):
                    cast = "const unsigned char *"
                param_setup.append(f"    {cast} {pname} = ({cast})(data + {offset});")
                param_args.append(pname)

            else:
                # Struct/opaque/unknown type — template can't handle this.
                # The LLM harness attempts (1-3) should have covered it.
                # Skip this function for template generation.
                logger.debug("[template] Skipping %s — can't auto-create param type: %s",
                            finding.function, param.full_text)
                return None

        min_size = max(offset + 16, 32)
        cleanup = "\n".join(needs_cleanup) if needs_cleanup else ""

        code = f"""\
/* Template harness for {finding.function} ({finding.bug_type}) */
/* Auto-generated from tree-sitter parameter analysis */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
{includes_code}
#include "{finding.file}"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {{
    if (size < {min_size}) return 0;

{chr(10).join(param_setup)}

    {finding.function}({', '.join(param_args)});

{cleanup}
    return 0;
}}
"""

        bin_path, err = compile_harness(
            code, f"template_{finding.function}", self.src_dir, self.output_dir,
        )

        if not bin_path:
            logger.warning("[template] Compile failed: %s", err[:200])
            return None

        logger.info("[template] Compiled! Fuzzing for %ds...", self.fuzz_seconds)
        crashed, poc_path, output = run_libfuzzer(
            bin_path, self.output_dir, self.fuzz_seconds,
        )

        if crashed:
            logger.info("[template] *** CRASH *** %s via template harness", finding.bug_type)
            return ExploitResult(
                finding=finding, crashed=True, poc_path=poc_path,
                crash_output=output, attempts=5,
                harness_path=bin_path,
            )

        return None


# ══════════════════════════════════════════════════════════════════
# AGENT 3: VERIFIER (LLM classification + reachability testing)
# ══════════════════════════════════════════════════════════════════

VERIFIER_PROMPT = """\
Given a crash report and source code, determine:
1. Real security vulnerability or benign crash?
2. CWE classification
3. Severity (critical/high/medium/low)

IMPORTANT: Do NOT guess exploitability. We will test that separately.
Focus only on what the crash report shows.

Respond ONLY with a JSON object. Start with { end with }.
{"real_vulnerability":true,"cwe":"CWE-122","cwe_name":"Heap Buffer Overflow",
 "severity":"high",
 "root_cause":"one sentence","impact":"one sentence"}"""

REACHABILITY_PROMPT = """\
A crash was found in the internal function `{function}` by calling it
directly with fuzz-controlled parameters. Now we need to check if this
bug is reachable through the library's PUBLIC API.

The bug: {bug_type} — {description}
Trigger: {trigger_hint}

Call chain from public API to the vulnerable function:
{call_chains}

Write a Python script that generates a CRAFTED INPUT FILE designed to
reach `{function}` through the public API with parameter values that
trigger the bug.

For example, if the bug is an integer overflow in a string length
calculation, generate input with extremely long strings that would cause
the parser to compute large lengths.

The script must write to "/tmp/reachability_input".
Use only Python standard library.

Output ONLY the Python script. No markdown. Start with a comment."""




class VerifierAgent:
    def __init__(self, llm):
        self.llm = llm

    def verify(self, crash_output, finding, call_graph, src_dir):
        """Classify the crash (CWE, severity) — does NOT judge exploitability."""
        if not self.llm.is_available():
            return {"real_vulnerability": True, "severity": "unknown", "verified": False}

        user_msg = f"## Crash\n```\n{crash_output[:3000]}\n```\n"
        if finding:
            user_msg += f"\nFunction: {finding.function}\nBug: {finding.bug_type}\n{finding.description}\n"
        if finding and call_graph:
            ctx = call_graph.get_function_context(finding.function, src_dir)
            if ctx:
                user_msg += f"\n```c\n{ctx[:3000]}\n```"

        response = self.llm.chat(system=VERIFIER_PROMPT, user=user_msg,
                                 max_tokens=500, temperature=0.1)
        if not response:
            return {"real_vulnerability": True, "severity": "unknown", "verified": False}

        clean = response.strip()
        s, e = clean.find("{"), clean.rfind("}")
        if s != -1 and e != -1:
            clean = clean[s:e + 1]
        try:
            r = json.loads(clean)
            r["verified"] = True
            return r
        except json.JSONDecodeError:
            return {"real_vulnerability": True, "severity": "unknown", "verified": False}

    def check_reachability(self, finding, call_graph, main_binary, src_dir):
        """Test if the bug is reachable through the public API.
        
        This is the key difference from naive CRSs:
        - Direct harness crash = the code HAS a bug
        - Public API crash = the bug is EXPLOITABLE
        - Only public API confirmed = we don't hallucinate exploitability
        """
        if not main_binary or not os.path.exists(main_binary):
            logger.warning("[reachability] No main binary available for testing.")
            return {"reachable": "unknown", "method": "no_binary"}

        if not self.llm.is_available():
            return {"reachable": "unknown", "method": "no_llm"}

        # Get call chains from public API to vulnerable function
        callers = call_graph.get_callers(finding.function, depth=6) if call_graph else []
        chains_str = ""
        if callers:
            for path in callers[:5]:
                chains_str += f"  {' → '.join(path)}\n"
        else:
            chains_str = "  (no call chain found)\n"

        # Ask LLM to generate a crafted public API input
        logger.info("[reachability] Generating public API input for %s...", finding.function)

        response = self.llm.chat(
            system="You are a security researcher. Output ONLY a Python script.",
            user=REACHABILITY_PROMPT.format(
                function=finding.function,
                bug_type=finding.bug_type,
                description=finding.description,
                trigger_hint=finding.trigger_hint,
                call_chains=chains_str,
            ),
            max_tokens=2000, temperature=0.3,
        )

        if not response:
            return {"reachable": "unknown", "method": "llm_failed"}

        script = _clean_code(response)

        # Run the script to generate the input
        script_path = tempfile.mktemp(suffix=".py")
        with open(script_path, "w") as f:
            f.write(script)

        input_path = "/tmp/reachability_input"
        try:
            os.unlink(input_path)
        except FileNotFoundError:
            pass

        try:
            subprocess.run(["python3", script_path],
                          capture_output=True, timeout=30)
        except Exception:
            pass
        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass

        if not os.path.exists(input_path):
            logger.info("[reachability] Script failed to generate input.")
            return {"reachable": "unknown", "method": "script_failed"}

        input_data = Path(input_path).read_bytes()
        logger.info("[reachability] Testing %d byte input against public API...",
                    len(input_data))

        # Run the crafted input against the MAIN binary (public API)
        env = os.environ.copy()
        env["ASAN_OPTIONS"] = "abort_on_error=1:detect_leaks=0"

        try:
            result = subprocess.run(
                [main_binary, input_path],
                capture_output=True, timeout=10, env=env,
            )
            output = (result.stderr or b"").decode("utf-8", errors="replace")

            if result.returncode != 0 and (
                "AddressSanitizer" in output or
                "SUMMARY:" in output or
                result.returncode < 0
            ):
                # CRASH through public API — genuinely exploitable!
                logger.info("[reachability] *** PUBLIC API CRASH *** — bug is EXPLOITABLE")

                return {
                    "reachable": True,
                    "method": "public_api_crash",
                    "crash_output": output[:1000],
                    "input_size": len(input_data),
                    "exploitable": True,
                }
            else:
                logger.info("[reachability] No crash through public API.")
                return {
                    "reachable": False,
                    "method": "public_api_no_crash",
                    "exploitable": False,
                    "note": "Bug exists in code but is not reachable through the public API. "
                            "The parser validates input before reaching the vulnerable function. "
                            "This is a hardening finding — the code lacks defensive checks "
                            "but is currently protected by input validation upstream.",
                }

        except subprocess.TimeoutExpired:
            logger.info("[reachability] Public API test timed out.")
            return {"reachable": "unknown", "method": "timeout"}
        except Exception as exc:
            return {"reachable": "unknown", "method": f"error: {exc}"}


# ══════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

class AgentOrchestrator:
    """Coordinates Scanner → Exploiter → Verifier + Reachability pipeline."""

    def __init__(self, llm, call_graph, src_dir, output_dir,
                 fuzz_seconds=120, main_binary=None, static_findings=""):
        self.scanner = ScannerAgent(llm, max_workers=3)
        self.exploiter = ExploiterAgent(llm, src_dir, output_dir, fuzz_seconds)
        self.verifier = VerifierAgent(llm)
        self.call_graph = call_graph
        self.src_dir = src_dir
        self.output_dir = output_dir
        self.main_binary = main_binary  # public API binary for reachability
        self.static_findings = static_findings  # tool output for Scanner
        self.all_findings = []
        self.all_exploits = []
        self.generated_harnesses = []
        self.bugs_dir = Path(output_dir) / "bugs"
        self.bugs_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self, max_scan_targets=8, max_exploit_targets=3):
        logger.info("╔══════════════════════════════════════════════════╗")
        logger.info("║  Multi-Agent Pipeline (v4)                      ║")
        logger.info("╚══════════════════════════════════════════════════╝")

        # Phase 1: Scan
        logger.info("[pipeline] Phase 1: Scanning top %d targets...", max_scan_targets)
        findings = self.scanner.scan_top_targets(
            self.call_graph, self.src_dir, max_scan_targets,
            static_context=self.static_findings,
        )
        self.all_findings.extend(findings)
        logger.info("[pipeline] Scanner: %d findings.", len(findings))

        if not findings:
            return {"scan_findings": 0, "crashes_found": 0, "harnesses_built": 0}

        # Phase 2: Exploit top findings
        top = sorted(findings, key=lambda f: -f.confidence)[:max_exploit_targets]
        for finding in top:
            logger.info("[pipeline] Exploiting: %s in %s (%.2f)",
                       finding.bug_type, finding.function, finding.confidence)

            result = self.exploiter.exploit(finding, self.call_graph)
            self.all_exploits.append(result)

            if result.harness_path:
                self.generated_harnesses.append(result.harness_path)

            if result.crashed:
                # Phase 3: SKEPTICAL TRIAGE (grep-verified)
                # Instead of guessing, we challenge the finding and grep
                # the actual codebase for evidence.
                logger.info("[pipeline] Running skeptical triage (grep-verified)...")

                # Get source code for triage
                triage_code = self.call_graph.get_function_context(
                    finding.function, self.src_dir,
                )

                triage_result = skeptical_triage(
                    self.exploiter.llm,
                    finding_function=finding.function,
                    finding_bug_type=finding.bug_type,
                    finding_description=finding.description,
                    filepath=finding.file,
                    code=triage_code[:6000],
                    src_dir=self.src_dir,
                    num_rounds=3,
                )

                report = {
                    "function": finding.function, "file": finding.file,
                    "bug_type": finding.bug_type, "confidence": finding.confidence,
                    "description": finding.description,
                    "poc_path": result.poc_path,
                    "triage_verdict": triage_result.verdict,
                    "triage_confidence": triage_result.confidence,
                    "triage_rounds": triage_result.rounds,
                    "triage_valid_votes": triage_result.valid_votes,
                    "triage_reasoning": triage_result.reasoning,
                    "triage_grep_evidence": triage_result.grep_evidence,
                    "strategy": "multi_agent_v5",
                    "timestamp": time.time(),
                }
                rh = hashlib.sha256(
                    json.dumps(report, sort_keys=True, default=str).encode()
                ).hexdigest()[:12]
                (self.bugs_dir / f"verified-{rh}.json").write_text(
                    json.dumps(report, indent=2, default=str)
                )

                if triage_result.verdict == "VALID":
                    logger.info("╔══════════════════════════════════════════╗")
                    logger.info("║  *** VALID VULNERABILITY ***             ║")
                    logger.info("║  %s in %s", finding.bug_type, finding.function)
                    logger.info("║  Confidence: %.0f%% (%d/%d rounds VALID)",
                               triage_result.confidence * 100,
                               triage_result.valid_votes, triage_result.rounds)
                    logger.info("║  Verified via grep — no defense found    ║")
                    logger.info("╚══════════════════════════════════════════╝")
                elif triage_result.verdict == "INVALID":
                    logger.info("╔══════════════════════════════════════════╗")
                    logger.info("║  INVALID (defense found via grep)        ║")
                    logger.info("║  %s in %s", finding.bug_type, finding.function)
                    logger.info("║  %s", triage_result.reasoning[:60])
                    logger.info("╚══════════════════════════════════════════╝")
                else:
                    logger.info("╔══════════════════════════════════════════╗")
                    logger.info("║  UNCERTAIN                               ║")
                    logger.info("║  %s in %s", finding.bug_type, finding.function)
                    logger.info("║  Confidence: %.0f%%", triage_result.confidence * 100)
                    logger.info("╚══════════════════════════════════════════╝")

        crashes = sum(1 for r in self.all_exploits if r.crashed)
        exploitable = sum(1 for r in self.all_exploits
                         if r.crashed and hasattr(r, '_exploitable') and r._exploitable)
        return {
            "scan_findings": len(findings),
            "exploit_attempts": len(self.all_exploits),
            "crashes_found": crashes,
            "harnesses_built": len(self.generated_harnesses),
        }


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _clean_code(response: str) -> str:
    code = response.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1] if "\n" in code else code[3:]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


def _parse_json_array(response: str) -> list[dict]:
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


def _extract_exec_count(output: str) -> str:
    for line in reversed(output.split("\n")):
        if "execs_per_sec" in line or "Total execs" in line:
            return line.strip()[:80]
    return "unknown"
