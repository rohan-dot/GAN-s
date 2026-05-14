"""
main.py — gemma-fuzzer orchestrator (v4).

Phase 0: Build call graph via tree-sitter (or regex fallback)
Phase 1: Pre-fuzzing — prescan + codebase map + multi-agent pipeline
Phase 2: Fuzzing + continuous strategies + parallel harnesses
Phase 3: Final analysis and summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from crash_analyzer import CrashAnalyzer
from fuzzer import LibFuzzerRunner
from llm_client import VLLMClient
from strategies import StrategyOrchestrator
from code_analysis import build_callgraph, get_file_includes, find_include_dirs
from agents import AgentOrchestrator, compile_harness

LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

AUTO_HARNESS_PROMPT = """\
You are a fuzzing engineer. Given the source code of a C library, generate
a LibFuzzer harness that exercises the library's main parsing/processing function.

Read the header files and source to understand:
1. What is the main entry-point function? (parse, read, decode, process, etc.)
2. What input format does it accept? (string, binary buffer, file, etc.)
3. What cleanup is needed? (free, delete, close, etc.)

The harness must:
- #include the necessary headers (read them from the source)
- Implement int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
- Convert fuzz input to the format the library expects
- Call the main processing function
- Clean up
- Return 0

If the library parses strings (JSON, XML, config files):
    null-terminate the fuzz data before passing it.

If the library processes binary data (images, archives, protocols):
    pass data and size directly.

Output ONLY C code. No markdown fences. Start with #include."""





def _auto_generate_harness(llm, src_dir, output_dir, log):
    """Auto-generate a basic harness by reading the source."""
    src_path = Path(src_dir)

    # Gather context: headers and main source files
    context = ""

    # Read header files (API surface)
    headers = sorted(src_path.rglob("*.h"))
    for h in headers[:5]:
        if any(s in str(h).lower() for s in [".git", "test", "example"]):
            continue
        try:
            content = h.read_text(errors="replace")
            rel = str(h.relative_to(src_path))
            context += f"\n// === {rel} ===\n{content[:3000]}\n"
        except Exception:
            pass

    # Read main source files
    sources = sorted(src_path.rglob("*.c"))
    for s in sources[:3]:
        if any(skip in str(s).lower() for skip in [".git", "test", "example", "fuzz"]):
            continue
        try:
            content = s.read_text(errors="replace")
            rel = str(s.relative_to(src_path))
            context += f"\n// === {rel} (first 2000 chars) ===\n{content[:2000]}\n"
        except Exception:
            pass

    if not context:
        log.error("[auto-harness] No source files found in %s", src_dir)
        return None

    log.info("[auto-harness] Asking LLM to generate entry-point harness...")

    response = llm.chat(
        system="You are a security fuzzing engineer. Output ONLY C code.",
        user=AUTO_HARNESS_PROMPT + f"\n\nSource code:\n{context[:8000]}",
        max_tokens=2000,
        temperature=0.2,
    )

    if not response:
        log.error("[auto-harness] LLM returned no response.")
        return None

    code = response.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1] if "\n" in code else code[3:]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]

    log.info("[auto-harness] Generated harness, compiling...")

    # Try to compile (with auto-retry for missing macros)
    bin_path, error = compile_harness(code, "auto_harness", src_dir, output_dir)

    if bin_path:
        log.info("[auto-harness] SUCCESS: %s", bin_path)
        return bin_path

    # If first attempt failed, show error to LLM and retry
    log.warning("[auto-harness] First compile failed, retrying with fix...")
    fix_response = llm.chat(
        system="You are a C programmer. Fix the compilation error. Output ONLY C code.",
        user=f"Code:\n```c\n{code}\n```\n\nError:\n```\n{error[:800]}\n```\n\nFix it. Start with #include.",
        max_tokens=2000,
        temperature=0.1,
    )

    if not fix_response:
        return None

    fixed = fix_response.strip()
    if fixed.startswith("```"):
        fixed = fixed.split("\n", 1)[1] if "\n" in fixed else fixed[3:]
    if fixed.endswith("```"):
        fixed = fixed.rsplit("```", 1)[0]

    bin_path, error = compile_harness(fixed, "auto_harness_v2", src_dir, output_dir)

    if bin_path:
        log.info("[auto-harness] SUCCESS (attempt 2): %s", bin_path)
        return bin_path

    log.error("[auto-harness] Failed after 2 attempts: %s", error[:200])
    return None


def parse_args():
    p = argparse.ArgumentParser(description="gemma-fuzzer orchestrator")
    p.add_argument("--build-dir", required=True)
    p.add_argument("--src-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed-dir", default=None)
    p.add_argument("--log-dir", default="/var/log/gemma-fuzzer")
    p.add_argument("--harness", required=True)
    p.add_argument("--vllm-host", default="127.0.0.1")
    p.add_argument("--vllm-port", default="8000")
    p.add_argument("--vllm-model", default="gpt-oss-120b")
    p.add_argument("--fuzz-timeout", type=int, default=3600)
    p.add_argument("--fuzz-jobs", type=int, default=1)
    p.add_argument("--llm-seed-interval", type=int, default=120)
    return p.parse_args()


def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(Path(log_dir) / "orchestrator.log"),
        ],
    )


def run_harness_background(harness_bin, output_dir, timeout, log):
    """Run a generated harness in a background thread."""
    name = Path(harness_bin).stem
    crash_dir = Path(output_dir) / "crashes" / name
    corpus_dir = Path(output_dir) / "corpus" / name
    crash_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        harness_bin, str(corpus_dir),
        f"-artifact_prefix={crash_dir}/",
        f"-max_total_time={timeout}",
        "-detect_leaks=0", "-max_len=65536", "-timeout=30",
    ]

    log.info("[harness-bg] Starting: %s (%ds)", name, timeout)
    try:
        env = os.environ.copy()
        env["ASAN_OPTIONS"] = "abort_on_error=1:symbolize=1:detect_leaks=0"
        subprocess.run(cmd, capture_output=True, timeout=timeout + 30, env=env)

        pov_dir = Path(output_dir) / "povs"
        for cf in crash_dir.glob("crash-*"):
            shutil.copy2(cf, pov_dir / f"{name}-{cf.name}")
            log.info("[harness-bg] *** CRASH from %s: %s ***", name, cf.name)
    except subprocess.TimeoutExpired:
        log.info("[harness-bg] %s finished (timeout).", name)
    except Exception as exc:
        log.error("[harness-bg] %s failed: %s", name, exc)


def main():
    args = parse_args()
    setup_logging(args.log_dir)
    log = logging.getLogger("gemma-fuzzer")

    log.info("=" * 60)
    log.info("gemma-fuzzer v4.0 (tree-sitter + parallel agents)")
    log.info("  harness:      %s", args.harness)
    log.info("  build_dir:    %s", args.build_dir)
    log.info("  src_dir:      %s", args.src_dir)
    log.info("  fuzz_timeout: %ds", args.fuzz_timeout)
    log.info("  vllm:         %s:%s (%s)", args.vllm_host, args.vllm_port, args.vllm_model)
    log.info("=" * 60)

    output_dir = Path(args.output_dir)
    pov_dir = output_dir / "povs"
    corpus_dir = output_dir / "corpus"
    crash_dir = output_dir / "crashes"
    for d in [pov_dir, corpus_dir, crash_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Initialize components ─────────────────────────────────────

    llm = VLLMClient(args.vllm_host, args.vllm_port, args.vllm_model)
    analyzer = CrashAnalyzer(llm, args.src_dir, args.output_dir)

    # ── Auto-generate harness if no binary exists ─────────────────

    harness_name = args.harness
    build_dir = args.build_dir

    binary_exists = False
    for candidate in [
        Path(build_dir) / harness_name,
        Path(build_dir) / f"{harness_name}_fuzzer",
    ]:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            binary_exists = True
            break
    if not binary_exists:
        for g in Path(build_dir).glob(f"*{harness_name}*"):
            if g.is_file() and os.access(g, os.X_OK):
                binary_exists = True
                break

    if not binary_exists and llm.is_available():
        log.info("╔══════════════════════════════════════════════════╗")
        log.info("║  No binary found — auto-generating harness      ║")
        log.info("╚══════════════════════════════════════════════════╝")

        generated_bin = _auto_generate_harness(llm, args.src_dir, args.output_dir, log)
        if generated_bin:
            build_dir = str(Path(generated_bin).parent)
            harness_name = Path(generated_bin).stem
            log.info("Auto-generated harness: %s", generated_bin)
        else:
            log.error("Failed to auto-generate harness. Provide a binary via --build-dir.")

    strategies = StrategyOrchestrator(
        llm, args.src_dir, build_dir, args.output_dir, harness_name,
    )
    runner = LibFuzzerRunner(
        build_dir, harness_name, str(corpus_dir),
        str(crash_dir), args.seed_dir, args.fuzz_jobs,
    )

    # Find the main binary path for reachability testing
    binary_path = None
    for candidate in [
        Path(build_dir) / harness_name,
        Path(build_dir) / f"{harness_name}_fuzzer",
    ]:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            binary_path = str(candidate)
            break
    if not binary_path:
        for g in Path(build_dir).glob(f"*{harness_name}*"):
            if g.is_file() and os.access(g, os.X_OK):
                binary_path = str(g)
                break

    # ══════════════════════════════════════════════════════════════
    # PHASE 0: BUILD CALL GRAPH
    # ══════════════════════════════════════════════════════════════

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  PHASE 0: Building Call Graph                    ║")
    log.info("╚══════════════════════════════════════════════════╝")

    call_graph = build_callgraph(args.src_dir)
    log.info(call_graph.to_summary())

    # ── Run static analysis tools ─────────────────────────────────
    from static_analysis import run_all as run_static, findings_to_context
    from taint_tracker import analyze_codebase as run_taint, flows_to_context

    log.info("[static] Running static analysis tools...")
    static_findings = run_static(args.src_dir)
    static_context = findings_to_context(static_findings)
    if static_findings:
        log.info("[static] %d findings from real tools. Top 5:", len(static_findings))
        for f in static_findings[:5]:
            log.info("  [%s] %s:%d — %s %s: %s",
                    f.tool, f.file, f.line, f.severity.upper(), f.cwe, f.message[:60])

    log.info("[taint] Running source-to-sink taint analysis...")
    taint_result = run_taint(args.src_dir)
    taint_context = flows_to_context(taint_result.flows)
    if taint_result.flows:
        log.info("[taint] %d source→sink flows found. Top 5:", len(taint_result.flows))
        for f in taint_result.flows[:5]:
            log.info("  [%s] %s:%s:%d — %s → %s",
                    f.severity.upper(), f.file, f.function, f.line,
                    f.source, f.sink)

    # Combine all tool evidence
    all_tool_context = ""
    if static_context:
        all_tool_context += static_context + "\n\n"
    if taint_context:
        all_tool_context += taint_context

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: PRE-FUZZING ANALYSIS
    # ══════════════════════════════════════════════════════════════

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  PHASE 1: Pre-Fuzzing LLM Analysis              ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # Strategy round 1: prescan + codebase map + audit + seeds
    pre_results = strategies.run_round(str(corpus_dir))
    for r in pre_results:
        log.info("  [pre-fuzz] %s: %d findings (%.1fs)",
                r.strategy_name, r.findings, r.elapsed)

    # Multi-agent pipeline
    if len(call_graph.functions) > 0:
        log.info("╔══════════════════════════════════════════════════╗")
        log.info("║  Multi-Agent Pipeline                            ║")
        log.info("╚══════════════════════════════════════════════════╝")

        agent_orch = AgentOrchestrator(
            llm, call_graph, args.src_dir, args.output_dir,
            fuzz_seconds=min(120, args.fuzz_timeout // 4),
            main_binary=binary_path,
            static_findings=all_tool_context,
        )
        pipeline_result = agent_orch.run_pipeline(
            max_scan_targets=8, max_exploit_targets=3,
        )

        # Collect generated harnesses for background fuzzing
        for h in agent_orch.generated_harnesses:
            strategies.generated_harnesses.append(h)

        # Feed crashes to strategy engine
        for exploit in agent_orch.all_exploits:
            if exploit.crashed:
                strategies.add_crash(
                    f"{exploit.finding.bug_type}: {exploit.finding.description}",
                    {"crash_type": exploit.finding.bug_type,
                     "affected_function": exploit.finding.function,
                     "root_cause": exploit.finding.description},
                )

        log.info("[pipeline] Results: %s",
                json.dumps(pipeline_result) if pipeline_result else "none")
    else:
        log.warning("Empty call graph — skipping multi-agent pipeline.")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: FUZZING + CONTINUOUS STRATEGIES
    # ══════════════════════════════════════════════════════════════

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  PHASE 2: Fuzzing + Continuous Strategies       ║")
    log.info("╚══════════════════════════════════════════════════╝")

    runner.start(duration=args.fuzz_timeout)
    log.info("LibFuzzer launched for %d seconds.", args.fuzz_timeout)

    # Start generated harnesses in background
    threads = []
    for h in strategies.get_generated_harnesses():
        t = threading.Thread(
            target=run_harness_background,
            args=(h, args.output_dir, args.fuzz_timeout, log),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Main loop
    crashes_processed = 0
    last_strategy = time.monotonic()

    while runner.is_running():
        time.sleep(5)

        # Process crashes from main fuzzer
        for crash in runner.get_new_crashes(since_idx=crashes_processed):
            log.info("CRASH: %s (%s)", crash.crash_type, crash.crash_file)
            report = analyzer.analyze_crash(crash.crash_file, crash.stack_trace)
            if crash.crash_file and Path(crash.crash_file).exists():
                shutil.copy2(crash.crash_file, pov_dir / Path(crash.crash_file).name)
            summary = crash.crash_type
            if report and report.get("root_cause"):
                summary = f"{crash.crash_type}: {report['root_cause']}"
            strategies.add_crash(summary, report)
            crashes_processed += 1

        # Periodic strategy rounds
        now = time.monotonic()
        if (now - last_strategy) >= args.llm_seed_interval:
            strategies.run_round(str(corpus_dir))
            last_strategy = now

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: FINAL ANALYSIS
    # ══════════════════════════════════════════════════════════════

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  PHASE 3: Final Analysis                        ║")
    log.info("╚══════════════════════════════════════════════════╝")

    runner.wait()

    for crash in runner.get_new_crashes(since_idx=crashes_processed):
        analyzer.analyze_crash(crash.crash_file, crash.stack_trace)
        if crash.crash_file and Path(crash.crash_file).exists():
            shutil.copy2(crash.crash_file, pov_dir / Path(crash.crash_file).name)

    for t in threads:
        t.join(timeout=10)

    # Summary
    total_crashes = len(runner.state.crashes)
    total_povs = len(list(pov_dir.glob("*")))
    total_bugs = len(list((output_dir / "bugs").glob("*"))) if (output_dir / "bugs").exists() else 0
    total_seeds = len(list((output_dir / "seeds").glob("*"))) if (output_dir / "seeds").exists() else 0
    total_harnesses = len(strategies.get_generated_harnesses())

    log.info("=" * 60)
    log.info("gemma-fuzzer v4.0 — FINAL RESULTS")
    log.info("  Total executions:      %d", runner.state.total_execs)
    log.info("  Unique crashes:        %d", total_crashes)
    log.info("  PoVs:                  %d", total_povs)
    log.info("  Bug reports:           %d", total_bugs)
    log.info("  LLM-generated seeds:   %d", total_seeds)
    log.info("  Generated harnesses:   %d", total_harnesses)
    log.info("  Strategy rounds:       %d", strategies.round_number)
    log.info("")
    log.info("Strategy breakdown:")
    for r in strategies.results_log:
        log.info("  %-20s  %3d findings  (%5.1fs)",
                r.strategy_name, r.findings, r.elapsed)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
