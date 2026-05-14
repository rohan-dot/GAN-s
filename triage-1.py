"""
triage.py — Skeptical triage with grep verification.

Inspired by AISLE's nano-analyzer. Three components:

1. CONTEXT GENERATION: LLM writes a security briefing about a file
   before scanning it. Identifies buffers, data flows, untrusted inputs.

2. GREP TOOL: LLM can request grep searches of the codebase to verify
   claims. "Is there a bounds check?" → grep for it → evidence-based answer.

3. SKEPTICAL TRIAGE: Each finding goes through N rounds of challenge.
   A skeptic tries to DISPROVE the finding using grep evidence.
   Only findings that survive are reported.

This replaces guessing with verification.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from llm_client import VLLMClient

logger = logging.getLogger("gemma-fuzzer.triage")


@dataclass
class TriageResult:
    """Result of skeptical triage."""
    finding_function: str
    finding_bug_type: str
    verdict: str           # "VALID", "INVALID", "UNCERTAIN"
    confidence: float      # 0.0-1.0 based on round votes
    rounds: int
    valid_votes: int
    reasoning: str         # final arbiter reasoning
    grep_evidence: list[str]  # grep queries used


# ══════════════════════════════════════════════════════════════════
# GREP TOOL
# ══════════════════════════════════════════════════════════════════

def run_grep(pattern: str, src_dir: str, max_results: int = 30) -> str:
    """Run grep on the source directory. Returns formatted results."""
    if not pattern or not pattern.strip():
        return "(empty pattern)"

    # Sanitize pattern — remove file path prefixes that users sometimes add
    pattern = pattern.strip().strip('"').strip("'")

    try:
        # Try ripgrep first (faster), fall back to grep
        rg = "rg" if os.path.exists("/usr/bin/rg") or os.path.exists("/usr/local/bin/rg") else None

        if rg:
            cmd = [rg, "-n", "--no-heading", "-M", "200", pattern, src_dir]
        else:
            cmd = ["grep", "-rn", "--include=*.c", "--include=*.h",
                   "-m", str(max_results), pattern, src_dir]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
        )

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

        # Filter out fuzzing infrastructure
        filtered = []
        for line in lines:
            if any(s in line.lower() for s in [
                "aflplusplus", "honggfuzz", "libfuzzer", "qemu_mode", "/test/"
            ]):
                continue
            filtered.append(line)

        if not filtered:
            return f"(no results for: {pattern})"

        # Limit results
        if len(filtered) > max_results:
            return "\n".join(filtered[:max_results]) + f"\n... ({len(filtered)} total)"

        return "\n".join(filtered)

    except subprocess.TimeoutExpired:
        return "(grep timed out)"
    except Exception as exc:
        return f"(grep error: {exc})"


def process_grep_requests(text: str, src_dir: str) -> str:
    """Find GREP: patterns in text, run them, append results."""
    grep_pattern = re.compile(r'GREP:\s*(.+?)(?:\n|$)', re.IGNORECASE)
    matches = grep_pattern.findall(text)

    if not matches:
        return ""

    results = []
    for pattern in matches[:5]:  # max 5 grep requests per response
        pattern = pattern.strip()
        grep_result = run_grep(pattern, src_dir)
        results.append(f"\n--- grep '{pattern}' ---\n{grep_result}")

    return "\n".join(results)


# ══════════════════════════════════════════════════════════════════
# CONTEXT GENERATION
# ══════════════════════════════════════════════════════════════════

CONTEXT_PROMPT = """\
You are preparing a security briefing for a vulnerability researcher.
Write a concise (~250 word) context briefing covering:

1. What this code does and where it sits in the project
2. How untrusted input reaches this code
3. Which variables carry attacker-controlled data — name them,
   trace the data flow from entry point to usage
4. All fixed-size buffers and size constants — name them with sizes
5. Dangerous data flows: attacker-controlled data → fixed-size buffer.
   Name source, destination, function, and the numeric buffer size
6. Parameters that could be NULL from malformed input but are
   dereferenced without checks
7. Which functions are public API vs static helpers
8. What bug classes are most likely given this code's structure

Name actual variables and constants from the code.
Do not find vulnerabilities — just provide context.

GREP TOOL: Include GREP: pattern to search the codebase for constant
values, callers, or data flow. For example:
GREP: MAX_BUF_SIZE
GREP: parse_input("""

_context_cache: dict[str, str] = {}


def generate_context(llm: VLLMClient, filepath: str, content: str,
                     src_dir: str) -> str:
    """Generate a security briefing for a source file."""
    if filepath in _context_cache:
        return _context_cache[filepath]

    if not llm.is_available():
        return ""

    response = llm.chat(
        system=CONTEXT_PROMPT,
        user=f"File: {filepath}\n\n```c\n{content[:12000]}\n```",
        max_tokens=1500, temperature=0.1,
    )

    if not response:
        return ""

    # Process any grep requests in the response
    grep_results = process_grep_requests(response, src_dir)
    if grep_results:
        # Send grep results back to LLM for updated briefing
        followup = llm.chat(
            system="Update your security briefing with these grep results. Keep it concise.",
            user=f"Original briefing:\n{response}\n\nGrep results:{grep_results}",
            max_tokens=1000, temperature=0.1,
        )
        if followup:
            response = followup

    _context_cache[filepath] = response
    logger.info("[context] Generated briefing for %s", filepath)
    return response


# ══════════════════════════════════════════════════════════════════
# SKEPTICAL TRIAGE
# ══════════════════════════════════════════════════════════════════

SKEPTIC_PROMPT = """\
A vulnerability scanner flagged this finding. Is it real?

Be skeptical — most scanner findings are false positives.

RULES:
- VALID: the bug is real AND an external attacker can trigger it.
  The attacker must control the input that reaches the bug.
- INVALID: the bug pattern doesn't exist, OR it's not attacker-reachable
  (only trusted internal callers), OR a concrete defense prevents it.
- UNCERTAIN: only if you genuinely cannot determine.

CRITICAL: When you cite a defense — a size limit, a NULL check, a
type validation — you MUST verify it works. Look up actual values.
Do the arithmetic. Show your work. "There exists a bound" is NOT
the same as "the bound is sufficient."

GREP TOOL: Include a grep pattern in your response to search the
codebase. Use this to verify defenses, find callers, resolve constants.
GREP: function_name(
GREP: MAX_BUFFER_SIZE

Do NOT invent defenses. If you can't find one via grep, it doesn't exist.

Respond with JSON:
{{"reasoning": "your analysis with evidence",
  "crux": "the single key fact the verdict depends on",
  "grep": "pattern to search for",
  "verdict": "VALID/INVALID/UNCERTAIN"}}

---

Finding: {bug_type} in {function}
Description: {description}
File: {filepath}

Code:
```c
{code}
```"""

SKEPTIC_FOLLOWUP = """\
Previous round verdict: {prev_verdict}
Previous reasoning: {prev_reasoning}

Grep results from previous round:
{grep_results}

{extra_rounds}

Re-evaluate with this new evidence. Look for something the previous
round MISSED. Try to find a DIFFERENT reason this is valid or invalid.
Don't just repeat the previous analysis.

Respond with JSON:
{{"reasoning": "new analysis with new evidence",
  "crux": "the single key fact",
  "grep": "new_pattern_to_search",
  "verdict": "VALID/INVALID/UNCERTAIN"}}"""

ARBITER_PROMPT = """\
You are the final arbiter. Multiple rounds of skeptical review have
analyzed this finding. Read all the evidence and make the final call.

Finding: {bug_type} in {function}
Description: {description}

Round verdicts and reasoning:
{all_rounds}

Make your decision based on the EVIDENCE, not the vote count.
If one round found a concrete defense via grep, that outweighs
five rounds of speculation.

Respond with JSON:
{{"final_verdict": "VALID/INVALID/UNCERTAIN",
  "confidence": 0.85,
  "reasoning": "your final analysis citing the strongest evidence"}}"""


def skeptical_triage(
    llm: VLLMClient,
    finding_function: str,
    finding_bug_type: str,
    finding_description: str,
    filepath: str,
    code: str,
    src_dir: str,
    num_rounds: int = 3,
) -> TriageResult:
    """Run skeptical triage on a finding. Returns verdict with evidence."""

    if not llm.is_available():
        return TriageResult(
            finding_function=finding_function,
            finding_bug_type=finding_bug_type,
            verdict="UNCERTAIN", confidence=0.5,
            rounds=0, valid_votes=0,
            reasoning="LLM unavailable", grep_evidence=[],
        )

    rounds_data = []
    all_grep_queries = []
    prev_verdict = ""
    prev_reasoning = ""
    prev_grep_results = ""

    for round_num in range(num_rounds):
        logger.info("[triage] Round %d/%d for %s in %s",
                    round_num + 1, num_rounds, finding_bug_type, finding_function)

        if round_num == 0:
            # First round: initial assessment
            prompt = SKEPTIC_PROMPT.format(
                bug_type=finding_bug_type,
                function=finding_function,
                description=finding_description,
                filepath=filepath,
                code=code[:6000],
            )
            response = llm.chat(
                system="You are a skeptical security reviewer. Respond ONLY with JSON.",
                user=prompt,
                max_tokens=1500, temperature=0.2,
            )
        else:
            # Subsequent rounds: re-evaluate with new evidence
            extra = ""
            if round_num > 1:
                extra = f"This is round {round_num + 1}. Previous rounds found:\n"
                for rd in rounds_data:
                    extra += f"  Round {rd['round']}: {rd['verdict']} — {rd.get('crux', '')}\n"

            prompt = SKEPTIC_FOLLOWUP.format(
                prev_verdict=prev_verdict,
                prev_reasoning=prev_reasoning[:500],
                grep_results=prev_grep_results[:2000],
                extra_rounds=extra,
            )
            response = llm.chat(
                system="You are a skeptical security reviewer. Respond ONLY with JSON.",
                user=prompt,
                max_tokens=1500, temperature=0.2 + round_num * 0.1,
            )

        if not response:
            continue

        # Parse JSON response
        parsed = _parse_json_obj(response)
        verdict = parsed.get("verdict", "UNCERTAIN").upper()
        reasoning = parsed.get("reasoning", "")
        crux = parsed.get("crux", "")
        grep_pattern = parsed.get("grep", "")

        # Run grep if requested
        grep_results = ""
        if grep_pattern:
            grep_results = run_grep(grep_pattern, src_dir)
            all_grep_queries.append(grep_pattern)
            logger.info("[triage] Grep '%s' → %d lines",
                       grep_pattern, grep_results.count("\n") + 1)

        rounds_data.append({
            "round": round_num + 1,
            "verdict": verdict,
            "reasoning": reasoning,
            "crux": crux,
            "grep_pattern": grep_pattern,
            "grep_results": grep_results[:500],
        })

        prev_verdict = verdict
        prev_reasoning = reasoning
        prev_grep_results = grep_results

    # Final arbiter
    all_rounds_text = ""
    for rd in rounds_data:
        all_rounds_text += (
            f"\nRound {rd['round']}: {rd['verdict']}\n"
            f"Reasoning: {rd['reasoning']}\n"
            f"Crux: {rd['crux']}\n"
        )
        if rd.get("grep_results"):
            all_rounds_text += f"Grep evidence: {rd['grep_results'][:300]}\n"

    arbiter_prompt = ARBITER_PROMPT.format(
        bug_type=finding_bug_type,
        function=finding_function,
        description=finding_description,
        all_rounds=all_rounds_text,
    )

    arbiter_response = llm.chat(
        system="You are a fair arbiter. Decide based on evidence. Respond ONLY with JSON.",
        user=arbiter_prompt,
        max_tokens=800, temperature=0.1,
    )

    # Parse arbiter decision
    final_verdict = "UNCERTAIN"
    final_confidence = 0.5
    final_reasoning = ""

    if arbiter_response:
        arbiter_parsed = _parse_json_obj(arbiter_response)
        final_verdict = arbiter_parsed.get("final_verdict", "UNCERTAIN").upper()
        final_confidence = float(arbiter_parsed.get("confidence", 0.5))
        final_reasoning = arbiter_parsed.get("reasoning", "")

    valid_votes = sum(1 for rd in rounds_data if rd["verdict"] == "VALID")

    logger.info("[triage] Final: %s (confidence=%.2f, votes=%d/%d VALID)",
               final_verdict, final_confidence, valid_votes, len(rounds_data))

    return TriageResult(
        finding_function=finding_function,
        finding_bug_type=finding_bug_type,
        verdict=final_verdict,
        confidence=final_confidence,
        rounds=len(rounds_data),
        valid_votes=valid_votes,
        reasoning=final_reasoning,
        grep_evidence=all_grep_queries,
    )


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _parse_json_obj(response: str) -> dict:
    """Parse a JSON object from LLM response."""
    clean = response.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()
    s, e = clean.find("{"), clean.rfind("}")
    if s != -1 and e != -1:
        clean = clean[s:e + 1]
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}
