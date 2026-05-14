"""
taint_tracker.py — Source-to-sink taint analysis using tree-sitter.

Traces data flow from SOURCES (function parameters, external input)
to SINKS (memcpy, malloc, sprintf, strcpy, etc.) within each function.

This is real static analysis, not LLM guessing. It uses the tree-sitter
AST to track variable assignments and function call arguments.

Limitations (vs Joern/CodeQL):
- Intra-procedural only (within one function, not across calls)
- No pointer aliasing analysis
- No control-flow sensitivity (doesn't track if/else branches)

But it catches the common pattern: parameter → local var → dangerous call.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("gemma-fuzzer.taint")

# Try tree-sitter
HAS_TREE_SITTER = False
_parser = None
_c_lang = None

try:
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser
    _c_lang = Language(tsc.language())
    _parser = Parser(_c_lang)
    HAS_TREE_SITTER = True
except ImportError:
    pass


# Dangerous sinks — function name → which argument indices are dangerous
SINKS = {
    # Buffer operations (size/length args are dangerous)
    "memcpy": [2],        # memcpy(dst, src, SIZE)
    "memmove": [2],       # memmove(dst, src, SIZE)
    "memset": [2],        # memset(dst, val, SIZE)
    "strncpy": [2],       # strncpy(dst, src, SIZE)
    "strncat": [2],       # strncat(dst, src, SIZE)

    # Unbounded string ops (dst buffer is dangerous)
    "strcpy": [0],        # strcpy(DST, src)
    "strcat": [0],        # strcat(DST, src)
    "sprintf": [0],       # sprintf(DST, fmt, ...)
    "gets": [0],          # gets(DST)

    # Allocation (size arg is dangerous — integer overflow)
    "malloc": [0],        # malloc(SIZE)
    "calloc": [0, 1],     # calloc(COUNT, SIZE)
    "realloc": [1],       # realloc(ptr, SIZE)

    # Format strings
    "printf": [0],        # printf(FMT, ...)
    "fprintf": [1],       # fprintf(f, FMT, ...)
    "snprintf": [2],      # snprintf(buf, size, FMT, ...)

    # Array indexing (index arg)
    "fread": [2],         # fread(buf, size, COUNT, f)
    "fwrite": [2],        # fwrite(buf, size, COUNT, f)
}


@dataclass
class TaintFlow:
    """A source-to-sink data flow."""
    source: str           # e.g., "parameter 'len'"
    sink: str             # e.g., "memcpy size argument"
    sink_function: str    # e.g., "memcpy"
    variable_chain: list[str]  # e.g., ["len", "size", "n"]
    function: str         # function where the flow occurs
    file: str
    line: int
    severity: str         # "high", "medium", "low"


@dataclass
class TaintResult:
    """Results of taint analysis on a codebase."""
    flows: list[TaintFlow] = field(default_factory=list)
    functions_analyzed: int = 0
    files_analyzed: int = 0


def analyze_codebase(src_dir: str) -> TaintResult:
    """Run taint analysis on all C files in a directory."""
    result = TaintResult()
    src_path = Path(src_dir)

    for fpath in src_path.rglob("*.c"):
        fstr = str(fpath).lower()
        if any(s in fstr for s in [
            ".git", "/test", "example", "python", "CMakeFiles",
            "aflplusplus", "honggfuzz", "libfuzzer", "qemu_mode",
        ]):
            continue

        try:
            content = fpath.read_text(errors="replace")
            rel_path = str(fpath.relative_to(src_path))
        except Exception:
            continue

        result.files_analyzed += 1

        if HAS_TREE_SITTER:
            flows = _analyze_file_treesitter(content, rel_path)
        else:
            flows = _analyze_file_regex(content, rel_path)

        result.flows.extend(flows)
        result.functions_analyzed += len(set(f.function for f in flows))

    # Sort by severity
    sev_order = {"high": 0, "medium": 1, "low": 2}
    result.flows.sort(key=lambda f: sev_order.get(f.severity, 9))

    logger.info("[taint] Analyzed %d files, %d functions. Found %d source→sink flows.",
                result.files_analyzed, result.functions_analyzed, len(result.flows))

    return result


def flows_to_context(flows: list[TaintFlow], max_flows: int = 15) -> str:
    """Format taint flows as context for LLM prompts."""
    if not flows:
        return ""

    lines = [f"## Source-to-Sink Taint Flows ({len(flows)} total)\n"]
    for f in flows[:max_flows]:
        lines.append(
            f"[{f.severity.upper()}] {f.file}:{f.function}:{f.line} — "
            f"{f.source} → {' → '.join(f.variable_chain)} → {f.sink}"
        )

    if len(flows) > max_flows:
        lines.append(f"\n... and {len(flows) - max_flows} more flows.")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# TREE-SITTER ANALYSIS
# ══════════════════════════════════════════════════════════════════

def _analyze_file_treesitter(content: str, filename: str) -> list[TaintFlow]:
    """Analyze a file using tree-sitter for taint tracking."""
    if not _parser:
        return []

    content_bytes = content.encode("utf-8", errors="replace")
    tree = _parser.parse(content_bytes)
    flows = []

    # Find all function definitions
    for node in _walk(tree.root_node):
        if node.type != "function_definition":
            continue

        func_flows = _analyze_function_treesitter(node, content_bytes, filename)
        flows.extend(func_flows)

    return flows


def _analyze_function_treesitter(func_node, content_bytes, filename):
    """Track taint from parameters to sinks within a function."""
    flows = []

    # Extract function name
    func_decl = _find_descendant(func_node, "function_declarator")
    if not func_decl:
        return []
    name_node = _find_child(func_decl, "identifier")
    if not name_node:
        return []
    func_name = content_bytes[name_node.start_byte:name_node.end_byte].decode()

    # Extract parameter names (these are our SOURCES)
    params_node = _find_child(func_decl, "parameter_list")
    param_names = set()
    if params_node:
        for child in _walk(params_node):
            if child.type == "identifier" and child.parent and \
               child.parent.type in ("parameter_declaration", "pointer_declarator"):
                pname = content_bytes[child.start_byte:child.end_byte].decode()
                if pname not in ("void", "const", "unsigned", "int", "char"):
                    param_names.add(pname)

    if not param_names:
        return []

    # Track tainted variables through assignments
    body = _find_child(func_node, "compound_statement")
    if not body:
        return []

    tainted = dict(param_names)  # variable → source description
    tainted_vars = set(param_names)

    # Pass 1: Find assignments that propagate taint
    for node in _walk(body):
        if node.type == "assignment_expression":
            left = _find_child(node, "identifier")
            if not left:
                continue
            lhs = content_bytes[left.start_byte:left.end_byte].decode()

            # Check if RHS uses any tainted variable
            rhs_text = content_bytes[node.start_byte:node.end_byte].decode()
            for tvar in list(tainted_vars):
                if re.search(r'\b' + re.escape(tvar) + r'\b', rhs_text):
                    tainted_vars.add(lhs)
                    break

        # Also track declarations with initializers
        elif node.type == "init_declarator":
            decl = _find_child(node, "identifier")
            if not decl:
                decl = _find_descendant(node, "identifier")
            if not decl:
                continue
            lhs = content_bytes[decl.start_byte:decl.end_byte].decode()
            init_text = content_bytes[node.start_byte:node.end_byte].decode()
            for tvar in list(tainted_vars):
                if re.search(r'\b' + re.escape(tvar) + r'\b', init_text):
                    tainted_vars.add(lhs)
                    break

    # Pass 2: Find calls to dangerous functions with tainted args
    for node in _walk(body):
        if node.type != "call_expression":
            continue

        callee_node = _find_child(node, "identifier")
        if not callee_node:
            continue
        callee = content_bytes[callee_node.start_byte:callee_node.end_byte].decode()

        if callee not in SINKS:
            continue

        dangerous_indices = SINKS[callee]

        # Get argument list
        arg_list = _find_child(node, "argument_list")
        if not arg_list:
            continue

        args = [c for c in arg_list.children
                if c.type not in ("(", ")", ",")]

        for idx in dangerous_indices:
            if idx >= len(args):
                continue

            arg_text = content_bytes[args[idx].start_byte:args[idx].end_byte].decode()

            # Check if this argument uses any tainted variable
            for tvar in tainted_vars:
                if re.search(r'\b' + re.escape(tvar) + r'\b', arg_text):
                    # Build variable chain
                    chain = _build_chain(tvar, param_names, tainted_vars)

                    severity = "high" if callee in ("strcpy", "strcat", "sprintf", "gets") \
                        else "high" if callee in ("malloc", "realloc", "calloc") and idx in (0, 1) \
                        else "medium"

                    flows.append(TaintFlow(
                        source=f"parameter '{tvar}'" if tvar in param_names
                               else f"derived from parameter (via '{tvar}')",
                        sink=f"{callee}() argument {idx + 1}",
                        sink_function=callee,
                        variable_chain=chain,
                        function=func_name,
                        file=filename,
                        line=node.start_point[0] + 1,
                        severity=severity,
                    ))
                    break  # one flow per sink argument

    return flows


def _build_chain(tvar, param_names, tainted_vars):
    """Build a simple variable chain from parameter to current var."""
    if tvar in param_names:
        return [tvar]
    # Can't easily trace back without more state, return simple chain
    return [f"param", tvar]


# ══════════════════════════════════════════════════════════════════
# REGEX FALLBACK
# ══════════════════════════════════════════════════════════════════

def _analyze_file_regex(content: str, filename: str) -> list[TaintFlow]:
    """Fallback: basic source-to-sink detection using regex."""
    flows = []

    # Find function definitions and their parameters
    func_pattern = re.compile(
        r'(\w+)\s*\(([^)]+)\)\s*\{',
        re.MULTILINE,
    )

    for match in func_pattern.finditer(content):
        func_name = match.group(1)
        if func_name in ("if", "while", "for", "switch"):
            continue

        params_text = match.group(2)
        # Extract parameter names (last word before comma or close paren)
        param_names = set()
        for param in params_text.split(","):
            tokens = param.strip().split()
            if tokens:
                name = tokens[-1].strip("*")
                if name and name not in ("void", "const"):
                    param_names.add(name)

        if not param_names:
            continue

        # Find the function body (rough — up to 200 lines)
        body_start = match.end()
        body_end = min(body_start + 10000, len(content))
        body = content[body_start:body_end]

        # Check for dangerous function calls with parameter-derived args
        for sink_name, dangerous_args in SINKS.items():
            sink_calls = re.finditer(
                re.escape(sink_name) + r'\s*\(([^)]+)\)', body,
            )
            for call_match in sink_calls:
                args_text = call_match.group(1)
                args = [a.strip() for a in args_text.split(",")]

                for idx in dangerous_args:
                    if idx >= len(args):
                        continue
                    arg = args[idx]

                    for pname in param_names:
                        if re.search(r'\b' + re.escape(pname) + r'\b', arg):
                            line_no = content[:body_start + call_match.start()].count("\n") + 1

                            flows.append(TaintFlow(
                                source=f"parameter '{pname}'",
                                sink=f"{sink_name}() argument {idx + 1}",
                                sink_function=sink_name,
                                variable_chain=[pname],
                                function=func_name,
                                file=filename,
                                line=line_no,
                                severity="high" if sink_name in ("strcpy", "sprintf", "gets") else "medium",
                            ))
                            break

    return flows


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _find_child(node, type_name):
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _find_descendant(node, type_name):
    for child in node.children:
        if child.type == type_name:
            return child
        found = _find_descendant(child, type_name)
        if found:
            return found
    return None
