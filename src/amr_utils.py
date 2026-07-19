"""Shared canonical AMR/PENMAN utilities.

Raw Vietnamese annotations contain unquoted multiword concept labels and a
small number of empty variable declarations.  This module performs the
deterministic, semantics-preserving normalization needed to create the active
PENMAN corpus while keeping the original files unchanged.
"""

from __future__ import annotations

import re
from collections import Counter

import penman


def _canonicalize_concepts(amr: str) -> str:
    """Make unquoted concept phrases single PENMAN symbols.

    The source annotation uses ``(x / multi word concept)``.  A whitespace-free
    symbol is the least disruptive canonical form for this corpus.  Quoted
    constants are left untouched.
    """
    result: list[str] = []
    index = 0
    while index < len(amr):
        if amr[index] != "/":
            result.append(amr[index])
            index += 1
            continue

        result.append("/")
        index += 1
        while index < len(amr) and amr[index].isspace():
            index += 1

        start = index
        quoted = False
        while index < len(amr):
            char = amr[index]
            if char == '"':
                quoted = not quoted
            if not quoted and char in ":)":
                break
            index += 1

        concept = re.sub(r"\s+", "_", amr[start:index].strip())
        result.append(" " + concept)
        if index < len(amr) and amr[index] == ":":
            result.append(" ")

    return "".join(result)


def _repair_empty_variables(amr: str) -> str:
    """Repair the source's empty variable declarations deterministically."""
    used = set(re.findall(r"\(\s*([^\s()/]+)\s*/", amr))
    # Some source graphs use ``(:role( / x))`` for a reference to an existing
    # node. Preserve that edge instead of inventing a new concept node.
    for variable in sorted(used, key=len, reverse=True):
        amr = re.sub(
            r"(:[A-Za-z0-9_-]+)\s*\(\s*/\s*"
            + re.escape(variable)
            + r"\s*\)",
            rf"\1 {variable}",
            amr,
        )

    # An atomic empty-variable node is an attribute, not a graph node.
    amr = re.sub(
        r"(:[A-Za-z0-9_-]+)\s*\(\s*/\s*([^()\s:]+)\s*\)",
        r"\1 \2",
        amr,
    )

    counters: Counter[str] = Counter()

    def replace(match: re.Match[str]) -> str:
        concept = match.group(1)
        base_match = re.search(r"[A-Za-zÀ-ỹ]", concept)
        base = base_match.group(0).lower() if base_match else "x"
        candidate = base
        counters[base] += 1
        if candidate in used:
            candidate = f"{base}{counters[base]}"
            while candidate in used:
                counters[base] += 1
                candidate = f"{base}{counters[base]}"
        used.add(candidate)
        return f"({candidate} / {concept}"

    return re.sub(r"\(\s*/\s*([^()\s:]+)", replace, amr)


def canonicalize_amr(amr: str) -> str:
    """Return canonical single-line PENMAN text without changing graph content."""
    normalized = re.sub(r"\s+", " ", amr).strip()
    normalized = _canonicalize_concepts(normalized)
    normalized = _repair_empty_variables(normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def decode_amr(amr: str):
    """Decode canonical AMR text or raise Penman's DecodeError."""
    return penman.decode(canonicalize_amr(amr))


def validate_amr(amr: str) -> bool:
    try:
        decode_amr(amr)
    except Exception:
        return False
    return True


def graph_components(graph) -> tuple[Counter, Counter, Counter, str]:
    """Return concept, relation, attribute counters and root concept."""
    triples = graph.triples
    variables = {source for source, _, _ in triples if source in graph.variables()}
    instance_map = {source: target for source, role, target in triples if role == ":instance"}
    concepts = Counter(instance_map.values())
    relations = Counter(
        (role, target)
        for source, role, target in triples
        if role != ":instance" and target in variables
    )
    attributes = Counter(
        (role, target)
        for source, role, target in triples
        if role != ":instance" and target not in variables
    )
    root = instance_map.get(graph.top, "")
    return concepts, relations, attributes, root
