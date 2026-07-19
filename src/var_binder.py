#!/usr/bin/env python3
from __future__ import annotations

"""Deterministic variable binder for no-var AMR.

Converts forms such as:
    (muốn :agent(tôi) :theme(đi))
to:
    (m / muốn :agent (t / tôi) :theme (đ / đi))
"""

import argparse
import re
from pathlib import Path


class NoVarBinder:
    def __init__(self):
        self.counts = {}

    def _new_var(self, concept: str) -> str:
        match = re.search(r"[A-Za-zÀ-ỹ]", concept)
        base = match.group(0).lower() if match else "x"
        self.counts[base] = self.counts.get(base, 0) + 1
        return base if self.counts[base] == 1 else f"{base}{self.counts[base]}"

    def bind(self, amr: str) -> str:
        self.counts = {}
        amr = amr.strip()
        if not amr:
            return ""
        if re.search(r"\(\s*[\wÀ-ỹ]+\s*/", amr):
            return re.sub(r"\s+", " ", amr).strip()
        text, pos = self._parse_node(amr, 0)
        if text:
            return re.sub(r"\s+", " ", text).strip()
        return ""

    def _skip_ws(self, text: str, pos: int) -> int:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return pos

    def _read_until_role_or_paren(self, text: str, pos: int) -> tuple[str, int]:
        start = pos
        while pos < len(text):
            ch = text[pos]
            if ch in "()":
                break
            if ch == ":":
                break
            pos += 1
        return text[start:pos].strip(), pos

    def _read_role(self, text: str, pos: int) -> tuple[str, int]:
        start = pos
        pos += 1
        while pos < len(text) and (text[pos].isalnum() or text[pos] in "-_+"):
            pos += 1
        return text[start:pos], pos

    def _read_atom(self, text: str, pos: int) -> tuple[str, int]:
        start = pos
        while pos < len(text):
            ch = text[pos]
            if ch in "()":
                break
            if ch == ":":
                break
            if ch.isspace():
                nxt = self._skip_ws(text, pos)
                if nxt >= len(text) or text[nxt] in ":()":
                    break
            pos += 1
        return text[start:pos].strip(), pos

    def _parse_node(self, text: str, pos: int) -> tuple[str | None, int]:
        pos = self._skip_ws(text, pos)
        if pos >= len(text) or text[pos] != "(":
            return None, pos
        pos += 1
        pos = self._skip_ws(text, pos)

        concept, pos = self._read_until_role_or_paren(text, pos)
        if not concept:
            concept = "amr-unknown"
        var = self._new_var(concept)
        parts = [f"({var} / {concept}"]

        while pos < len(text):
            pos = self._skip_ws(text, pos)
            if pos >= len(text):
                break
            if text[pos] == ")":
                pos += 1
                parts.append(")")
                return " ".join(parts), pos
            if text[pos] == ":":
                role, pos = self._read_role(text, pos)
                pos = self._skip_ws(text, pos)
                if pos < len(text) and text[pos] == "(":
                    child, pos = self._parse_node(text, pos)
                    if child:
                        parts.append(f"{role} {child}")
                else:
                    atom, pos = self._read_atom(text, pos)
                    if atom:
                        parts.append(f"{role} {atom}")
                continue
            atom, pos = self._read_atom(text, pos)
            if atom:
                parts.append(atom)
            else:
                pos += 1

        parts.append(")")
        return " ".join(parts), pos


def bind_no_var_amr(amr: str) -> str:
    return NoVarBinder().bind(amr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    binder = NoVarBinder()
    with open(args.input, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    bound = [binder.bind(line) for line in lines]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(bound))

    print(f"Bound {len(bound)} AMRs -> {args.output}")


if __name__ == "__main__":
    main()
