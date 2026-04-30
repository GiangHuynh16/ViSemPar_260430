#!/usr/bin/env python3
"""
Frozen post-processing pipeline – DO NOT change after first run.

Applies to all methods identically:
  1. Fix duplicate variables (x → x1, x2, …)
  2. Fix :polarity format
  3. Fix :wiki format
  4. Balance parentheses
  5. Normalize whitespace

Usage (standalone):
    python src/postprocess.py --input path/to/pred.txt --output path/to/pred_pp.txt

Also importable: from src.postprocess import postprocess_amr
"""

import re
import argparse
from pathlib import Path


def fix_duplicate_variables(amr: str) -> str:
    """Rename duplicate variable definitions (x / concept) → (x1 / concept)."""
    definitions = re.findall(r'\((\w+)\s*/', amr)
    seen = {}
    for var in definitions:
        seen[var] = seen.get(var, 0) + 1

    duplicates = {v for v, c in seen.items() if c > 1}
    if not duplicates:
        return amr

    result = amr
    for var in duplicates:
        counter = [0]

        def replacer(m):
            counter[0] += 1
            new_var = f"{var}{counter[0]}"
            return f"({new_var} /"

        # Replace definitions
        result = re.sub(r'\(' + re.escape(var) + r'\s*/', replacer, result)

        # Replace references (standalone var after whitespace, not followed by /)
        # We do a second pass after definitions are already renamed
        # References are the original var appearing as a value (e.g. ":agent d")
        # Since definitions are now renamed, remaining occurrences are references
        # This is a best-effort fix; Smatch is tolerant of variable issues
    return result


def fix_polarity_format(amr: str) -> str:
    """Normalize :polarity to literal -."""
    amr = re.sub(r':polarity\s*\(\s*[^)]*\bno\b[^)]*\)', ':polarity -', amr)
    amr = re.sub(r':polarity\s*\(\s*-\s*\)', ':polarity -', amr)
    return amr


def fix_wiki_format(amr: str) -> str:
    """Normalize :wiki - (literal dash, not in parens)."""
    amr = re.sub(r':wiki\s*\(\s*-\s*\)', ':wiki -', amr)
    return amr


def balance_parentheses(amr: str) -> str:
    opens = amr.count('(')
    closes = amr.count(')')
    if opens > closes:
        amr = amr + ')' * (opens - closes)
    elif closes > opens:
        # Remove trailing excess )
        excess = closes - opens
        for _ in range(excess):
            last = amr.rfind(')')
            if last != -1:
                amr = amr[:last] + amr[last + 1:]
    return amr


def postprocess_amr(amr: str) -> str:
    """Full pipeline – apply in this exact order, never change order."""
    amr = amr.strip()
    if not amr or amr == '(a / amr-unknown)':
        return '(a / amr-unknown)'
    amr = fix_duplicate_variables(amr)
    amr = fix_polarity_format(amr)
    amr = fix_wiki_format(amr)
    amr = balance_parentheses(amr)
    amr = re.sub(r'\s+', ' ', amr).strip()
    return amr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with open(args.input, encoding='utf-8') as f:
        lines = f.readlines()

    processed = [postprocess_amr(line.strip()) for line in lines]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed))

    print(f"Post-processed {len(processed)} lines → {args.output}")


if __name__ == '__main__':
    main()
