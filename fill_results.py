"""
fill_results.py
===============
Reads analysis/final_results_summary.json and replaces every \RESULT{key}
placeholder in paper.tex with the corresponding value, writing paper_final.tex.

Usage:
    python fill_results.py [--input paper.tex] [--output paper_final.tex]
                           [--summary analysis/final_results_summary.json]

The \RESULT{key} macro in paper.tex renders as a yellow-highlighted box
while awaiting results. This script removes those boxes and slots in real
numbers, producing a camera-ready draft.

Keys in final_results_summary.json that are not found in paper.tex are
reported as unused. Keys in paper.tex that are not in the JSON are reported
as missing (still placeholder).
"""

import json
import re
import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',   default='paper.tex',
                   help='Source .tex file with \\RESULT{key} placeholders')
    p.add_argument('--output',  default='paper_final.tex',
                   help='Destination .tex file with placeholders filled')
    p.add_argument('--summary', default='analysis/final_results_summary.json',
                   help='JSON file mapping result keys to values')
    p.add_argument('--strict',  action='store_true',
                   help='Exit with error if any placeholder is still missing')
    return p.parse_args()


def load_results(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    # Flatten one level of nesting if values are dicts
    flat = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f'{k}.{kk}'] = str(vv)
        else:
            flat[k] = str(v)
    return flat


def find_placeholders(tex: str) -> list[str]:
    """Return list of all unique keys referenced as \\RESULT{key}."""
    return list(dict.fromkeys(re.findall(r'\\RESULT\{([^}]+)\}', tex)))


def fill(tex: str, results: dict) -> tuple[str, list[str], list[str]]:
    """
    Replace \\RESULT{key} tokens with values from results dict.
    Returns (filled_tex, missing_keys, unused_keys).
    """
    placeholders = find_placeholders(tex)
    missing  = [k for k in placeholders if k not in results]
    unused   = [k for k in results       if k not in placeholders]

    def replacer(m):
        key = m.group(1)
        if key in results:
            return results[key]
        return m.group(0)   # leave placeholder intact if key missing

    filled = re.sub(r'\\RESULT\{([^}]+)\}', replacer, tex)
    return filled, missing, unused


def main():
    args = parse_args()

    # Load JSON
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"[ERROR] Summary file not found: {summary_path}")
        print("        Run analysis/alignment_faithfulness_analysis.py first.")
        sys.exit(1)
    results = load_results(args.summary)
    print(f"Loaded {len(results)} result keys from {args.summary}")

    # Load tex
    tex_path = Path(args.input)
    if not tex_path.exists():
        print(f"[ERROR] Input file not found: {tex_path}")
        sys.exit(1)
    tex = tex_path.read_text(encoding='utf-8')

    placeholders = find_placeholders(tex)
    print(f"Found {len(placeholders)} unique \\RESULT{{}} placeholders in {args.input}")

    # Fill
    filled, missing, unused = fill(tex, results)

    # Write output
    out_path = Path(args.output)
    out_path.write_text(filled, encoding='utf-8')
    print(f"Written: {args.output}")

    # Report
    filled_count = len(placeholders) - len(missing)
    print(f"\nFilled:  {filled_count}/{len(placeholders)} placeholders")

    if missing:
        print(f"\n[WARN] {len(missing)} placeholder(s) still missing (no JSON key):")
        for k in missing:
            print(f"  \\RESULT{{{k}}}")
    else:
        print("All placeholders filled.")

    if unused:
        print(f"\n[INFO] {len(unused)} JSON key(s) unused in {args.input}:")
        for k in unused:
            print(f"  {k} = {results[k][:60]}")

    if args.strict and missing:
        print("\n[ERROR] --strict mode: exiting because placeholders remain.")
        sys.exit(2)

    print("\nDone.")


if __name__ == '__main__':
    main()
