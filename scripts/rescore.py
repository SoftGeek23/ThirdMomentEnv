#!/usr/bin/env python3
"""Recompute rewards for finished trials with a new REF_SPEEDUP (no re-running).

The grader logs the raw measurements (speedup_geomean, dnll_worst, ranking checks) in each
trial's verifier/reward.json, so the score can be recomputed with the same formula as
tests/grade.py once the reference is (re)calibrated.

  python3 scripts/rescore.py --ref 1.62 jobs/<job> [jobs/<job> ...]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks/qwen3-w8a8-recipe/tests"))
import grade as G  # noqa: E402  (same quality_gate / speed_credit / thresholds as the grader)


def rescore(r: dict, ref: float) -> float | None:
    if "speedup_geomean" not in r:  # build or quality step failed -> grader already scored 0
        return 0.0
    ranking_ok = (r.get("spearman_min", 0) >= G.MIN_SPEARMAN
                  and r.get("keep_agreement_min", 0) >= G.MIN_KEEP_AGREEMENT)
    q = G.quality_gate(r["dnll_worst"]) if ranking_ok else 0.0
    return q * G.speed_credit(r["speedup_geomean"], ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=float, required=True, help="new REF_SPEEDUP (geomean)")
    ap.add_argument("jobs", nargs="+", type=Path)
    a = ap.parse_args()
    for job in a.jobs:
        rows = []
        for rp in sorted(job.glob("*/verifier/reward.json")):
            r = json.loads(rp.read_text())
            new = rescore(r, a.ref)
            rows.append((rp.parts[-3], r.get("reward"), new, r.get("speedup_geomean"), r.get("dnll_worst")))
        print(f"### {job.name}  (REF_SPEEDUP {G.REF_SPEEDUP} -> {a.ref})")
        print("| trial | old reward | new reward | speedup | worst dNLL |\n|---|---|---|---|---|")
        for t, old, new, sp, dn in rows:
            f = lambda x, n=4: "" if x is None else f"{x:.{n}f}"  # noqa: E731
            print(f"| {t} | {f(old)} | {f(new)} | {f(sp, 3)} | {f(dn, 5)} |")
        new = [r[2] for r in rows]
        if len(new) > 1:
            print(f"\nmean {statistics.fmean(new):.4f}, std (sample) {statistics.stdev(new):.4f}\n")
        elif new:
            print(f"\nreward {new[0]:.4f}\n")


if __name__ == "__main__":
    main()
