#!/usr/bin/env python3
"""Summarize Harbor job directories: per-trial rewards, components, mean/std, cost, errors.

  summarize.py jobs/<job> [jobs/<job> ...]               print a table per job
  summarize.py --markdown report/x.md jobs/<job>         also write a markdown report
  summarize.py --check tasks/<task> jobs/<job> ...       validate the oracle/nop/cheat jobs
                                                         (decided by job name; exits 1 on failure)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

PASS_TARGETS = {"mean_lt": 0.5, "std_gt": 0.08}


def load_trials(job_dir: Path) -> list[dict]:
    trials = []
    for rp in sorted(job_dir.glob("*/result.json")):
        r = json.loads(rp.read_text())
        vr = r.get("verifier_result") or {}
        ar = r.get("agent_result") or {}
        ex = r.get("exception_info")
        trials.append(
            {
                "trial": r.get("trial_name", rp.parent.name),
                "rewards": vr.get("rewards") or {},
                "cost_usd": ar.get("cost_usd"),
                "tokens_in": ar.get("n_input_tokens"),
                "tokens_out": ar.get("n_output_tokens"),
                "error": f"{ex['exception_type']}: {ex['exception_message'][:120]}" if ex else None,
                "path": str(rp.parent),
            }
        )
    return trials


def stats(trials: list[dict]) -> dict:
    # Trials with no verifier result count as 0. A failed run is a failed rollout.
    xs = [float(t["rewards"].get("reward", 0.0)) for t in trials]
    out = {"n": len(xs), "rewards": xs}
    if xs:
        out["mean"] = statistics.fmean(xs)
        out["std_sample"] = statistics.stdev(xs) if len(xs) > 1 else 0.0
        out["std_pop"] = statistics.pstdev(xs)
    costs = [t["cost_usd"] for t in trials if t["cost_usd"] is not None]
    out["cost_usd"] = sum(costs) if costs else None
    return out


def render(job_dir: Path, trials: list[dict], s: dict) -> str:
    comp_keys = sorted({k for t in trials for k in t["rewards"] if k != "reward"})
    head = ["trial", "reward", *comp_keys, "cost $", "error"]
    lines = [f"### {job_dir.name}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t in trials:
        row = [t["trial"], _f(t["rewards"].get("reward"))]
        row += [_f(t["rewards"].get(k)) for k in comp_keys]
        row += [_f(t["cost_usd"], 3), (t["error"] or "").replace("|", "/")]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if s["n"]:
        ok_mean = s["mean"] < PASS_TARGETS["mean_lt"]
        ok_std = s["std_sample"] > PASS_TARGETS["std_gt"]
        lines += [
            f"- n = {s['n']}",
            f"- mean = {s['mean']:.4f}  (target < 0.5: {'OK' if ok_mean else 'NO'})",
            f"- std (sample, ddof=1) = {s['std_sample']:.4f}  (target > 0.08: {'OK' if ok_std else 'NO'})",
            f"- std (population, ddof=0) = {s['std_pop']:.4f}",
        ]
    if s["cost_usd"] is not None:
        lines.append(f"- total agent cost = ${s['cost_usd']:.2f}")
    return "\n".join(lines) + "\n"


def check(task: Path, job_dir: Path, trials: list[dict]) -> list[str]:
    """Return failures for a validation job, based on its name."""
    name = job_dir.name
    rewards = [float(t["rewards"].get("reward", float("nan"))) for t in trials]
    errors = [t for t in trials if t["error"] or "reward" not in t["rewards"]]
    fails = []
    if not trials:
        return [f"{name}: no trials found"]
    if m := re.search(r"-cheat-(.+)-\d{8}-\d{6}$", name):
        cname = m.group(1)
        cap_file = task / "cheat" / cname / "expected_max"
        cap = float(cap_file.read_text().strip()) if cap_file.exists() else 0.0
        if errors:
            fails.append(f"{name}: verifier error ({errors[0]['error']})")
        if any(r > cap for r in rewards):
            fails.append(f"{name}: cheat scored {rewards} > allowed {cap}")
    elif "-oracle-" in name:
        if errors or any(r != 1.0 for r in rewards):
            fails.append(f"{name}: oracle rewards {rewards} (all must be exactly 1.0)")
        comps = [json.dumps(t["rewards"], sort_keys=True) for t in trials]
        if len(set(comps)) > 1:
            fails.append(f"{name}: oracle grading not identical across runs")
    elif "-nop-" in name:
        if errors or any(r != 0.0 for r in rewards):
            fails.append(f"{name}: nop rewards {rewards} (must be 0.0)")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", type=Path)
    ap.add_argument("--markdown", type=Path)
    ap.add_argument("--check", type=Path, metavar="TASK_DIR")
    a = ap.parse_args()

    md, failures = [], []
    for job in a.jobs:
        trials = load_trials(job)
        s = stats(trials)
        block = render(job, trials, s)
        print(block)
        md.append(block)
        (job / "summary.json").write_text(json.dumps({"stats": s, "trials": trials}, indent=2))
        if a.check:
            failures += check(a.check, job, trials)

    if a.markdown:
        a.markdown.parent.mkdir(parents=True, exist_ok=True)
        a.markdown.write_text("\n".join(md))
        print(f"wrote {a.markdown}")
    if a.check:
        print("VALIDATION: " + ("PASS" if not failures else "FAIL"))
        for f in failures:
            print("  - " + f)
        sys.exit(1 if failures else 0)


def _f(x, nd: int = 4) -> str:
    return "" if x is None else f"{x:.{nd}f}"


if __name__ == "__main__":
    main()
