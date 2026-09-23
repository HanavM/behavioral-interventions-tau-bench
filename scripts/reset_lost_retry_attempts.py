#!/usr/bin/env python
"""One-off: clear retry counts burned by the pre-fix merge bug.

Before the fix, a retry of a "no intervention was done" task was never merged, so
those tasks ran up MAX_TASK_ATTEMPTS without any retry ever being kept. Reset the
counts of tasks that are still unsettled in their canonical file. Run only while
the sweep driver is stopped. Pass --apply to write; default is a dry run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_intervenor_sweep as R

apply = "--apply" in sys.argv
for key in R.baselines():
    for iv in R.INTERVENORS:
        p = R.attempts_path(key, iv)
        files = R.iv_files(key, iv)
        if not os.path.exists(p) or not files:
            continue
        counts = R.load(p)
        rows = R.by_task(R.load_tolerant(files[0], repair=False))
        trusted = R.history_ids(os.path.dirname(files[0]))
        reset = sorted(int(t) for t in counts if int(t) in rows and not R.settled(rows[int(t)], int(t) in trusted))
        if not reset:
            continue
        print(f"{key:26} {iv:18} reset {reset} (counts {[counts[str(t)] for t in reset]})")
        if apply:
            R.dump(p, {t: c for t, c in counts.items() if int(t) not in reset})
print("applied" if apply else "dry run (pass --apply while the driver is stopped)")
