#!/usr/bin/env python
"""Recompute every sweep-derived cell and diff it against the LaTeX tables.

Rows that came from the previously published runs (the GPT intervenors and the
GPT blocks' controls) have no data under results/intervenor_sweep, so they are
reported as unverifiable rather than silently passed.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_intervenor_sweep as R

DEN = {"airline": 50, "retail": 115}
SWEEP_INTERVENORS = {"DeepSeek-V3.2", "Mistral-Large-3", "Llama-4-Maverick", "Llama-3.3-70B",
                     "Mistral-Small", "Mistral-Medium-2505"}
# airline GPT-5-mini intervenor cells were scored against the sweep's own baseline
DAGGER_BLOCK = "airline_gpt-5-mini"


def cell_values(key):
    """{row label: (after, boost)} recomputed from disk for one sweep cell."""
    n = DEN[R.baselines()[key]["env"]]
    base = R.load(os.path.join(R.SWEEP, key, "transcript.json"))
    passed = {r["task_id"] for r in base if r.get("traj") and r["reward"] == 1.0}
    fails = {r["task_id"] for r in base if r.get("traj") and r["reward"] != 1.0}
    b = 100 * len(passed) / n
    out = {"None": (b, None)}

    def add(label, solved):
        a = 100 * (len(passed) + len(solved)) / n
        out[label] = (a, 100 * (a - b) / b)

    fs = sorted(glob.glob(os.path.join(R.SWEEP, key, "reflexion-by_gpt-4o_*", "reflexion-transcripts.json")),
                key=os.path.getmtime)
    if fs:
        bt = R.by_task([r for r in R.load_tolerant(fs[-1], repair=False) if not R.is_err(r)])
        add("None (Reflexion)", {t for t, v in bt.items() if t in fails and v and v[0]["reward"] == 1.0})
        add("None (Reflexion, Bo-6)", {t for t, v in bt.items() if t in fails and any(x["reward"] == 1.0 for x in v)})
    fs = sorted(glob.glob(os.path.join(R.SWEEP, key, "partial-reroll-by_*", "partial-reroll-transcripts.json")),
                key=os.path.getmtime)
    if fs:
        bt = R.by_task([r for r in R.load_tolerant(fs[-1], repair=False) if not R.is_err(r)])
        add("None (Partial RR)", {t for t, v in bt.items() if t in fails and any(x["reward"] == 1.0 for x in v)})
    seeds = glob.glob(os.path.join(R.SWEEP, key, "full_rr_runs", "*", "transcript.json"))
    if seeds:
        solved = set()
        for f in seeds:
            solved |= {r["task_id"] for r in R.load_tolerant(f, repair=False)
                       if r.get("traj") and r["reward"] == 1.0} & fails
        add("None (Full RR)", solved)
    for iv in R.baselines()[key].get("intervenors", R.INTERVENORS):
        fs = R.iv_files(key, iv)
        if not fs:
            continue
        d = R.load_tolerant(fs[0], repair=False)
        trusted = R.history_ids(os.path.dirname(fs[0]))
        bt = R.by_task(d)
        att = {t for t, rows in bt.items() if t in fails and R.settled(rows, t in trusted)}
        add(f"{iv}\\bof", {t for t in att if any(x["reward"] == 1.0 for x in bt[t])})
        add(f"{iv}\\sing", {t for t in att for r in bt[t]
                            if str(r.get("intervened_first_or_last")) == "0" and r["reward"] == 1.0})
    return out


NUM = r"(?:\\textbf\{)?(\d+\.\d+)\}?"


def parse_rows(path):
    """[(label, [values...])] for each body row of a table."""
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("%") or "&" not in line or "\\\\" not in line:
            continue
        if "multicolumn{2}{c}" in line or "parbox{2.1cm}" in line:
            continue
        cells = [c.strip() for c in line.split("\\\\")[0].split("&")]
        label = None
        for c in cells:
            m = re.search(r"(None \(Reflexion, Bo-6\)|None \(Reflexion\)|None \(Full RR\)|None \(Partial RR\)|None|[A-Za-z0-9.\-]+\\(?:bof|sing))\s*$", c)
            if m:
                label = m.group(1)
                break
        if not label:
            continue
        vals = []
        for c in cells[cells.index(next(c for c in cells if label in c)) + 1:]:
            m = re.match(r"^" + NUM, c.replace("$^\\dag$", "").replace("$^\\ddag$", ""))
            boost = re.search(r"\\boost\{([\d.]+)\}", c)
            vals.append((float(m.group(1)) if m else None, float(boost.group(1)) if boost else None))
        rows.append((label, vals))
    return rows


def close(a, b, tol=0.06):
    return a is not None and b is not None and abs(a - b) <= tol


def boost_close(got, want):
    if got is None or want is None:
        return got is None and want is None
    return abs(got - float(f"{want:.3g}")) <= max(0.06, abs(want) * 0.006)


def main():
    tex = sys.argv[1]
    blocks = sys.argv[2:]           # worker keys in table order
    rows = parse_rows(tex)
    # split rows into blocks at each "None" baseline row
    groups, cur = [], None
    for label, vals in rows:
        if label == "None":
            cur = []
            groups.append(cur)
        if cur is not None:
            cur.append((label, vals))
    if not groups:          # a table whose baselines live in the column headers
        groups = [rows]
    print(f"{os.path.basename(tex)}: {len(groups)} blocks, {len(rows)} rows\n")
    ok = bad = skip = 0
    for gi, group in enumerate(groups):
        keys = blocks[gi].split(",")          # one key per value column
        computed = [cell_values(k) for k in keys]
        print(f"--- block {gi+1}: {', '.join(keys)}")
        for label, vals in group:
            for ci, (got_a, got_boost) in enumerate(vals):
                if got_a is None:
                    continue
                name = label.split("\\")[0] if "\\" in label else label
                iv = name if name in SWEEP_INTERVENORS else None
                if label.startswith("None") or iv:
                    want = computed[ci].get(label)
                else:
                    want = None
                if want is None:
                    skip += 1
                    print(f"    {label:26} col{ci+1} {got_a:6.1f}  (no data on disk - published row)")
                    continue
                a_ok = close(got_a, want[0])
                b_ok = boost_close(got_boost, want[1])
                if a_ok and b_ok:
                    ok += 1
                else:
                    bad += 1
                    print(f"  ** {label:26} col{ci+1} table {got_a:6.1f}/{got_boost}  data {want[0]:6.1f}/{want[1] and round(want[1],3)}")
        print()
    print(f"verified {ok} cells, {bad} MISMATCHES, {skip} unverifiable (published rows)")


if __name__ == "__main__":
    main()
