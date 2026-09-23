#!/usr/bin/env python
"""Intervenor sweep driver.

Runs N=5 interventions for {open-source intervenors} x {staged baselines}, with the
intervenor's transcript-search tool pinned to one fixed open-source model so tool
quality never confounds intervenor quality.

Every job re-derives what is already on disk and runs only the gap, so the whole
sweep is safe to kill and restart at any moment (e.g. laptop shutdown).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(ROOT, "results", "intervenor_sweep")
PY = os.path.join(ROOT, ".venv", "bin", "python")

# Kimi-K2.6 was dropped: its deployment is pinned at capacity 100 and 429'd on
# essentially every intervenor call. Llama-3.3-70B replaces it (dense 70B, a
# different architecture class from the Maverick MoE).
INTERVENORS = ["DeepSeek-V3.2", "Mistral-Large-3", "Llama-4-Maverick", "Llama-3.3-70B"]
SEARCH_MODEL = "Llama-4-Scout"
SEARCH_PROVIDER = "azure_ai"
N = "5"
CONCURRENCY = "10"
IV_TEMP = "0.2"


def log(msg):
    print(f"SWEEP {datetime.now():%m-%d %H:%M:%S} {msg}", flush=True)


def load(p):
    with open(p) as f:
        return json.load(f)


def load_tolerant(p, repair=True):
    """Load a checkpoint, salvaging complete records if the file is truncated.

    A run killed mid-write (machine sleep, crash) can leave a partial array; the
    completed records in it are still valid work and should not be thrown away.
    """
    raw = open(p).read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    dec, recs = json.JSONDecoder(), []
    i = raw.index("[") + 1
    while True:
        while i < len(raw) and raw[i] in " \n\r\t,":
            i += 1
        if i >= len(raw) or raw[i] != "{":
            break
        try:
            obj, i = dec.raw_decode(raw, i)
        except json.JSONDecodeError:
            break
        recs.append(obj)
    if not repair:  # the file may still be mid-write by a live run; leave it alone
        return recs
    os.replace(p, p + ".corrupt")
    dump(p, recs)
    log(f"repaired truncated checkpoint {os.path.basename(os.path.dirname(p))}: kept {len(recs)} records")
    return recs


def dump(p, d):
    # write-then-rename so a kill mid-write never leaves a truncated checkpoint
    tmp = f"{p}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, p)


def is_err(r):
    return isinstance(r.get("info"), dict) and "error" in r["info"]


def baselines():
    return load(os.path.join(SWEEP, "baselines.json"))


NO_IV = "no intervention was done"


def by_task(rows):
    out = {}
    for r in rows:
        out.setdefault(r["task_id"], []).append(r)
    return out


def no_intervention(rows):
    return {str(r.get("intervened_message", "")) for r in rows} == {NO_IV}


def history_ids(folder):
    """Tasks a run finished: run.py writes this file only once the whole run is done."""
    p = os.path.join(folder, "agent_conversation_history.json")
    return {e.get("task_id") for e in load(p) if isinstance(e, dict)} if os.path.exists(p) else set()


def settled(rows, trusted):
    """Whether a task's rows are a usable verdict.

    Rows are appended one candidate at a time, so a kill mid-task leaves a partial
    best-of-N. A task is settled if its run finished (trusted), it already passed
    at baseline, or its candidate indices run contiguously 0..k-1 with k >= N.
    "No intervention was done" is never settled: it is usually a rate-limit
    casualty and is retried.
    """
    if any(is_err(r) for r in rows) or no_intervention(rows):
        return False
    if trusted:
        return True
    if len(rows) == 1 and str(rows[0].get("intervened_message", "")).startswith("no intervention was needed"):
        return True
    idx = {str(r.get("intervened_first_or_last")) for r in rows}
    return len(rows) >= int(N) and idx == {str(i) for i in range(len(rows))}


def iv_files(key, intervenor):
    """Canonical intervention file first, then any top-up files not yet folded in.

    The canonical file is the one covering the most tasks. Recency is not a safe
    signal: a top-up killed before its merge leaves a newer but far smaller folder.
    """
    fs = glob.glob(os.path.join(SWEEP, key, f"intervened-by_{intervenor}_*", "intervened-transcripts.json"))
    sizes = {f: len(by_task(load_tolerant(f, repair=False))) for f in fs}
    return sorted(fs, key=lambda f: (-sizes[f], os.path.basename(os.path.dirname(f))))


def fold_topups(key, intervenor):
    """Fold every non-canonical intervention folder into the canonical one.

    A task is taken from the top-up when the canonical file lacks it, or when the
    canonical rows are unsettled and the top-up's are settled. A task's rows always
    come from a single run, and its history entry follows them. Folded folders are
    archived under .merged/ rather than deleted.
    """
    fs = iv_files(key, intervenor)
    if len(fs) < 2:
        return
    tpath, td = fs[0], os.path.dirname(fs[0])
    for npath in fs[1:]:
        nd = os.path.dirname(npath)
        base, new = load_tolerant(tpath), load_tolerant(npath)
        base_t, new_t = by_task(base), by_task(new)
        base_ok, new_ok = history_ids(td), history_ids(nd)
        take = {t for t, rows in new_t.items()
                if t not in base_t
                or (not settled(base_t[t], t in base_ok) and settled(rows, t in new_ok))}
        dump(tpath, [r for r in base if r["task_id"] not in take] + [r for r in new if r["task_id"] in take])
        hp, ht = os.path.join(nd, "agent_conversation_history.json"), os.path.join(td, "agent_conversation_history.json")
        new_h = [e for e in load(hp) if isinstance(e, dict) and e.get("task_id") in take] if os.path.exists(hp) else []
        if os.path.exists(ht) or new_h:
            # a taken task's old canonical entry no longer describes its rows
            h = [e for e in (load(ht) if os.path.exists(ht) else []) if not (isinstance(e, dict) and e.get("task_id") in take)]
            dump(ht, h + new_h)
        log(f"INTERV {key} x {intervenor}: merged {len(take)} task(s) from {os.path.basename(nd)}")
        archive = os.path.join(SWEEP, key, ".merged")
        os.makedirs(archive, exist_ok=True)
        shutil.move(nd, os.path.join(archive, os.path.basename(nd)))


def live_run(key, intervenor):
    """A run.py for this cell still alive, e.g. one that outlived a killed driver."""
    pat = f"run.py --run_intervention --baseline_path {os.path.join(SWEEP, key)} .*--intervention_model {intervenor} "
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0


# Deployment capacity governs how many rollouts a model can really absorb.
# Oversubscribing a small deployment does not go faster: every extra caller just
# collects a 429 and sleeps, so throughput collapses.
TASK_CONCURRENCY_BY_MODEL = {
    "Kimi-K2.6": 2,          # capacity 100; 4 still 429'd on every call
    "Llama-4-Maverick": 8,   # capacity 250
    "Llama-3.3-70B": 8,      # capacity 250
    "DeepSeek-V3.2": 12,     # capacity 1000
    "Mistral-Large-3": 12,   # capacity 1000
    "Mistral-Small": 12,     # capacity 1000
    "Mistral-Medium": 12,    # capacity 1000
    "Mistral-Medium-2505": 12,  # capacity 1000
    "Llama-4-Scout": 12,     # capacity 1000 (search tool)
}
DEFAULT_TASK_CONCURRENCY = 20  # GPT deployments, provisioned far above this sweep


def concurrency_for(meta, intervenor=None):
    """(task concurrency, attempts-in-parallel) for one cell.

    A cell is gated by whichever participant has the least headroom: the worker
    drives the rollouts, but the intervenor and the search tool are hit once per
    task as well, so the tightest of them sets the task pool.
    """
    caps = [TASK_CONCURRENCY_BY_MODEL.get(meta["model"], DEFAULT_TASK_CONCURRENCY)]
    if intervenor:
        caps.append(TASK_CONCURRENCY_BY_MODEL.get(intervenor, DEFAULT_TASK_CONCURRENCY))
        caps.append(TASK_CONCURRENCY_BY_MODEL.get(SEARCH_MODEL, DEFAULT_TASK_CONCURRENCY))
    tasks = min(caps)
    # attempts hit only the worker, so they can stay wide even for a slow intervenor
    attempts = 5 if meta["model_provider"] == "azure" else 3
    return str(tasks), str(attempts)


# The rollout code sets no request timeout, so a wedged HTTPS connection parks a
# worker thread forever: the process keeps its sockets open but burns no CPU and
# writes nothing. Kill a child that has consumed no CPU for this long and let the
# next pass resume it - its finished tasks are already checkpointed.
STALL_MINUTES = 15
MIN_CPU_PROGRESS = 5.0   # seconds of CPU in a window; retry timers alone tick far less
STALL_RC = -9            # what proc.kill() reports, so callers can tell a stall from a crash


def child_cpu_seconds(pid):
    out = subprocess.run(["ps", "-o", "time=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    if not out:
        return None
    parts = [float(x) for x in out.replace("-", ":").split(":")]
    secs = 0.0
    for part in parts:
        secs = secs * 60 + part
    return secs


def run_cmd(cmd, logfile, attempt_concurrency="1"):
    env = dict(os.environ, TAU_ATTEMPT_CONCURRENCY=str(attempt_concurrency))
    with open(logfile, "a") as lf:
        lf.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd[:8])} ...\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        window_cpu, window_start = child_cpu_seconds(proc.pid) or 0.0, time.time()
        while True:
            try:
                return proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            cpu = child_cpu_seconds(proc.pid)
            if cpu is None:
                continue
            if time.time() - window_start < STALL_MINUTES * 60:
                pass
            elif cpu - window_cpu >= MIN_CPU_PROGRESS:
                window_cpu, window_start = cpu, time.time()   # real work happened, keep going
            else:
                log(f"STALLED: only {cpu - window_cpu:.1f}s CPU in {STALL_MINUTES} min, killing {os.path.basename(cmd[1])} (pid {proc.pid})")
                proc.kill()
                return proc.wait()


# --------------------------------------------------------------------------
# baseline jobs (only for the new open-source worker+user runs)
# --------------------------------------------------------------------------
def baseline_missing(key, meta):
    """Task ids still needed, minus ones that never produce a usable trajectory.

    A task that errors every time would otherwise be re-run on every pass forever,
    so retries are capped the same way intervention tasks are, and a cell can name
    known-bad ids in "exclude_tasks".
    """
    path = os.path.join(SWEEP, key, "transcript.json")
    dead = set(meta.get("exclude_tasks", [])) | exhausted(key, "baseline")
    if not os.path.exists(path):
        return [t for t in range(meta["tasks"]) if t not in dead]
    done = {r["task_id"] for r in load(path) if not is_err(r) and r.get("traj")}
    return [t for t in range(meta["tasks"]) if t not in done and t not in dead]


def run_baseline_job(key, meta):
    missing = baseline_missing(key, meta)
    if not missing:
        return True
    log(f"BASELINE {key}: {len(missing)} task(s) missing -> running")
    workdir = os.path.join(SWEEP, key, "baseline_runs")
    os.makedirs(workdir, exist_ok=True)
    cmd = [PY, "run.py", "--model", meta["model"], "--model-provider", meta["model_provider"],
           "--user-model", meta["user_model"], "--user-model-provider", meta["user_model_provider"],
           "--agent-strategy", "react", "--env", meta["env"],
           "--temperature", str(meta["temperature"]), "--max-concurrency", concurrency_for(meta)[0],
           "--log-dir", workdir, "--task-ids", *map(str, missing)]
    rc = run_cmd(cmd, os.path.join(SWEEP, key, "baseline.log"))
    log(f"BASELINE {key}: run rc={rc}")

    canonical = os.path.join(SWEEP, key, "transcript.json")
    merged = {r["task_id"]: r for r in (load(canonical) if os.path.exists(canonical) else [])}
    for f in glob.glob(os.path.join(workdir, "*", "transcript.json")):
        for r in load(f):
            if r.get("traj") and not is_err(r):
                merged[r["task_id"]] = r
    dump(canonical, [merged[k] for k in sorted(merged)])
    left = baseline_missing(key, meta)
    if rc in (0, STALL_RC) and left:
        # count attempts only once this run's results are merged, or every task
        # would look missing and the whole cell would retire after three passes
        bump_attempts(key, "baseline", left)
    log(f"BASELINE {key}: {len(merged)} task(s) on disk, {len(left)} still missing")
    return not left


# --------------------------------------------------------------------------
# intervention jobs
# --------------------------------------------------------------------------
MAX_TASK_ATTEMPTS = 3
MIN_FREE_GB = 5  # a full-reroll cell writes whole transcripts; stop before the disk does


def free_gb():
    st = os.statvfs(SWEEP)
    return st.f_bavail * st.f_frsize / 1e9


def attempts_path(key, intervenor):
    return os.path.join(SWEEP, key, f".attempts_{intervenor}.json")


def bump_attempts(key, intervenor, task_ids):
    """Count how often each task has been attempted so a task that fails
    deterministically cannot trap the driver in an endless retry loop."""
    p = attempts_path(key, intervenor)
    counts = load(p) if os.path.exists(p) else {}
    for t in task_ids:
        counts[str(t)] = counts.get(str(t), 0) + 1
    dump(p, counts)
    return counts


def exhausted(key, intervenor):
    p = attempts_path(key, intervenor)
    counts = load(p) if os.path.exists(p) else {}
    return {int(t) for t, c in counts.items() if c >= MAX_TASK_ATTEMPTS}


def intervention_state(key, intervenor, readonly=False):
    """Return (target_folder|None, covered_task_ids, all_task_ids).

    Baseline tasks with no trajectory cannot be intervened on, and tasks that
    have already failed MAX_TASK_ATTEMPTS times are treated as terminal. readonly
    never writes, since status() may run while run.py is appending to this file.
    """
    bdir = os.path.join(SWEEP, key)
    all_ids = {r["task_id"] for r in load(os.path.join(bdir, "transcript.json")) if r.get("traj")}
    all_ids -= exhausted(key, intervenor)
    files = iv_files(key, intervenor)
    if not files:
        return None, set(), all_ids
    target = files[0]
    d = load_tolerant(target, repair=not readonly)
    bad = {r["task_id"] for r in d if is_err(r)}
    if bad and not readonly:  # drop errored tasks entirely so they are retried cleanly
        dump(target, [r for r in d if r["task_id"] not in bad])
        d = load_tolerant(target)
    trusted = history_ids(os.path.dirname(target))
    covered = {t for t, rows in by_task(d).items() if settled(rows, t in trusted)}
    return os.path.dirname(target), covered, all_ids


def run_intervention_job(key, meta, intervenor):
    bdir = os.path.join(SWEEP, key)
    if live_run(key, intervenor):
        log(f"INTERV {key} x {intervenor}: a run.py for this cell is still alive, skipping")
        return False
    fold_topups(key, intervenor)  # orphan top-up of a run killed before its merge
    target, covered, all_ids = intervention_state(key, intervenor)
    missing = sorted(all_ids - covered)
    if not missing:
        return True
    log(f"INTERV {key} x {intervenor}: {len(missing)}/{len(all_ids)} task(s) missing -> running")
    cmd = [PY, "run.py", "--run_intervention", "--baseline_path", bdir,
           "--model", meta["model"], "--model-provider", meta["model_provider"],
           "--user-model", meta["user_model"], "--user-model-provider", meta["user_model_provider"],
           "--env", meta["env"], "--agent-strategy", "react-intervened",
           "--intervention_model", intervenor, "--intervention-model-provider", "azure_ai",
           "--search_model", SEARCH_MODEL, "--search-model-provider", SEARCH_PROVIDER,
           "--intervenor-temperature", IV_TEMP, "--temperature", str(meta["temperature"]),
           "--best_of_N", N, "--max-concurrency", concurrency_for(meta, intervenor)[0]]
    if covered:  # top-up run: only the gap
        cmd += ["--task-ids", *map(str, missing)]
    rc = run_cmd(cmd, os.path.join(bdir, f"interv_{intervenor}.log"),
                 attempt_concurrency=concurrency_for(meta, intervenor)[1])
    log(f"INTERV {key} x {intervenor}: run rc={rc}")

    fold_topups(key, intervenor)
    _, covered, all_ids = intervention_state(key, intervenor)
    left = sorted(all_ids - covered)
    if rc in (0, STALL_RC) and left:
        # only count against the retry budget when the run completed or was killed
        # for stalling; a crashed process (bad env, auth) says nothing about the tasks
        bump_attempts(key, intervenor, left)
    log(f"INTERV {key} x {intervenor}: {len(covered)}/{len(all_ids)} covered, {len(left)} still missing")
    return not left


# --------------------------------------------------------------------------
# control jobs: the no-intervenor baselines the paper table reports alongside
# the intervenor cells (Reflexion, Full RR, Partial RR). Run for any baseline
# marked "controls": true in baselines.json.
# --------------------------------------------------------------------------
CONTROLS = ("reflexion", "full_rr", "partial_rr")
REFLECTION_WRITER = "gpt-4o"   # same reflection writer as the published Reflexion rows
REROLL_TEMP = "1.0"            # partial reroll relies on sampling noise; 0.0 would repeat the run
BASE_SEED = 10                 # seed the staged baselines were run with


def control_file(key, control, model):
    if control == "reflexion":
        pat = os.path.join(SWEEP, key, f"reflexion-by_{REFLECTION_WRITER}_*", "reflexion-transcripts.json")
    else:
        pat = os.path.join(SWEEP, key, f"partial-reroll-by_{model}_*", "partial-reroll-transcripts.json")
    fs = sorted(glob.glob(pat), key=os.path.getmtime)
    return fs[-1] if fs else None


def control_state(key, meta, control):
    """(covered, all_ids) for one control run, in the units that control retries.

    Reflexion and partial reroll retry per failing task and stop early on success,
    so a task is covered once it succeeded or used its N attempts. Full reroll
    instead needs every task of every extra seed, so its units are seeds.
    """
    bdir = os.path.join(SWEEP, key)
    base = [r for r in load(os.path.join(bdir, "transcript.json")) if r.get("traj")]
    if control == "full_rr":
        all_tasks = {r["task_id"] for r in base}
        seeds = range(BASE_SEED + 1, BASE_SEED + int(N))
        covered = set()
        for seed in seeds:
            done = set()
            for f in glob.glob(os.path.join(bdir, "full_rr_runs", f"*_seed-{seed}_*", "transcript.json")):
                done |= {r["task_id"] for r in load_tolerant(f, repair=False) if r.get("traj") and not is_err(r)}
            if not all_tasks - done:
                covered.add(seed)
        return covered, set(seeds) - exhausted(key, f"control_{control}")
    fails = {r["task_id"] for r in base if r["reward"] != 1.0}
    fails -= exhausted(key, f"control_{control}")
    f = control_file(key, control, meta["model"])
    if not f:
        return set(), fails
    rows = load_tolerant(f, repair=False)
    # an errored attempt still spends part of the task's budget: the control scripts
    # count it, so counting only clean rows here would ask for attempts they refuse
    # to run and leave a gap that never closes
    covered = {t for t, entries in by_task(rows).items()
               if any(e["reward"] == 1.0 and not is_err(e) for e in entries) or len(entries) >= int(N)}
    return covered & fails, fails


def run_control_job(key, meta, control):
    bdir = os.path.join(SWEEP, key)
    if control == "full_rr":
        # run_baseline_n_times skips any task already in a seed's checkpoint, errored
        # ones included, so a seed with an errored task can never complete. Drop those
        # rows and the rerun picks the task up.
        for f in glob.glob(os.path.join(bdir, "full_rr_runs", "*", "transcript.json")):
            d = load_tolerant(f, repair=False)
            keep = [r for r in d if r.get("traj") and not is_err(r)]
            if len(keep) != len(d):
                dump(f, keep)
                log(f"CONTROL {key} x full_rr: dropped {len(d) - len(keep)} errored row(s) from {os.path.basename(os.path.dirname(f))}")
    covered, all_ids = control_state(key, meta, control)
    missing = sorted(all_ids - covered)
    if not missing:
        return True
    log(f"CONTROL {key} x {control}: {len(missing)}/{len(all_ids)} missing -> running")
    tasks, attempts = concurrency_for(meta)
    common = ["--model", meta["model"], "--model-provider", meta["model_provider"],
              "--user-model", meta["user_model"], "--user-model-provider", meta["user_model_provider"],
              "--env", meta["env"], "--max-concurrency", tasks]
    if control == "reflexion":
        cmd = [PY, "run_reflexion.py", "--baseline_path", bdir, *common,
               "--agent-strategy", "react-reflexion",
               "--intervention_model", REFLECTION_WRITER, "--intervention-model-provider", "azure",
               "--temperature", REROLL_TEMP, "--best_of_N", N, "--task-ids", *map(str, missing)]
    elif control == "partial_rr":
        cmd = [PY, "run_partial_reroll.py", "--baseline_path", bdir, *common,
               "--agent-strategy", "react-intervened",
               "--intervention_model", "DeepSeek-V3.2", "--intervention-model-provider", "azure_ai",
               "--search_model", SEARCH_MODEL, "--search-model-provider", SEARCH_PROVIDER,
               "--temperature", REROLL_TEMP, "--best_of_N", N, "--task-ids", *map(str, missing)]
    else:  # full_rr: the script resumes each seed folder itself, so it takes no task list
        cmd = [PY, "run_baseline_n_times.py", "--input_path", bdir, *common,
               "--agent-strategy", "react", "--num-trials", N, "--seed", str(BASE_SEED),
               "--temperature", str(meta["temperature"]), "--log-dir", os.path.join(bdir, "full_rr_runs")]
    rc = run_cmd(cmd, os.path.join(bdir, f"control_{control}.log"), attempt_concurrency=attempts)
    log(f"CONTROL {key} x {control}: run rc={rc}")
    covered, all_ids = control_state(key, meta, control)
    left = sorted(all_ids - covered)
    if rc == 0 and left:
        # a stall kill is not evidence a task cannot finish: controls resume per
        # task and each pass makes real progress, so only a clean run that still
        # leaves gaps counts against a task's budget
        bump_attempts(key, f"control_{control}", left)
    log(f"CONTROL {key} x {control}: {len(covered)}/{len(all_ids)} covered, {len(left)} still missing")
    return not left


# --------------------------------------------------------------------------
def job_list():
    """Ordered work plan: new baselines first, then airline, then retail."""
    meta = baselines()
    jobs = []
    for key, m in meta.items():
        if not m.get("skip") and not m.get("staged"):
            jobs.append(("baseline", key, m, None))
    for env in ("airline", "retail"):
        for key, m in meta.items():
            if m["env"] != env or m.get("skip"):
                continue
            for iv in m.get("intervenors", INTERVENORS):
                jobs.append(("intervention", key, m, iv))
    for key, m in meta.items():
        if m.get("skip") or not m.get("controls"):
            continue
        # "controls": true runs all of them; a list runs only those named
        wanted = CONTROLS if m["controls"] is True else tuple(m["controls"])
        for c in wanted:
            jobs.append(("control", key, m, c))
    return jobs


def status():
    meta = baselines()
    rows = []
    for key, m in meta.items():
        if m.get("skip"):
            print(f"-- {key}: skipped ({m['skip'].split(':')[0]})")
            continue
        if not m.get("staged"):
            miss = baseline_missing(key, m)
            rows.append((f"{key} [BASELINE]", "", m["tasks"] - len(miss), m["tasks"]))
        if not os.path.exists(os.path.join(SWEEP, key, "transcript.json")):
            continue
        for iv in m.get("intervenors", INTERVENORS):
            _, covered, all_ids = intervention_state(key, iv, readonly=True)
            rows.append((key, iv, len(covered), len(all_ids)))
        if m.get("controls"):
            for c in (CONTROLS if m["controls"] is True else tuple(m["controls"])):
                covered, all_ids = control_state(key, m, c)
                rows.append((key, f"[{c}]", len(covered), len(all_ids)))
    done = sum(1 for r in rows if r[2] >= r[3] and r[3])
    print(f"{'baseline':26} {'intervenor':18} {'covered':>12}")
    for k, iv, c, t in rows:
        mark = "OK " if (t and c >= t) else "-> "
        print(f"{mark}{k:24} {iv:18} {c:>5}/{t}")
    print(f"\n{done}/{len(rows)} jobs complete")


def main():
    if "--status" in sys.argv:
        return status()
    for kind, key, m, iv in job_list():
        if free_gb() < MIN_FREE_GB:
            # a job that runs out of disk loses its work and leaves half-written
            # folders; pause the pass instead and let the next one pick up
            log(f"LOW DISK: {free_gb():.1f} GB free (< {MIN_FREE_GB}), pausing before {kind} {key} x {iv}")
            return
        try:
            if baselines().get(key, {}).get("skip"):
                continue   # re-read: a cell held mid-pass should not keep running
            if kind == "control":
                run_control_job(key, m, iv)
            elif kind == "baseline":
                ok = run_baseline_job(key, m)
                if not ok:
                    log(f"BASELINE {key}: incomplete, will retry on next pass")
            else:
                # re-read: a baseline skipped mid-pass should not run its queued jobs
                if baselines().get(key, {}).get("skip"):
                    continue
                if not os.path.exists(os.path.join(SWEEP, key, "transcript.json")):
                    log(f"INTERV {key} x {iv}: baseline not ready, skipping for now")
                    continue
                run_intervention_job(key, m, iv)
        except Exception as e:
            log(f"ERROR on {kind} {key} x {iv}: {type(e).__name__}: {e}")
    log("PASS COMPLETE")


if __name__ == "__main__":
    main()
