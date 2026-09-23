#!/bin/zsh
# Keeps the intervenor sweep making progress across laptop shutdowns and sleeps.
# Launched hourly by launchd; the driver itself is idempotent, so a restart just
# resumes whatever was left unfinished.
set -u
REPO=/Users/hanavmodasiya/tau-bench
SWEEP=$REPO/results/intervenor_sweep
LOCK=$SWEEP/.sweep.lock
LOG=$SWEEP/supervisor.log
mkdir -p $SWEEP

stamp() { date '+%F %T' }

if [[ -f $LOCK ]]; then
  PID=$(cat $LOCK 2>/dev/null)
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "$(stamp) already running (pid $PID) - nothing to do" >> $LOG
    exit 0
  fi
  echo "$(stamp) stale lock (pid ${PID:-none} gone) - resuming sweep" >> $LOG
fi

cd $REPO || exit 1
if [[ ! -f $REPO/.azure_foundry.env ]]; then
  echo "$(stamp) FATAL: .azure_foundry.env missing" >> $LOG
  exit 1
fi
source $REPO/.azure_foundry.env

echo $$ > $LOCK
echo "$(stamp) starting sweep driver (pid $$)" >> $LOG
# caffeinate -i prevents *idle* sleep for as long as the driver runs, so a
# multi-hour job is not suspended halfway; closing the lid still sleeps.
caffeinate -i $REPO/.venv/bin/python $REPO/run_intervenor_sweep.py >> $SWEEP/driver.log 2>&1
rc=$?
echo "$(stamp) driver exited rc=$rc" >> $LOG
rm -f $LOCK
exit $rc
