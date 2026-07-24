#!/usr/bin/env bash
set -euo pipefail

config=
checkpoint=
continuations=2
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config=$2; shift 2 ;;
    --checkpoint) checkpoint=$2; shift 2 ;;
    --continuations) continuations=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$config" && -n "$checkpoint" ]] || {
  echo "usage: $0 --config CONFIG --checkpoint CHECKPOINT [--continuations N] [--dry-run]" >&2
  exit 2
}
[[ "$continuations" =~ ^[0-9]+$ ]] || { echo "continuations must be nonnegative" >&2; exit 2; }

submit() {
  if $dry_run; then
    printf 'DRY_RUN'
    printf ' %q' "$@"
    printf '\n'
    echo "dry-run-job"
  else
    "$@"
  fi
}

extra="--config $config"
job=$(submit /opt/pbs/bin/qsub -v "EXTRA=$extra" jobs_next_train.pbs | tail -n 1)
echo "train_job=$job"
for ((index=1; index<=continuations; index++)); do
  extra="--config $config --resume"
  job=$(submit /opt/pbs/bin/qsub -W "depend=afterany:$job" -v "EXTRA=$extra" jobs_next_train.pbs | tail -n 1)
  echo "resume_${index}_job=$job"
done
audit_extra="--config $config --checkpoint $checkpoint --tag development"
audit=$(submit /opt/pbs/bin/qsub -W "depend=afterok:$job" -v "EXTRA=$audit_extra" jobs_next_audit.pbs | tail -n 1)
echo "audit_job=$audit"
