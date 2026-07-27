#!/usr/bin/env bash
set -euo pipefail

config=
checkpoint=
stage=
source_audit=
continuations=2
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config=$2; shift 2 ;;
    --checkpoint) checkpoint=$2; shift 2 ;;
    --stage) stage=$2; shift 2 ;;
    --source-audit) source_audit=$2; shift 2 ;;
    --continuations) continuations=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$config" ]] || {
  echo "usage: $0 --config CONFIG [--stage STAGE] [--checkpoint SOURCE] [--source-audit REPORT] [--continuations N] [--dry-run]" >&2
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

if [[ -n "$stage" ]]; then
  output="_next/checkpoints/bilinear_progressive_${stage}.pt"
  extra="--config $config --stage $stage"
  [[ -z "$checkpoint" ]] || extra+=" --checkpoint $checkpoint"
  [[ -z "$source_audit" ]] || extra+=" --source-audit $source_audit"
else
  [[ -n "$checkpoint" ]] || {
    echo "--checkpoint is required for the schema-4 workflow" >&2
    exit 2
  }
  output=$checkpoint
  extra="--config $config"
fi
job=$(submit /opt/pbs/bin/qsub -v "EXTRA=$extra" jobs_next_train.pbs | tail -n 1)
echo "train_job=$job"
for ((index=1; index<=continuations; index++)); do
  if [[ -n "$stage" ]]; then
    extra="--config $config --stage $stage --resume"
    [[ -z "$checkpoint" ]] || extra+=" --checkpoint $checkpoint"
    [[ -z "$source_audit" ]] || extra+=" --source-audit $source_audit"
  else
    extra="--config $config --resume"
  fi
  job=$(submit /opt/pbs/bin/qsub -W "depend=afterany:$job" -v "EXTRA=$extra" jobs_next_train.pbs | tail -n 1)
  echo "resume_${index}_job=$job"
done
audit_extra="--config $config --checkpoint $output --tag development"
audit=$(submit /opt/pbs/bin/qsub -W "depend=afterok:$job" -v "EXTRA=$audit_extra" jobs_next_audit.pbs | tail -n 1)
echo "audit_job=$audit"
