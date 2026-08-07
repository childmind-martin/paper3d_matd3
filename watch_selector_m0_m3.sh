#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-selector_m0_m3_env4_seed101_v10}"
FOLLOW="${FOLLOW:-1}"
LINES="${LINES:-80}"
UNIT_LINES="${UNIT_LINES:-25}"
SHOW_PIPELINE_TAIL="${SHOW_PIPELINE_TAIL:-1}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[STOP] RUN_ID 非法: $RUN_ID" >&2
  exit 2
fi
if [[ "$FOLLOW" != "0" && "$FOLLOW" != "1" ]]; then
  echo "[STOP] FOLLOW 只能是 0 或 1" >&2
  exit 2
fi
if [[ "$SHOW_PIPELINE_TAIL" != "0" && "$SHOW_PIPELINE_TAIL" != "1" ]]; then
  echo "[STOP] SHOW_PIPELINE_TAIL 只能是 0 或 1" >&2
  exit 2
fi
if [[ ! "$LINES" =~ ^[1-9][0-9]*$ ]]; then
  echo "[STOP] LINES 必须是正整数" >&2
  exit 2
fi
if [[ ! "$UNIT_LINES" =~ ^[1-9][0-9]*$ ]]; then
  echo "[STOP] UNIT_LINES 必须是正整数" >&2
  exit 2
fi

STATE_DIR="$REPO_ROOT/selector_experiment_runs/$RUN_ID"
STATUS_FILE="$STATE_DIR/status.txt"
LOG_FILE="$STATE_DIR/pipeline.log"

echo "============================================================"
echo "[Selector Status] time=$(date --iso-8601=seconds)"
echo "[Selector Status] run_id=$RUN_ID"
echo "[Selector Status] state_dir=$STATE_DIR"

pipeline_pid=""
pipeline_running=0
if [ -f "$STATUS_FILE" ]; then
  echo
  echo "[阶段状态]"
  sed 's/^/  /' "$STATUS_FILE"
  pipeline_pid="$(sed -n 's/^pipeline_pid=//p' "$STATUS_FILE" | tail -n 1)"
  if [[ "$pipeline_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pipeline_pid" 2>/dev/null; then
    pipeline_running=1
  fi
else
  echo
  echo "[阶段状态] status.txt 尚不存在"
fi

echo
if [ "$pipeline_running" -eq 1 ]; then
  echo "[主进程] RUNNING pid=$pipeline_pid"
  ps -p "$pipeline_pid" -o pid=,ppid=,stat=,etime=,cmd=
else
  echo "[主进程] NOT RUNNING"
fi

echo
echo "[GPU 计算进程]"
gpu_processes="$(
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true
)"
if [ -n "$gpu_processes" ]; then
  printf '%s\n' "$gpu_processes" | sed 's/^/  /'
else
  echo "  未返回独立计算进程 PID（WSL2 下可能不提供该列表）"
fi
gpu_summary="$(
  nvidia-smi \
    --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || true
)"
if [ -n "$gpu_summary" ]; then
  echo "[GPU 利用率/显存]"
  printf '%s\n' "$gpu_summary" | sed 's/^/  /'
else
  echo "[GPU 利用率/显存] 无法读取 nvidia-smi"
fi

active_unit_log=""
active_unit_kind=""
candidate_records=""
for stamp_name in formal_parent_stamp.txt pilot_parent_stamp.txt; do
  stamp_file="$STATE_DIR/$stamp_name"
  if [ ! -f "$stamp_file" ]; then
    continue
  fi
  stamp="$(sed -n '1p' "$stamp_file")"
  if [[ ! "$stamp" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
    continue
  fi
  parent_dir="$REPO_ROOT/ablation_experiments/multi_seed_groupB_${stamp}"
  if [ ! -d "$parent_dir" ]; then
    continue
  fi
  candidate_records+="$(
    find "$parent_dir" -type f \
      \( -path '*/launcher_logs/*.log' -o -name 'post_eval.log' -o -name 'worker.log' \) \
      -printf '%T@|%p\n'
  )"$'\n'
done

for eval_root in \
  "$REPO_ROOT/evaluation_results_${RUN_ID}_pilot" \
  "$REPO_ROOT/evaluation_results_${RUN_ID}_formal_gpu_v10"
do
  if [ ! -d "$eval_root" ]; then
    continue
  fi
  candidate_records+="$(
    find "$eval_root" -type f \
      \( -name 'post_eval.log' -o -name 'worker.log' \) \
      -printf '%T@|%p\n'
  )"$'\n'
done

latest_candidate="$(
  printf '%s' "$candidate_records" \
    | sed '/^[[:space:]]*$/d' \
    | sort -t '|' -k1,1nr \
    | head -n 1
)"
if [ -n "$latest_candidate" ]; then
  active_unit_log="${latest_candidate#*|}"
  case "$active_unit_log" in
    */launcher_logs/*) active_unit_kind="训练 launcher" ;;
    */post_eval.log) active_unit_kind="训练后评估" ;;
    */worker.log) active_unit_kind="正式评估分片" ;;
    *) active_unit_kind="运行单元" ;;
  esac
fi

if [ -n "$active_unit_log" ]; then
  latest_progress="$(
    tr '\r' '\n' < "$active_unit_log" \
      | grep -E \
        '回合 [0-9]+/[0-9]+:|训练进度 .* [0-9]+/[0-9]+|\[EvalTiming\] episode=[0-9]+/[0-9]+|[Ee]pisode[ =][0-9]+/[0-9]+' \
      | tail -n 1 || true
  )"
  if [[ "$latest_progress" == *"[EvalTiming]"* ]]; then
    active_unit_kind="训练后/正式评估"
  fi
  echo
  echo "[当前${active_unit_kind}] $active_unit_log"
  echo "[日志更新时间] $(stat -c '%y' "$active_unit_log")"
  if [ -n "$latest_progress" ]; then
    echo "[最新进度] $latest_progress"
  else
    echo "[最新进度] 尚在初始化或尚未完成第 1 个回合"
  fi
  echo "[运行单元最近日志]"
  tail -n "$UNIT_LINES" "$active_unit_log"
fi

echo
echo "[最近日志] $LOG_FILE"
echo "============================================================"
if [ ! -f "$LOG_FILE" ]; then
  echo "日志尚未创建。"
  exit 1
fi

if [ "$FOLLOW" -eq 1 ] && [ "$pipeline_running" -eq 1 ]; then
  echo "[持续刷新] 每 5 秒重新定位当前训练/评估日志；Ctrl-C 退出监控，不会停止实验。"
  while kill -0 "$pipeline_pid" 2>/dev/null; do
    sleep 5
    if [ -t 1 ]; then
      printf '\033[H\033[2J'
    fi
    RUN_ID="$RUN_ID" \
    FOLLOW=0 \
    LINES="$LINES" \
    UNIT_LINES="$UNIT_LINES" \
    SHOW_PIPELINE_TAIL=0 \
      bash "$REPO_ROOT/watch_selector_m0_m3.sh" || true
    current_state="$(
      sed -n 's/^state=//p' "$STATUS_FILE" 2>/dev/null | tail -n 1
    )"
    if [ "$current_state" != "running" ]; then
      break
    fi
  done
elif [ "$SHOW_PIPELINE_TAIL" -eq 1 ]; then
  tail -n "$LINES" "$LOG_FILE"
fi
