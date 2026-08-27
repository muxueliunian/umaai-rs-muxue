#!/usr/bin/env bash
# MCTS search_n 收益曲线 + UCB 对照扫描
#
# 目的（两个问题一次回答）：
#   Q1 搜索质量随 search_n 是否饱和？—— 看 uniform 档 256/512/1024 的增量
#   Q2 UCB 在 512+ 是否净胜？—— 同 n 下 ucb 与 uniform 的配对差
#
# 已知（本地 24 核 10 对实测，仅供估成本，服务器会重测）：
#   n=128 → 1100 核·秒/局，n=256 → 2218；ucb 约为同 n uniform 的 0.57 倍。
#
# 用法：
#   bash scripts/mcts_sweep.sh calib     # 先跑校准，打印全量预估，**不要跳过**
#   RUNS=20 bash scripts/mcts_sweep.sh full
#
# 输出：logs/sweep_<config>/mcts_panel_results.csv + logs/sweep_<config>.log
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="${RUNS:-20}"          # 每个 build 的局数；总局数 = RUNS × 2
SEED="${SEED:-61444}"
GROUP="${GROUP:-32}"        # UCB 每轮追加批量，必须远小于 search_n
MODE="${1:-calib}"

NPROC="$(nproc)"
echo "== 环境 =="
echo "核数: $NPROC"
rustc --version
cargo --version
echo "（edition 2024 需要 rustc >= 1.85）"
echo

cargo build --release --bin mcts_panel_probe

# run <名字> <MP_TRAINER> <search_n> <ucb> <runs>
run() {
  local name="$1" tr="$2" n="$3" ucb="$4" runs="$5"
  local out="logs/sweep_${name}"
  echo "---- ${name}: trainer=${tr} search_n=${n} ucb=${ucb} runs=${runs}/build ----"
  local t0; t0=$(date +%s)
  MP_TRAINER="$tr" MP_SEARCH_N="$n" MP_RF=0 MP_UCB="$ucb" MP_GROUP="$GROUP" \
  MP_RUNS="$runs" MP_SEED="$SEED" MP_STAGES=train,ramen MP_OUT="$out" \
    ./target/release/mcts_panel_probe 2>&1 | tee "logs/sweep_${name}.log"
  local t1; t1=$(date +%s)
  local wall=$((t1 - t0))
  local games=$((runs * 2))
  echo ">> ${name}: 墙钟 ${wall}s，$((wall * NPROC / games)) 核·秒/局" | tee -a "logs/sweep_${name}.log"
  echo
}

mkdir -p logs

if [ "$MODE" = "calib" ]; then
  # 校准局数必须让「总 job 数 >= 核数」，否则核跑不满、核·秒/局被高估
  # （本地 24 核上 4 job 量出 3132，铺满时实为 2218，高估 1.4 倍）
  CALIB_RUNS=$(( (NPROC + 1) / 2 ))
  [ "$CALIB_RUNS" -lt 2 ] && CALIB_RUNS=2
  echo "校准局数 ${CALIB_RUNS}/build（总 job $((CALIB_RUNS * 2)) >= 核数 ${NPROC}）"
  run calib_u256 mcts 256 0 "$CALIB_RUNS"
  CS=$(grep -o '[0-9]\+ 核·秒/局' "logs/sweep_calib_u256.log" | tail -1 | grep -o '[0-9]\+')
  echo "== 全量预估（RUNS=${RUNS}）=="
  echo "以 u256 = ${CS} 核·秒/局 为锚，按 search_n 线性、ucb 取 0.57 倍外推："
  TOT=0
  for spec in "u256 256 1" "u512 512 1" "u1024 1024 1" "ucb256 256 0.57" "ucb512 512 0.57" "ucb1024 1024 0.57"; do
    set -- $spec
    C=$(awk -v cs="$CS" -v n="$2" -v f="$3" 'BEGIN{printf "%d", cs*n/256*f}')
    G=$(awk -v c="$C" -v r="$RUNS" 'BEGIN{printf "%d", c*r*2}')
    TOT=$(awk -v t="$TOT" -v g="$G" 'BEGIN{printf "%d", t+g}')
    printf "  %-8s %7d 核·秒/局 → %8.1f 核·时\n" "$1" "$C" "$(awk -v g="$G" 'BEGIN{print g/3600}')"
  done
  echo "  ------------------------------------------"
  printf "  合计 %.1f 核·时 → 本机 %d 核约 %.1f 小时墙钟\n" \
    "$(awk -v t="$TOT" 'BEGIN{print t/3600}')" "$NPROC" \
    "$(awk -v t="$TOT" -v p="$NPROC" 'BEGIN{print t/3600/p}')"
  echo
  echo "把上面这段回报给发起方，确认 RUNS 后再跑：RUNS=<N> bash scripts/mcts_sweep.sh full"
  exit 0
fi

if [ "$MODE" != "full" ]; then echo "未知模式: $MODE（只接受 calib / full）"; exit 1; fi

echo "== 全量扫描 RUNS=${RUNS}/build，共 7 档 =="
run hw      hw   256  0 "$RUNS"      # 手写基线，同种子配对，几乎不耗时
run u256    mcts 256  0 "$RUNS"
run ucb256  mcts 256  1 "$RUNS"
run u512    mcts 512  0 "$RUNS"
run ucb512  mcts 512  1 "$RUNS"
run u1024   mcts 1024 0 "$RUNS"
run ucb1024 mcts 1024 1 "$RUNS"

echo "== 全部完成，回传以下文件 =="
ls -la logs/sweep_*/mcts_panel_results.csv logs/sweep_*.log
