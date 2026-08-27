#!/usr/bin/env bash
# 期权择时信号台 · 盘中 HVWMA 刷新(轻量)。
# 3H HVWMA 只在每根 60min bar 收完时变化(10:30/11:30/14:00/15:00),
# 故 cron 排每根 bar 收完后 5 分钟跑, 更高频无信息量。
# 只刷: hvwma_live(新浪60min) → 重汇总信号台(因子/仓位仍是EOD值) → 推 webroot。
# 不碰 tushare、不动曲面/timing_viz。全链 ~20s。
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
WEBROOT="/var/www/cnstock_html"
LOG="$ROOT/logs/intraday.log"
mkdir -p "$ROOT/logs"

log(){ echo "$(TZ=Asia/Shanghai date '+%F %T') | $*" >> "$LOG"; }

log "── intraday 开始"
if ! "$PY" scripts/build_hvwma_live.py >>"$LOG" 2>&1; then
  log "✗ hvwma_live 失败, 中止(保留上次值)"; exit 1
fi
if ! "$PY" scripts/build_signal_desk.py >>"$LOG" 2>&1; then
  log "✗ signal_desk 失败, 中止"; exit 1
fi
for f in hvwma_live.json option_signal_desk.json; do
  cp "data/processed/$f" "$WEBROOT/$f.tmp" && mv "$WEBROOT/$f.tmp" "$WEBROOT/$f"
done
DIRS=$("$PY" -c "import json;d=json.load(open('data/processed/hvwma_live.json'));print(' '.join(f\"{k}:{'G' if v['dir']>0 else 'R'}\" for k,v in d.items()))" 2>/dev/null)
log "✓ intraday 完成  $DIRS"
