#!/usr/bin/env bash
# 期权择时信号台 · 每日增量更新 → 推 cnstock webroot。
# 交易日收盘后跑(opt_daily 结算约傍晚可得)。flock 防重入, 全程记日志。
#   1. 增量补期权曲面(只补缺失交易日)
#   2. 重算入场因子 timing_viz
#   3. 刷新 live HVWMA 趋势态(akshare, 对齐实盘3H)
#   4. 汇总研究向信号台 JSON
#   5. 拷贝到 /var/www/cnstock_html/ (nginx: cnstock.k2database.xyz)
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
WEBROOT="/var/www/cnstock_html"
LOG="$ROOT/logs/daily.log"
mkdir -p "$ROOT/logs"
export TUSHARE_TOKEN="$(cat "$ROOT/config/tushare_token.txt" 2>/dev/null)"

log(){ echo "$(TZ=Asia/Shanghai date '+%F %T') | $*" | tee -a "$LOG"; }

log "===== daily_update 开始 ====="

step(){ # step "名称" cmd...
  local name="$1"; shift
  log "▶ $name"
  if "$@" >>"$LOG" 2>&1; then log "  ✓ $name"; else log "  ✗ $name 失败(退出码 $?), 中止"; exit 1; fi
}

# 1. 增量曲面(--no-reports: 只补数据不跑旧HTML报告)
step "增量补曲面" "$PY" scripts/update_latest_data.py --no-reports
# 1b. GEX 高阶 greeks(非致命: 失败则沿用上日 gex_*.json, timing_viz 自动对齐缺失=None)
log "▶ 重算 GEX"; "$PY" scripts/build_gex.py >>"$LOG" 2>&1 && log "  ✓ GEX" || log "  ⚠ GEX 异常(沿用上日值继续)"
# 2. 入场因子
step "重算 timing_viz" "$PY" scripts/build_timing_viz.py
# 3. live HVWMA(akshare 失败会自动回退日线, 不阻塞)
log "▶ 刷新 live HVWMA"; "$PY" scripts/build_hvwma_live.py >>"$LOG" 2>&1 && log "  ✓ live HVWMA" || log "  ⚠ live HVWMA 异常(用回退值继续)"
# 4. 信号台汇总(v1 出场机)
step "汇总信号台" "$PY" scripts/build_signal_desk.py
# 4b. v2 数据层: 上游新库双策略净值 + 今日动作 + 统一口径(非致命, 失败不拖垮 v1)
log "▶ 双策略净值 + v2 数据层"
if "$PY" scripts/build_equity.py >>"$LOG" 2>&1 && "$PY" scripts/build_desk_v2.py >>"$LOG" 2>&1; then
  log "  ✓ v2 数据层"
else
  log "  ⚠ v2 数据层异常(沿用上次 option_desk_v2.json)"
fi

# 5. 推 webroot(原子替换: 先写 .tmp 再 mv)
for f in option_signal_desk.json hvwma_live.json option_desk_v2.json; do
  if [ -f "data/processed/$f" ]; then
    cp "data/processed/$f" "$WEBROOT/$f.tmp" && mv "$WEBROOT/$f.tmp" "$WEBROOT/$f" && log "  ✓ 发布 $f"
  fi
done

# 6. 定向跟进原作者上游展示文件(只拉研究总结, 管线代码钉死不动), 网络失败不阻塞
#    origin 固定为自有恢复仓库；原作者仓库使用 upstream remote。
( git fetch -q "${OPTION_LEARNING_UPSTREAM_REMOTE:-upstream}" 2>/dev/null && \
  git checkout -q "${OPTION_LEARNING_UPSTREAM_REMOTE:-upstream}/main" -- \
    "reports/A股股指ETF期权恐慌择时警告_研究总结.html" 2>/dev/null \
  && log "  ✓ 上游展示文件已同步" ) || log "  ⚠ 上游同步跳过(网络/无更新)"

# 7. 原始信号台数据(独立发布, 前端 CnOptionRawPage 运行时 fetch /option_timing_data.json;
#    原静态 HTML 已退役, 视图迁入 SPA tab) + 研究总结 → webroot
if [ -f data/processed/timing_viz.json ]; then
  cp data/processed/timing_viz.json "$WEBROOT/option_timing_data.json.tmp" \
    && mv "$WEBROOT/option_timing_data.json.tmp" "$WEBROOT/option_timing_data.json" \
    && log "  ✓ 发布 option_timing_data.json" || log "  ⚠ option_timing_data.json 发布失败"
else
  log "  ⚠ timing_viz.json 缺失, 跳过原始信号台数据发布"
fi
cp reports/A股股指ETF期权恐慌择时警告_研究总结.html "$WEBROOT/option_research_notes.html" 2>/dev/null || true

ASOF=$("$PY" -c "import json;print(json.load(open('data/processed/option_signal_desk.json'))['meta']['asof'])" 2>/dev/null)
log "===== daily_update 完成 (asof=$ASOF) ====="
