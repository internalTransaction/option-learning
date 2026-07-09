"""一键重跑数据管线 + 注入 HTML 报告。

依赖顺序(不可乱):
    build_gex.py          期权链 → GEX 时序(gex_{key}.json)
    build_timing_viz.py   曲面 + GEX → 可视化数据(timing_viz.json, 内部并入 gex_z, 故须在 build_gex 后)
    build_equity.py       仓位系统 → 净值/超额(equity.json, 依赖 timing_viz.json)
    inject                模板 + 数据 → reports/*.html

用法:
    python scripts/refresh_reports.py

── 上游数据层(半手动, 需 Tushare token, 不在本脚本内)──
    1. python -m scripts.fetch_data --refresh          # 拉取 raw(optbasic/optdaily/fund/qvix)
    2. 重建曲面 parquet(data/processed/surface_<key>_<start>_<end>.parquet)
       ── 注: build_surface.py 目前只重建打印、不落盘; 落盘步骤在 notebook/手动。
       只有原始数据更新到更晚日期时才需要重跑上游; 平时刷新报告直接跑本脚本即可。
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PY = sys.executable

STEPS = ["build_gex.py", "build_timing_viz.py", "build_equity.py"]
INJECT = [
    ("timing_template.html", "timing_viz.json", "/*__TIMING_DATA__*/",
     "A股ETF期权_波动率择时信号台.html"),
    ("equity_template.html", "equity.json", "/*__EQUITY_DATA__*/",
     "A股ETF期权_策略净值超额.html"),
]


def run(script: str) -> None:
    print(f"\n▶ {script}")
    r = subprocess.run([PY, str(ROOT / "scripts" / script)], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"✗ {script} 失败(退出码 {r.returncode})，管线中止。")


def inject(tpl: str, data: str, token: str, out: str) -> None:
    t = (ROOT / "scripts" / tpl).read_text()
    d = (PROC / data).read_text()
    if token not in t:
        sys.exit(f"✗ 模板 {tpl} 缺少占位符 {token}")
    (REPORTS / out).write_text(t.replace(token, d))
    kb = (REPORTS / out).stat().st_size / 1024
    print(f"  ✓ {out}  ({kb:.0f} KB)")


def main() -> None:
    for s in STEPS:
        run(s)
    print("\n▶ 注入 HTML")
    for tpl, data, token, out in INJECT:
        inject(tpl, data, token, out)
    print("\n✔ 全部完成。入口页(静态, 无需注入): "
          "reports/期权择时系统_总览与导航.html")


if __name__ == "__main__":
    main()
