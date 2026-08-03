"""美股信号台一键刷新: 重建曲面数据 → 注入模板 → 输出 reports/ HTML。

用法:
    python -m scripts.refresh_us_reports              # 日线, 用缓存
    python -m scripts.refresh_us_reports --no-cache   # 强制重新拉取
    python -m scripts.refresh_us_reports hour          # 盘中(小时线, 需模板支持)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PY = sys.executable

TOKEN = "/*__TIMING_DATA__*/"
# gran -> (模板, 数据文件, 输出报告)
CFG = {
    "day":  ("us_timing_template.html", "us_timing_viz.json",
             "美股指数期权_波动率择时信号台.html"),
    "hour": ("us_timing_intraday_template.html", "us_timing_viz_hour.json",
             "美股指数期权_波动率择时信号台_盘中.html"),
}


def build_unified() -> None:
    """把已有的日线+盘中数据注入合并模板(日线/小时可切), 不重建数据。"""
    day_f, hour_f = PROC / "us_timing_viz.json", PROC / "us_timing_viz_hour.json"
    for f in (day_f, hour_f):
        if not f.exists():
            sys.exit(f"✗ 缺少 {f.name}; 请先跑 refresh_us_reports day / hour")
    tpl = (ROOT / "scripts" / "us_timing_unified_template.html").read_text()
    tpl = tpl.replace("/*__DAILY_DATA__*/", day_f.read_text())
    tpl = tpl.replace("/*__HOURLY_DATA__*/", hour_f.read_text())
    out = REPORTS / "美股指数期权_波动率择时信号台_合并.html"
    out.write_text(tpl)
    print(f"  ✓ {out.name}  ({out.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    gran = next((a for a in sys.argv[1:] if not a.startswith("--")), "day")
    if gran == "unified":
        build_unified()
        return
    tpl_name, data_name, out_name = CFG[gran]
    build = [PY, "-m", "scripts.build_us_timing_viz"] + sys.argv[1:]
    print("▶ 重建数据:", " ".join(build[2:]))
    if subprocess.run(build, cwd=ROOT).returncode != 0:
        sys.exit("✗ 数据构建失败")

    tpl = (ROOT / "scripts" / tpl_name).read_text()
    if TOKEN not in tpl:
        sys.exit(f"✗ 模板 {tpl_name} 缺少占位符 {TOKEN}")
    out = REPORTS / out_name
    out.write_text(tpl.replace(TOKEN, (PROC / data_name).read_text()))
    print(f"  ✓ {out.name}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
