"""把 reports/ 打包成可直接部署的静态站点 site/。

做三件事:
  1. 中文文件名 -> ASCII slug(中文路径在部分服务器/CDN 上会有编码问题)
  2. 重写报告之间的链接, 使其指向新文件名
  3. 导航页复制一份为 index.html 作入口

产物是纯静态、无构建步骤, 直接把 site/ 的内容丢到 nginx 的 root 即可。

用法: python scripts/build_site.py [--out site]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# 中文文件名 -> 部署用的 ASCII 名。导航页额外复制一份 index.html
SLUGS = {
    "期权择时系统_总览与导航.html": "index.html",
    "A股ETF期权_波动率择时信号台.html": "cn-timing-desk.html",
    "A股ETF期权_策略净值超额.html": "cn-equity.html",
    "A股期权_波动概率锥.html": "cn-vol-cone.html",
    "A股股指ETF期权恐慌择时警告_研究总结.html": "cn-panic-research.html",
    "期权抄底凸性策略_框架与证据边界.html": "cn-convexity.html",
    "美股指数期权_波动率择时信号台_合并.html": "us-timing-desk.html",
    "美股指数期权_波动率择时信号台.html": "us-timing-daily.html",
    "美股指数期权_波动率择时信号台_盘中.html": "us-timing-intraday.html",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    args = ap.parse_args()
    out = ROOT / args.out

    missing = [n for n in SLUGS if not (REPORTS / n).exists()]
    if missing:
        sys.exit("✗ reports/ 里缺少: " + ", ".join(missing))

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total = 0
    for src_name, dst_name in SLUGS.items():
        html = (REPORTS / src_name).read_text()
        # 把报告间的中文链接换成 slug
        for a, b in SLUGS.items():
            html = html.replace(f'href="{a}"', f'href="{b}"')
        (out / dst_name).write_text(html)
        kb = len(html.encode()) / 1024
        total += kb
        print(f"  {dst_name:26s} {kb:7.0f} KB   ← {src_name}")

    # 检查有没有漏网的中文链接
    bad = []
    for f in out.glob("*.html"):
        for m in re.findall(r'href="([^"]+\.html)"', f.read_text()):
            if not (out / m).exists():
                bad.append(f"{f.name} -> {m}")
    if bad:
        sys.exit("✗ 断链: " + "; ".join(bad))

    # 提醒仍在依赖外网的资源(国内服务器上会拖慢首屏)
    ext = []
    for f in sorted(out.glob("*.html")):
        for m in re.findall(r'(?:src|href)="(https?://[^"]+)"', f.read_text()):
            ext.append(f"{f.name}: {m}")

    print(f"\n✔ {len(SLUGS)} 个页面 -> {out}  共 {total/1024:.1f} MB")
    print(f"  入口 index.html  链接自检通过")
    if ext:
        print("\n⚠ 仍依赖外网资源(离线/内网环境会退化):")
        for e in ext:
            print("   " + e)


if __name__ == "__main__":
    main()
