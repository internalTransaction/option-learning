# A股指数ETF期权 · 波动率择时与仓位系统

用 A 股宽基 **ETF 期权的波动率曲面**刻画市场恐慌/贪婪,为**个股组合做仓位管理**(择时加减仓,非做空),
并研究期权买方/对冲的时机。核心定位:**期权数据是"质量过滤器",不是方向预测器**——
它擅长在别人恐慌时判断"这波跌该不该抄",而不是预测涨跌。

研究标的:沪深300ETF(510300)、中证1000(000852)、科创50ETF(588000)、创业板ETF(159915)。
数据源:Tushare 历史期权链 → 重建波动率曲面。分析窗口默认聚焦 **2024-09-24(924 行情起点)** 之后。
另有一条美股线(SPY/QQQ/SOXX,Polygon 数据)复现同一套框架。

## 报告(自包含 HTML,`reports/`)

先开 **`期权择时系统_总览与导航.html`** —— 整条逻辑链、参数、研究依据的导航页,并链到:

| 报告 | 内容 |
|------|------|
| **① 波动率择时信号台** | 主图价格 + 分位副图(含 GEX、vol-of-vol、隐含相关性) + **今日期权动作提示**;十字光标逐日读数;恐慌灯/melt-up 触发带、阴跌未出清警告。4 标的、924以来/全部可切。 |
| **② 策略净值与择时超额** | **双策略对比**(纯期权择时 / HVWMA结合) + 净值/仓位/超额/回撤子图;vs 满仓、vs 等均仓恒定;含 5bps 成本。 |
| **③ 波动概率锥** | SOXX expected-move 口径的**前瞻锥**,给出 −0.5/1/1.5/2σ 的**可挂单价位**;顶点可拉回历史看当时的锥与后来实际走势;附挂单回测。 |
| **④ 美股信号台** | SPY/QQQ/SOXX,**日线/小时可切**,同一套五灯框架的美股复现。观察式仪表盘。 |
| **⑤ 恐慌择时警告 · 研究总结** | 五灯极值口径与事件复盘的早期研究总结。 |
| **⑥ 纯期权抄底凸性策略** | 恐慌灯买 call / 价差的加仓腿,以及这套证据到哪儿为止。 |

## 核心结论速记

- **期权信号是条件性 alpha**:五灯单独无领先性(没跌之前假信号多),但控制跌幅后有强增量(中证1000 有灯 vs 无灯胜率 40%→77%)。跌幅=触发器、期权=质量过滤器。
- **擅抄底、不擅逃顶**(波动率不对称:底有 capitulation 尖峰、顶是低波 grind)。逃顶只做降仓,不反向开空。
- **动态阈值(滚动分位)> hardcode**;**无底仓 + 趋势持有** 让均仓自然升、超额不降;**GEX 正交增量**(低 GEX 恐慌抄底反弹更猛),只作 regime 择时,A股无 pin。
- **vol-of-vol 背离**:IV 高但蝶式凸度低 = 假恐慌(胜率 49% vs 66%),作抄底质量过滤。
- **隐含相关性**:A股只能做半隐含(指数用 IV、成分股用已实现波动率反推)。与美股相反,**A股大涨时相关性比大跌还高**,所以"高相关=恐慌"不成立;真正有用的是警告侧——已跌到位 + 齐跌 + 但期权不慌 = 阴跌未出清。
- **概率锥挂单**:**抄不到底**——成交后平均还要再跌 5%,成交价比真底高约 4%,且挂更深并不少被套。能做的是"买得合理便宜":−0.5σ 几乎无增量,−1.5σ 才拉开(+3.8%/胜率66%)但成交率仅 14%。**恐慌门几乎免费**(panic≥0.7 时 −1σ 成交率仍 33%,但成交后 +5.6%/70%)。
- **对冲**:过热(高IV)买 put 是灾难(方向错+买贵+IV回落三杀);正解是**低 IV(平静)时买 2 周 put 保险**;对冲科创下行用**做空中证1000(IM 期货)或直接减仓**;期权买方主场在**恐慌抄底买 call**。
- **模型保持简单**:连续指数 ≈ 离散三档,ML 仅用 Lasso 做因子筛选,别上 XGBoost。

## 快速开始

```bash
pip install -r requirements.txt

# 一键重算数据 + 注入所有 HTML 报告
# (build_gex → build_timing_viz → build_equity → build_cone_viz → inject)
python scripts/refresh_reports.py

# 拉取/更新上游原始数据(需 Tushare token: config/tushare_token.txt)
python -m scripts.fetch_data --refresh
```

隐含相关性是可选上游(目前仅中证1000),缺失时该副图/读数留空,不影响其余管线:

```bash
python -m scripts.fetch_constituents --start 20220801 --end <最新>
python -m scripts.build_implied_corr --window 21
```

HVWMA 趋势引擎目前用日线近似;精确 3H 版需 60min 分钟数据(`scripts/fetch_intraday.py`,受 `stk_mins` 限频)。

## 部署(静态站点)

报告是自包含 HTML,无构建步骤、无后端。`build_site.py` 会把中文文件名换成 ASCII slug、
重写报告间的链接、并把导航页复制为 `index.html`:

```bash
python scripts/build_site.py          # 产出 site/,约 3.7 MB
```

把 `site/` 的内容放到静态服务器 root 即可(nginx `root /var/www/option-desk;`)。
`site/` 是生成物,已在 `.gitignore` 里。

> 唯一的外网依赖:`cn-panic-research.html` 仍从 jsDelivr 加载 Chart.js。
> 内网/离线部署需要把它 vendor 进去。其余页面零外部请求。

## 脚本(`scripts/`)

- **数据管线**:`build_surface`(期权链→曲面)· `build_gex`(dealer GEX + max-OI)· `build_timing_viz`(可视化数据)· `build_equity`(双策略净值/超额)· `build_cone_viz`(概率锥 + 挂单回测)· `build_implied_corr`(半隐含相关性)· `refresh_reports`(一键重算+注入)· `build_site`(打包静态站点)
- **研究**:`ablation_lights_vs_drawdown`(灯×跌幅消融)· `topping_dynamic/second_order`(逃顶,证伪)· `exit_logic/exit_trailing`(出场)· `floor_sweep`(底仓权衡)· `gex_enhance/gex_position`(GEX)· `hvwma_strategy/build_hvwma_3h`(趋势引擎)· `cone_limit_backtest`(锥下沿挂单)· `period_cone`(周期重置带)· `corr_validate`(相关性验证)· `iv_range_calibration`(±1σ 覆盖率校准)
- **对冲研究**:`put_on_overheat`(过热买put)· `cross_hedge`(代理品种对冲)· `cross_put_hedge`(跨品种 put)· `lowvol_put`(低IV买put时机)· `put_hedge_moneyness`(虚值凸性)

## 免责

研究性总结。所有胜率/夏普/超额基于历史样本回测(含 5bps 成本),样本有限(924 后约 21 个月、以 V 型急反弹为主),
极值样本成簇、小样本高胜率需打折看待;挂单回测样本重叠严重(每天挂单),胜率是方向性参考而非独立试验。
**不构成投资建议。**
