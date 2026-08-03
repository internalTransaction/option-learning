# A股指数ETF期权 · 波动率择时与仓位系统

用 A 股宽基 **ETF 期权的波动率曲面**刻画市场恐慌/贪婪,为**个股组合做仓位管理**(择时加减仓,非做空),
并研究期权买方/对冲的时机。核心定位:**期权数据是"质量过滤器",不是方向预测器**——
它擅长在别人恐慌时判断"这波跌该不该抄",而不是预测涨跌。

研究标的:沪深300ETF(510300)、中证1000(000852)、科创50ETF(588000)、创业板ETF(159915)。
数据源:Tushare 历史期权链 → 重建波动率曲面。分析窗口默认聚焦 **2024-09-24(924 行情起点)** 之后。

## 报告(自包含 HTML,`reports/`)

先开 **`期权择时系统_总览与导航.html`** —— 整条逻辑链、参数、研究依据的导航页,并链到:

| 报告 | 内容 |
|------|------|
| **波动率择时信号台** | 主图价格 + 6 个分位副图 + GEX + 建议仓位 + **今日期权动作提示**;十字光标逐日读数;恐慌灯/melt-up 触发带。4 标的、924以来/全部可切。 |
| **策略净值与择时超额** | **双策略对比**(纯期权择时 / HVWMA结合) + 净值/仓位/超额/回撤子图;vs 满仓、vs 等均仓恒定;含 5bps 成本。 |
| **恐慌择时警告 · 研究总结** | 五灯极值口径与事件复盘的早期研究总结。 |

## 核心结论速记

- **期权信号是条件性 alpha**:五灯单独无领先性(没跌之前假信号多),但控制跌幅后有强增量(中证1000 有灯 vs 无灯胜率 40%→77%)。跌幅=触发器、期权=质量过滤器。
- **擅抄底、不擅逃顶**(波动率不对称:底有 capitulation 尖峰、顶是低波 grind)。逃顶只做降仓,不反向开空。
- **动态阈值(滚动分位)> hardcode**;**无底仓 + 趋势持有** 让均仓自然升、超额不降;**GEX 正交增量**(低 GEX 恐慌抄底反弹更猛),只作 regime 择时,A股无 pin。
- **对冲**:过热(高IV)买 put 是灾难(方向错+买贵+IV回落三杀);正解是**低 IV(平静)时买 2 周 put 保险**(创业板/沪深300 最优、赔率才匹配);对冲科创下行用**做空中证1000(IM 期货)或直接减仓**;期权买方主场在**恐慌抄底买 call**。
- **模型保持简单**:连续指数 ≈ 离散三档,ML 仅用 Lasso 做因子筛选,别上 XGBoost。

## 快速开始

```bash
pip install -r requirements.txt

# 一键重算数据 + 注入所有 HTML 报告(build_gex → build_timing_viz → build_equity → inject)
python scripts/refresh_reports.py

# 拉取/更新上游原始数据(需 Tushare token: config/tushare_token.txt)
python -m scripts.fetch_data --refresh
```

HVWMA 趋势引擎目前用日线近似;精确 3H 版需 60min 分钟数据(`scripts/fetch_intraday.py`,受 `stk_mins` 限频)。

## 脚本(`scripts/`)

- **数据管线**:`build_surface`(期权链→曲面)· `build_gex`(dealer GEX + max-OI)· `build_timing_viz`(可视化数据,含 GEX 并入)· `build_equity`(双策略净值/超额)· `refresh_reports`(一键重算+注入)
- **研究**:`ablation_lights_vs_drawdown`(灯×跌幅消融)· `topping_dynamic/second_order`(逃顶,证伪)· `exit_logic/exit_trailing`(出场)· `floor_sweep`(底仓权衡)· `gex_enhance/gex_position`(GEX 强化+接入)· `hvwma_strategy/build_hvwma_3h`(HVWMA 趋势引擎)
- **对冲研究**:`put_on_overheat`(过热买put)· `cross_hedge`(代理品种对冲)· `cross_put_hedge`(跨品种 put)· `lowvol_put`(低IV买put时机)· `put_hedge_moneyness`(虚值凸性)

## 免责

研究性总结。所有胜率/夏普/超额基于历史样本回测(含 5bps 成本),样本有限(924 后约 21 个月、以 V 型急反弹为主),
极值样本成簇、小样本高胜率需打折看待。**不构成投资建议。**
