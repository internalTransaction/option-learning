# A股指数ETF期权择时信号研究

利用 A 股宽基 **ETF 期权** 蕴含的市场预期信息（隐含波动率、情绪、偏度），
构建对**标的指数的择时信号**并回测其有效性。

研究标的：沪深300ETF(510300)、科创50ETF(588000)、创业板ETF(159915)。
数据源：[AkShare](https://akshare.akfamily.xyz/)（免费，无需注册）。

## 核心逻辑

期权价格里藏着现货看不到的信息，本项目把它们提炼成三类因子并转成择时信号：

| 因子类 | 模块 | 直觉 |
|--------|------|------|
| **波动率** | `factors/volatility.py` | IV 水平/百分位、IV−HV 方差风险溢价、期限结构。恐慌高 IV 常见阶段底，自满低 IV 累积风险。 |
| **情绪/持仓** | `factors/sentiment.py` | PCR（认沽/认购比），成交量 PCR 反映短期情绪，持仓 PCR 反映仓位结构。 |
| **偏度/微笑** | `factors/skew.py` | 虚值 put 相对 call 的 IV 溢价，反映市场为下跌尾部风险付的价。 |

> ⚠️ 信号方向（逆向 vs 顺势）与阈值都是**待回测校验的假设**，不是结论。先搭框架，再用数据证伪/校准。

## 目录结构

```
config/config.yaml       标的、因子、信号、回测参数
src/
  data/                  akshare 数据层 + 本地缓存
  factors/               波动率 / 情绪 / 偏度因子(含 base 基类)
  signals/               因子 -> 择时信号
  backtest/              向量化回测引擎 + 绩效
  utils/                 配置、日志
scripts/
  fetch_data.py          拉取并缓存历史数据
  run_signal.py          端到端: 数据->因子->信号->回测
tests/                   合成数据冒烟测试(不联网)
data/                    缓存(raw/processed, 不入库)
notebooks/               探索性分析
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 拉取历史数据(QVIX 波动率指数 + ETF 日线)
python -m scripts.fetch_data

# 2. 端到端跑一个波动率择时信号并回测(默认沪深300)
python -m scripts.run_signal              # 逆向: 高IV看多
python -m scripts.run_signal --key kc50 --trend

# 3. 冒烟测试(不联网)
python -m pytest -q
```

## 数据说明

| 数据 | 接口 | 性质 |
|------|------|------|
| 隐含波动率指数 QVIX | `index_option_*_qvix` | 历史序列(2015 至今)，波动率因子主干 |
| 标的 ETF 日线 | `fund_etf_hist_em` | 历史序列，算 HV 与回测收益 |
| 全市场期权链 | `option_current_em` | **当日实时快照**，PCR/偏度需按日累积落盘 |

因为 PCR 与偏度依赖逐日期权链快照，建议用定时任务每个交易日收盘后跑
`python -m scripts.fetch_data --snapshot` 累积历史，再启用情绪/偏度因子的时序计算。

## 路线图

- [x] 项目框架、数据层、三类因子接口、波动率择时端到端跑通
- [ ] 定时累积期权链快照，启用 PCR / 偏度时序因子
- [ ] 波动率期限结构因子（近月 vs 次月 IV）
- [ ] 多因子合成与参数网格回测、样本外验证
- [ ] 信号可视化 notebook 与绩效报告
```
