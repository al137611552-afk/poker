# 德扑训练台 · 项目须知

本地运行的德州扑克训练与复盘系统。设计方案见
<https://claude.ai/code/artifact/9a1d285c-c5e6-4613-90ef-f83507b04b2b>（要更新就改这个 URL，别新建）。

## 常用命令

```bash
.venv/bin/python -m pytest           # 全量单测（约 38 秒，含服务端与 PokerKit 互认）
python3 -m pytest                    # 系统 Python：自动跳过需要额外依赖的测试
python3 -m pytest tests/test_state.py -k all_in   # 跑单个主题
python3 scripts/soak.py 200000       # 随机自对弈压测 + 牌谱往返抽验 + 吞吐测量
python3 scripts/build_preflop_equity.py   # 重算翻前权益表（2 核约 22 分钟，通常不用跑）
python3 scripts/build_preflop_ranges.py   # 重算六人桌翻前范围表（约 20 分钟，通常不用跑）
.venv/bin/python -m pytest -m slow   # 慢测：翻前全枚举 + 完整翻前树求解（约 2 分钟）

# 起服务（手机需与电脑同一 Wi-Fi，启动时会打印手机用的网址）
PYTHONPATH=src .venv/bin/python -m holdem_server --port 8000
```

依赖分层：**`holdem` 包零依赖**，服务端才需要 fastapi/uvicorn，`pokerkit` 仅供测试。
`.venv` 里装齐了全部；系统 Python 下相关测试会自动跳过。

## 目录结构

```
src/holdem/      纯逻辑引擎，不碰 IO、不摸随机数
  cards.py       牌编码 card = rank*4 + suit（rank 0..12 = 2..A，suit 0..3 = cdhs）
  evaluator.py   5–7 张求值，返回可直接比较的整数分数
  pots.py        主池/边池切分、未跟注退款、按牌力分配（纯函数）
  actions.py     动作与合法动作集
  state.py       手牌状态机：发牌、下注轮、街推进、结算
  deck.py        全项目唯一制造随机数的地方（种子由调用方给）
  positions.py   位置命名（BTN/SB/BB/UTG/HJ/CO…），统计按位置拆分的唯一口径
  history.py     从事件流还原每个决策点的上下文，M2 统计全部建在它上面
  phh.py         PHH 牌谱读写（纯逻辑）
  ranges.py      169 个起手牌类 + 范围记法；编号直接对应 13×13 图表布局
  equity.py      蒙特卡洛权益估算 + 精确枚举基准值（exact_equity）
  equity_table.py 预计算的 169×169 翻前权益表 + 共牌权重；只用 array，零依赖
  data/          预计算数据（翻前权益表约 112 KB，随包分发）
  pushfold.py    短筹码推弃纳什求解（CFR+）；全项目唯一可精确求解的策略
  preflop_tree.py 翻前抽象博弈树（离散下注尺度的公共树，不含牌）
  realization.py  翻后收益的压缩模型（权益兑现系数）；参数是未校准的假设
  preflop_solver.py 翻前树的 CFR+ 求解器（两人）；正确性靠可利用度 + 退化交叉验证
  preflop_chain.py  六人桌按位置分解成一串两人子博弈，再链式合成开牌范围
  preflop_ranges.py 读取离线生成的翻前范围表（只查表，不求解）
  bots.py        **占位**规则 bot + 风格预设；M1 会整体替换实现
  store.py       SQLite 落库——引擎包里唯一的 IO 模块
src/holdem_server/
  table.py       牌桌编排，不 import 任何 Web 框架，可脱离 HTTP 单测
  app.py         FastAPI 适配层：HTTP + WebSocket + 静态页
  static/        单文件前端（无构建步骤，手机浏览器直接可用）
tests/           单测；求值器用「双实现交叉验证」，牌谱用 PokerKit 外部互认
scripts/soak.py  压测脚本
docs/            PRD / ARCHITECTURE / ADR / DEVLOG
```

## 已知坑

- **下注额一律是「加注到（raise-to）」的目标总额**，不是本次追加的增量。所有 API、
  测试、将来的牌谱都用这个口径，换算错误是这类引擎最高频的 bug 来源。
- **发牌顺序依赖按钮位**。`stacked_deck()` 造测试牌时必须传与 `HandConfig.button`
  一致的 `button`，否则底牌会发错人。
- **短额全下不重开加注权**，但已行动的玩家仍然要面对新的下注额做跟/弃决策。判定靠
  `acted_at_level[seat] < last_full_raise_level`，改动这里务必跑 `test_state.py` 全部用例。
- **「无人可跟时不能加注」会先于重开权生效**。写测试验证重开权时，桌上必须留至少一个
  未弃牌且未全下的第三方，否则测到的是另一条规则（踩过一次）。
- **筹码单位是整数**，不用浮点。分池余数按庄家左手顺序逐枚分配。
- **不能加注到「没人跟得起」的额度**。`max_raise_to` 封顶在对手能跟到的最大额，
  超出部分只会原样退回，而且会被 PokerKit 判为非法动作。想全下 1000 但对手最多只能
  凑 300 时，合法动作是加注到 300。
- **PHH 的玩家编号不是座位号**：p1 是小盲、pN 是按钮，用 `phh_player_order()` 换算。
- **PHH 单挑是特例**：参考实现在两人局把盲注数组反向套用，`[5,10]` 实际是 p1 押 10、
  p2 押 5，**p1 是大盲、p2 才是按钮**。这条是实测出来的，别凭直觉改。
- **平分底池的零头规则与 PokerKit 不同**：我们按赌场惯例从庄家左手起依次多分一枚，
  PokerKit 用自己的 divmod。跨工具比对时零头要给容差，不是 bug。
- **`bots.py` 是占位实现，不是训练用对手**。翻前用单挑权益当牌力尺度、翻后才用多路权益
  ——如果翻前也用多路权益去比即时底池赔率，会算出「跟注要 40% 权益」而把所有牌都弃掉。
  真策略在 M1 接入，接口保持不变。
- **视图绝不能泄露对手底牌**。`TableSession.view()` 只公开英雄本人与摊牌者的牌，
  改动这里必须跑 `test_table.py` 与 `test_server.py` 里的泄露测试。
- **权益数值的基准是 `exact_equity`（全枚举），不是蒙特卡洛**。翻前单个对局要枚举
  171 万个牌面、约 16 秒，所以整表预计算不能走这条路；蒙特卡洛估计器的正确性由
  `test_exact_equity.py` 拿精确值校验。
- **牌类编号即图表坐标**：`index = 行*13 + 列`，第 0 行是 A，右上三角同花、左下三角不同花。
  前端直接按编号摆格子，改编号方案会同时弄歪图表与范围表。
- **推弃求解用 CFR+，别退回虚拟对局**。虚拟对局是 O(1/√t)：400 次迭代后可利用度仍有
  0.005 bb/手（0.5 bb/100），对一个号称「纳什」的模块太大。CFR+ 在同样开销下 50–275 次
  迭代就到 1e-4 以下，还快 5 倍。
- **求解正确性靠可利用度自证，不靠公开图表**。网页上的纳什表抓下来经常串行——实测
  HoldemResources 的 A2o/Q9s/J8s 跟注门槛与直接算出的 EV 明显矛盾（10bb 时 A2o 有约
  50% 权益、只需 45% 就该跟，公开值却说 8.1bb 弃）。拿公开值当基准前先用 EV 验一遍。
- **深处节点的频率必须按到达概率加权**。CFR 只约束走得到的信息集，走不到的地方策略是
  任意的——直接读 `strategy_at` 会得出「72o 有四成跟 3bet」这种鬼话（它在根节点就 100% 弃）。
  统计与画图一律用 `action_range` / `action_frequency` / `arriving_range`，它们乘过到达概率。
- **`realization.py` 的参数是假设，不是测量值**。位置基准、牌型修正、锐化系数 γ 都取自公开
  求解器的常识量级，尚未用 Slumbot 实测校准（计划见 ADR-0003）。所以测试只验性质
  （守恒、单调、退化），**别把某个具体数字钉成基准**；界面上这类建议只能标 C 级。
- **终局份额必须守恒**。份额写成归一化比值（而不是 `R · eq · P`），两边恒好加成 1；
  改这块要跑 `test_realization.py` 的守恒测试，不守恒的终局会让求解器凭空造钱。
- **完整六人翻前树有 62 万个节点**，单挑只有 40 个。所以六人桌**不建整棵树**，而是按位置
  拆成一串两人子博弈再链式合成（`preflop_chain.py`，ADR-0004）。想直接解整桌的念头先打消。
- **范围表是离线产物，不是运行时算的**。改了兑现模型或开牌尺度，要重跑
  `scripts/build_preflop_ranges.py`（约 20 分钟）才会生效；产物里存着当时的参数，
  对不上就说明表过期了。
- **链式求解有三条已知简化**（ADR-0004）：只有第一个不弃牌的人继续（不建模多人底池与挤压）、
  弃牌者不带走牌、防守者不担心身后。前后两条都让范围偏松，评估结果时要把这个方向记在心里。
- **子博弈里玩家 0 是防守者、玩家 1 是开牌者**。取错一侧会得到符号相反的 EV，而数值看着
  仍然「像那么回事」——`test_preflop_chain.py` 的凸包测试就是为这个设的。
- 交叉验证的参考求值器在 `tests/test_evaluator.py`，替换快路径实现时用它当基准。

## 目标运行环境

开发在 Linux；真机验证在 **Windows + RTX 4060 Ti 8GB**。写代码时避免 POSIX-only 依赖
与硬编码路径分隔符。8GB 显存是后续接 GPU 求解器时的预算上限，大动作树要能回退到 CPU。
