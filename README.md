# 德扑训练台

本地运行的德州扑克训练与复盘系统：多 bot 对战、可调对手风格、牌谱统计、
基于求解器 EV 损失的复盘打分、锦标赛与水平评级。

完整设计方案：<https://claude.ai/code/artifact/9a1d285c-c5e6-4613-90ef-f83507b04b2b>

> 仅用于离线训练与复盘，不提供接入真钱平台的实时辅助。

## 当前状态

**M0 已完成**：牌局引擎、PHH 牌谱读写（与 PokerKit 双向互认）、SQLite 落库、
本地服务端与网页牌桌。

**M1 进行中**：翻前范围表已解出并接入 bot——**六人 100bb**（按位置分解、链式合成）与
**单挑 200bb**（整树精确解，可利用度 0.0008 大盲/手），都是离线求解、随包分发。
另有六档对手风格、批量自对弈与 bb/100 置信区间、Slumbot 强度基线。
**翻后仍是规则启发式**，翻前模型的兑现系数也尚未校准——界面上的建议因此只标 C 级。

## 开始玩

```bash
pip install -e ".[server,dev]"
python -m holdem_server            # 启动后会打印电脑与手机各自的访问网址
```

手机需与电脑处于同一 Wi-Fi，打开打印出的局域网网址即可。牌局自动存进 `hands.sqlite`。

开桌时可给每个座位单独挑风格：照解 / 紧凶 / 松凶 / 岩石 / 跟注站 / 疯子。
**bot 的翻前照解出来的范围表打**（表覆盖不到的局面退回规则），翻后是启发式。
别把它当 GTO 对手：翻前是我们自己模型的解、翻后是规则。

## 开发

```bash
python3 -m pytest                                        # 单测
python3 scripts/soak.py 200000                           # 随机自对弈压测
python3 scripts/play_batch.py --hands 10000 --workers 2  # 六个 bot 打一万手，报 bb/100
python3 scripts/play_slumbot.py --hands 3000 --workers 3 # 跟 Slumbot 打，量强度基线
```

批量对局与 Slumbot 基线都报**带 95% 置信区间**的 bb/100——扑克单手方差极大，
不给区间的胜率数字没有意义（几百手的区间通常有上百 bb/100 宽）。

引擎用法：

```python
from holdem import HandConfig, HandState, deck_from_seed, call, check, raise_to

config = HandConfig(stacks=(1000, 1000, 1000), button=0, big_blind=10, small_blind=5)
hand = HandState(config, deck_from_seed(42))

while not hand.is_complete:
    legal = hand.legal_actions()
    print(hand.describe(), legal)
    hand.apply(call() if legal.can_call else check())

print(hand.result.net)  # 每个座位的净盈亏
```

存档与回放：

```python
from holdem import HandStore, to_phh, parse_phh

with HandStore("hands.sqlite") as store:
    session = store.create_session("练习", small_blind=5, big_blind=10)
    hand_id = store.save_hand(hand, session_id=session,
                              players=["me", "bot1", "bot2"])
    print(store.load_phh(hand_id))          # PHH 牌谱原文
    print(store.player_summary("me").bb_per_100)
```

牌谱是标准 PHH（TOML），可被 PokerKit 等外部工具直接读取，也能读进外部牌谱：

```python
from holdem.phh import loads
replayed = loads(open("hand.phh").read())   # 重放成引擎状态
```

## 文档

- [PRD](docs/PRD.md) — 需求与验收标准
- [ARCHITECTURE](docs/ARCHITECTURE.md) — 分层与关键实现要点
- [ADR](docs/adr/) — 关键决策记录
- [DEVLOG](docs/DEVLOG.md) — 开发日志
- [CLAUDE.md](CLAUDE.md) — 常用命令、结构速览、已知坑
