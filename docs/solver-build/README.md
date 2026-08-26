# 自己编译 TexasSolver（console 分支）

**AGPL-3.0：源码与产物都不进这个仓库、不随包分发**（ADR-0005、ADR-0006）。
这里只放「怎么编」和我们打的补丁，编出来的二进制放在仓库外，用 `TEXAS_SOLVER_HOME` 指过去。

## 为什么不用官方预编译包

- 预编译的 v0.2.0 **范围只认 169 个牌类**，喂 `AhKs` 会 SIGABRT；
  `console` 分支支持具体组合（见 ADR-0006 实测表）。
- dump 里没有 EV，翻牌决策的 EV 需要整树导三层＝GB 级产物。自己编才加得上 EV 查询。

## 步骤（Linux；Windows 同理，用 MinGW + CMake）

```bash
sudo apt-get install -y cmake g++            # Windows：MinGW-w64 + CMake
mkdir -p ~/tools/src && cd ~/tools/src
curl -sSL -o ts-console.tar.gz \
  https://codeload.github.com/bupticybee/TexasSolver/tar.gz/refs/heads/console
tar xzf ts-console.tar.gz && cd TexasSolver-console

# 补丁（有的话）：见本目录下的 *.patch
# git apply ../../path/to/holdem-trainer/docs/solver-build/*.patch

mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make TexasSolver console_solver -j1      # 开发机 2 核 3.6GB：必须 -j1，并行会顶到内存
```

### CMake 4.x 会拒绝这份工程（2026-08-25 Windows 实测）

上游的 `CMakeLists.txt` 声明的最低版本太老，**CMake 4.x 直接报错拒绝配置**。
加一个策略下限绕开：

```powershell
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
```

PowerShell 里那个参数**要带引号**（`"-DCMAKE_POLICY_VERSION_MINIMUM=3.5"`），
否则会被拆开。CMake 3.x 不需要这一条。

实测通过的组合：Windows 11 + CMake 4.4.1 + MinGW-w64，`-j 16`，38 个目标，
产物 `console_solver.exe` 4.5 MB。

编完把二进制与 `resources/` 摆成后端要的样子：

```bash
mkdir -p ~/tools/TexasSolver-console-build && cd ~/tools/TexasSolver-console-build
cp ~/tools/src/TexasSolver-console/build/console_solver .
ln -sfn ~/tools/src/TexasSolver-console/resources resources   # Windows 用复制
export TEXAS_SOLVER_HOME=~/tools/TexasSolver-console-build
```

## 每次重编之后的验收（两道，都别跳）

1. **新旧二进制同树**：同一份命令文件，两个二进制 dump 出来的动作标签必须逐字相同。
   标签一变，我们所有按标签走线路的代码都会静悄悄地找不到路。
2. **慢测**：`TEXAS_SOLVER_HOME=... .venv/bin/python -m pytest tests/test_solver.py
   tests/test_solver_ev.py tests/test_review.py -m slow -q`

## 别踩

- **别照 master 分支编**：那是 Qt 的 GUI 工程（`PokerSolver.h` 里就有 `QDebug`/`QFile`），
  console 分支才是我们这个命令行二进制的来源，且不需要 Qt。
- `ext/` 里的 fmt / googletest / pybind11 **随分支一起发**，不用拉子模块、不用联网装依赖。
- **范围里出现具体组合会自动关掉花色同构**（它会打印警告）。同构是主要加速手段之一，
  代价还没量——见 ADR-0006「已知代价」。
