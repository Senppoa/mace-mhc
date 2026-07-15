# MACE-OMol 上的 Multi-Head Committee (MHC) 不确定度流水线

在 mace-omol extra-large 基座上，用多头委员会实现"预测值 + 不确定度"的直接输出。
路线：**原生集成（mace-0.3.16 分支）+ 冻结主干只训头 + overlapping 采样**。

> **MHC 已原生集成进本项目的 `mace-0.3.16/`**（见其 `MHC_README.md` 与
> `train_test/PR_DESCRIPTION_MHC.md`）：
> - 推理：`MACECalculator(head_committee=True)` / `mace_eval_configs --head_committee auto`
> - 测试：`tests/test_mhc.py` 5/5 通过，回归通过
>
> 数据格式：训练/验证集是 fairchem 的 `.aselmdb`。MACE ≥0.3.11 原生支持读取
> `.aselmdb` 文件/目录，每个 `--heads` 头能各指一份（lmdb 头必须显式给 atomic_numbers）。

---

## Stage 1 — 采样 N 个 overlapping 子集  ✅ `mhc_make_committee_subsets.py`

```bash
python mhc_make_committee_subsets.py \
    --src ./datasets/train \            # 你的 4M 全量 aselmdb 目录
    --out-dir ./datasets/committee \
    --n-heads 4 \
    --frac 0.15 \                       # 每头独立采 ~15% → 头间重叠
    --seed 42
```
产物：`datasets/committee/train_committee-{0,1,2,3}.aselmdb`。
charge/spin/energy/forces 随结构原样带出。overlapping = 每个结构对每个头独立掷骰，
同一结构可进多个头 → 头间保留分歧（4M 大数据下尤其重要，论文 Fig.3）。

调参直觉：头间分歧太小 / 力的 Pearson r 低 → 调**小** `--frac` 或**增大** `--n-heads`。
N=4~8、frac=0.1~0.2 是合理起点。

---

## Stage 2 — 微调（原生 multihead，冻结主干只训头）

训练侧**不需要任何代码改动**，全部用 0.3.16 现有 flag（已读代码确认）。
**开箱即用的完整配置见 `mace-omrxn_mhc_config.yaml`**，直接改路径即可跑：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    $(python -c "import mace.cli.run_train as m; print(m.__file__)") \
    --config="mace-omrxn_mhc_config.yaml"
```

关键片段（配置里已按下述三个实测坑处理好）：

```yaml
foundation_model: "./foundation_model/MACE-omol-0-extra-large-1024.model"
foundation_model_elements: yes
multiheads_finetuning: no    # 坑①：默认 True，会自动注入 pt_head（MP replay）
                             #      污染委员会；必须显式关掉
freeze: 6                    # 冻 embedding+interactions+products，只训 readout 头
                             # （tests/test_freeze.py 确认的用法）
force_mh_ft_lr: yes          # 不让多头微调把 lr 强制改成 1e-4
loss: "weighted"
enable_cueq: yes             # 只加速共享主干，与多头正交
default_dtype: "float32"

# N 个头各指一份 overlapping 子集；头名必须含 committee（推理端靠它识别）
heads:
  committee-0:
    train_file: "./datasets/committee/train_committee-0.aselmdb"
    valid_file: "./datasets/val"
    E0s: "foundation"        # 坑②：aselmdb 头不能用 "estimated"（那要求 xyz 数据）
    atomic_numbers: "[1, 3, 4, 5, 6, 7, 8, 9, 11, ...]"   # 坑③：必须是带引号的字符串
  committee-1: { ... }       # 其余头同构，见 mace-omrxn_mhc_config.yaml
```

**三个必须注意的坑（0.3.16 实测 + 读代码确认）：**
1. `multiheads_finetuning: no` —— 不关的话 `run_train.py` 会自动加名为 `pt_head` 的
   MP replay 头，既污染委员会 std，又把 lr 悄悄改成 1e-4。
2. 各头 `E0s: "foundation"` —— **不能用 `"estimated"`**。`run_train.py` 里 E0s 估计
   有 `assert ... check_path_ase_read`，要求 xyz 训练数据；aselmdb 头会直接断言失败。
   用 `"foundation"` 直接取基座 E0s，四个头共用，保证同一物理目标。
3. 各头 `atomic_numbers` 必须写成**带引号的字符串** `"[1, 3, ...]"`。
   `run_train.py` 对它做 `ast.literal_eval`，写成 YAML 原生列表 `[1, 3, ...]`
   会报 `malformed node or string`。列表内容抄 `datasets/statistics.json`
   的 `atomic_numbers` 字段。

valid_file 各头可共用一份验证集（验证只按头分别报表）。

---

## Stage 3 — 推理（原生 head_committee）  ✅ `mhc_calculator.py`

原生用法（最直接）：
```python
from mace.calculators import MACECalculator
calc = MACECalculator(model_paths="mace-omrxn-mhc.model", device="cuda",
                      default_dtype="float32", enable_cueq=True,
                      head_committee=True)      # 自动选 committee-* 头
atoms.calc = calc
atoms.get_potential_energy()
calc.results["energy_var"]     # 能量方差 -> std = sqrt(var)
calc.results["forces_comm"]    # [n_heads, n_atoms, 3] 各头力
```

便利封装（自动换算 std、乘标定 alpha）：
```python
from mhc_calculator import MHCCommitteeCalculator
calc = MHCCommitteeCalculator(model_path="mace-omrxn-mhc.model",
                              device="cuda", default_dtype="float32",
                              alpha=1.0)        # 标定后填 Stage 4 的 alpha_forces
atoms.calc = calc
atoms.get_potential_energy()
atoms.info["energy_std"]         # 能量不确定度（标量）
atoms.arrays["forces_std_atom"]  # 每原子力不确定度（论文 Eq.4 风格）
```

批量 CLI（写进 xyz）：
```bash
mace_eval_configs --configs data.xyz --model mace-omrxn-mhc.model \
    --output out.xyz --device cuda --default_dtype float32 \
    --head_committee auto
# 输出含 MACE_energy_std（每结构）、MACE_forces_std（每原子）
```

**代价**：一次预测把主干重复算 N 次 → 只在需要 UQ 的帧调用（主动学习/误差监控），
不要每步 MD 都出 std。cueq 主干前向 OK；力的多次反传保持 eager。

---

## Stage 4 — 标定 + 质量评估  ✅ `mhc_calibrate.py`

```bash
python mhc_calibrate.py \
    --model ./models/mace-omrxn-mhc.model \
    --valid ./datasets/val \
    --device cuda --default-dtype float32 \
    --energy-key REF_energy --forces-key REF_forces \
    --per-atom-energy --max-configs 2000
```
输出 `alpha_energy/alpha_forces`（论文 Eq.5 缩放因子）与 `Pearson r`（Eq.6）。
把 `alpha_forces` 填回 Stage 3 的 `MHCCommitteeCalculator(alpha=...)` 即上线。

判读：力的 `r` 通常显著高于能量（论文一致结论）；若力 `r < 0.3` → 头训得太趋同，
回 Stage 1 减小 frac / 增大头数重采。

---

## 与论文的对应
- 架构（共享主干 + 多 readout 头，std 作不确定度）：Fig.1, Eq.1–4
- overlapping vs disjoint 数据分发：II.B（这里用 overlapping）
- foundation model 只训头：II + III.D（对应 `--freeze 6`）
- 标定 alpha 与 Pearson r：Eq.5, Eq.6
- 力 std 需多次反传的开销：II.E / II.A
