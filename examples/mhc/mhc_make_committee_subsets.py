#!/usr/bin/env python
"""
Stage 1 — 为 Multi-Head Committee (MHC) 从 4M 的 .aselmdb 全量训练集
按 "overlapping" 策略采样出 N 个重叠子集，每个子集对应一个 committee 头。

overlapping 的定义（论文 Beck et al. 2025, II.B）：
    对每个头 *独立* 地对全量按概率 FRAC 掷骰采样。
    因此同一个结构可能同时进入多个头的子集（头间重叠），
    这会让头之间保留更强的分歧 → 更有信息量的不确定度（对大数据集尤其重要）。

用法：
    python mhc_make_committee_subsets.py \
        --src ./datasets/train \        # 全量 aselmdb 文件或目录（冒号分隔可给多个）
        --out-dir ./datasets/committee \
        --n-heads 4 \
        --frac 0.15 \
        --seed 42

产物：
    ./datasets/committee/train_committee-0.aselmdb
    ./datasets/committee/train_committee-1.aselmdb
    ...
训练时在 --heads dict 里把每个头的 train_file 指向对应文件即可（见 README_MHC.md）。

注意（LMDB 硬约束）：
    - 一个进程对一个 .aselmdb 同一时刻只能有一个打开的写连接（LMDB 锁槽限制），
      所以这里对每个头顺序打开/写入/关闭，而不是并行。
    - charge / spin / energy / forces 等信息随 atoms 对象原样写出，不做任何改动。
"""
import argparse
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mhc-subset")


# ----------------------------------------------------------------------------- readers
def open_reader(src):
    """
    返回 (get_atoms(i), n_total)。
    读取优先级：
      1. MACE 内置的 fairchem dataset 读取器（只要装了 mace 就有）
      2. 外部 fairchem（fairchem.core.datasets）
      3. ase.db.connect 退化路径（需要 ase_db_backends 才能读 .aselmdb）
    src 支持冒号分隔的多个文件/目录（与 MACE train_file 语义一致）。
    """
    src_arg = src.split(":") if ":" in src else src

    # 1) MACE 内置读取器 —— 最稳健，装了 mace 就有，不需要额外依赖
    try:
        from mace.tools.fairchem_dataset import AseDBDataset

        ds = AseDBDataset(config={"src": src_arg})
        n = len(ds)
        log.info("reader = mace.tools.fairchem_dataset.AseDBDataset, n_total = %d", n)
        # ds.ids[i] 兼容两种情况：
        #   - MACE bundled: ids = range(n)，get_atoms 接收位置索引
        #   - 外部 fairchem: ids 是 DB id，get_atoms 接收 DB id
        return (lambda i: ds.get_atoms(ds.ids[i])), n
    except Exception as e:
        log.warning("mace 内置 AseDBDataset 不可用（%s），尝试外部 fairchem", e)

    # 2) 外部 fairchem（pip install fairchem-core）
    try:
        from fairchem.core.datasets import AseDBDataset

        ds = AseDBDataset(config={"src": src_arg})
        n = len(ds)
        log.info("reader = fairchem.core.datasets.AseDBDataset, n_total = %d", n)
        return (lambda i: ds.get_atoms(ds.ids[i])), n
    except Exception as e:
        log.warning("外部 fairchem 不可用（%s），改用 ase.db.connect 逐库读取", e)

    # 3) ase.db.connect —— 需要 ase_db_backends 才能读 .aselmdb
    #    若还是报 ModuleNotFoundError: ase_db_backends，
    #    请在服务器上执行：pip install ase-db-backends
    from ase.db import connect

    paths = []
    for token in (src.split(":") if isinstance(src, str) else src):
        if os.path.isdir(token):
            paths += [
                os.path.join(token, f)
                for f in sorted(os.listdir(token))
                if f.endswith(".aselmdb")
            ]
        elif token.endswith(".aselmdb"):
            paths.append(token)
    if not paths:
        raise FileNotFoundError(f"在 {src} 下找不到 .aselmdb 文件")

    # 建立 (全局 index) -> (db, 局部 row id) 的映射；ase db 行号从 1 开始
    index = []
    for p in paths:
        with connect(p) as db:
            index += [(p, rid) for rid in range(1, len(db) + 1)]
    log.info("reader = ase.db.connect, files = %d, n_total = %d", len(paths), len(index))

    def _get(i):
        p, rid = index[i]
        with connect(p) as db:
            return db.get_atoms(id=rid)

    return _get, len(index)


# ----------------------------------------------------------------------------- writer
# ASE 保留的 calculator 属性名：不允许作为 db key_value_pairs 的键。
# MACE 内置读取器会把它们塞进 atoms.info（同时也放在 atoms.calc 里），
# 写回前必须从 info 清掉，否则 ase.db 的 check() 抛 "Bad key: energy"。
# 数据不丢：它们保留在 atoms.calc（SinglePointCalculator）中，
# AtomsRow 会自动提取并存为行属性（row.energy / row.forces / ...），
# 与原始数据的存储方式一致（读取端正是从 row.energy 读取的）。
_CALC_PROPS = (
    "energy", "forces", "stress", "free_energy",
    "energies", "stresses", "dipole", "charges", "magmom", "magmoms",
)
# 读取端遗留的内部结构键，写回会造成嵌套污染，也一并清掉。
_INTERNAL_KEYS = ("__arrays__", "__info__")


def sanitize_atoms(atoms):
    """清理读取器塞进 atoms.info 的 calculator 属性与内部键，保证可安全写回。"""
    for k in _CALC_PROPS + _INTERNAL_KEYS:
        atoms.info.pop(k, None)
    return atoms


def make_writer(path):
    """
    返回 (write(atoms), close())。
    写入优先级：
      1. MACE 内置的 LMDBDatabase（只要装了 mace 就有）
      2. 外部 fairchem 的 LMDBDatabase
      3. ase.db.connect 退化写（同样需要 ase_db_backends）
    所有分支写入前都会先 sanitize_atoms()。
    """
    # 1) MACE 内置写入器
    try:
        from mace.tools.fairchem_dataset.lmdb_dataset_tools import LMDBDatabase

        db = LMDBDatabase(path)
        return (lambda atoms: db.write(sanitize_atoms(atoms))), db.close
    except Exception as e:
        log.warning("mace 内置 LMDBDatabase 不可用（%s），尝试外部 fairchem", e)

    # 2) 外部 fairchem
    try:
        from fairchem.core.datasets.lmdb_database import LMDBDatabase

        db = LMDBDatabase(path)
        return (lambda atoms: db.write(sanitize_atoms(atoms))), db.close
    except Exception as e:
        log.warning("外部 fairchem LMDBDatabase 不可用（%s），改用 ase.db.connect", e)

    # 3) ase.db.connect —— 退化路径（需要 ase_db_backends）
    from ase.db import connect

    db = connect(path)
    return (lambda atoms: db.write(sanitize_atoms(atoms))), (lambda: None)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="MHC overlapping 子集采样（aselmdb）")
    ap.add_argument("--src", required=True, help="全量训练集 aselmdb 文件/目录，冒号分隔可给多个")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    ap.add_argument("--n-heads", type=int, default=4, help="committee 头数 N")
    ap.add_argument("--frac", type=float, default=0.15, help="每个头独立采样的比例 p (0-1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--prefix", default="train_committee-", help="输出文件名前缀（保留 committee- 供推理端识别）"
    )
    args = ap.parse_args()

    assert 0.0 < args.frac <= 1.0, "--frac 必须在 (0, 1]"
    assert "committee-" in args.prefix, "prefix 必须含 'committee-'，推理端靠它筛选委员会成员"

    os.makedirs(args.out_dir, exist_ok=True)
    get_atoms, n_total = open_reader(args.src)
    if n_total == 0:
        log.error("全量为空，退出")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    # 预生成 [n_total, n_heads] 的布尔掩码：每个 (结构, 头) 独立掷骰
    # 对 4M x 4 的 bool 矩阵约 16MB，完全放得下；避免逐帧调用 rng 的开销。
    mask = rng.random((n_total, args.n_heads)) < args.frac

    out_paths = [
        os.path.join(args.out_dir, f"{args.prefix}{i}.aselmdb") for i in range(args.n_heads)
    ]
    for p in out_paths:
        if os.path.exists(p):
            log.error("输出已存在，先删除或换目录：%s", p)
            sys.exit(1)

    counts = np.zeros(args.n_heads, dtype=np.int64)
    # 逐头写：满足 LMDB「一进程一文件同时只一个写连接」的约束
    for h in range(args.n_heads):
        idx_h = np.nonzero(mask[:, h])[0]
        write, close = make_writer(out_paths[h])
        try:
            for k, i in enumerate(idx_h):
                write(get_atoms(int(i)))
                if k and k % 50000 == 0:
                    log.info("head %d: 已写 %d/%d", h, k, len(idx_h))
        finally:
            close()
        counts[h] = len(idx_h)
        log.info("head %d 完成：%d 个结构 -> %s", h, counts[h], out_paths[h])

    # 概览
    log.info("=========== 采样完成 ===========")
    log.info("全量 n_total = %d, n_heads = %d, frac = %.3f", n_total, args.n_heads, args.frac)
    for h in range(args.n_heads):
        log.info("  committee-%d : %d (%.1f%%)", h, counts[h], 100 * counts[h] / n_total)
    # 覆盖率：至少被一个头采到的结构占比（overlapping 下通常 < 1）
    covered = np.count_nonzero(mask.any(axis=1))
    overlap = np.count_nonzero(mask.sum(axis=1) >= 2)
    log.info("  被>=1个头覆盖：%d (%.1f%%)，被>=2个头重叠：%d (%.1f%%)",
             covered, 100 * covered / n_total, overlap, 100 * overlap / n_total)
    log.info("训练时 --heads 各头 train_file 指向上面对应文件即可。")


if __name__ == "__main__":
    main()
