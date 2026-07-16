#!/usr/bin/env python
"""
Stage 4 — 不确定度标定 + 质量评估。

在一个带真值(REF_energy/REF_forces)的留出验证集上：
  1. 用委员会得到每个结构/原子的预测与 std（Stage 3）。
  2. 计算缩放因子 alpha（论文 Beck et al. 2025, Eq.5）：
         alpha^2 = (1/N_val) * sum_i (Δy_i^2 / sigma_i^2)
     力和能量各算一个 alpha（论文对二者分别标定）。
  3. 计算 Pearson 相关系数 r（论文 Eq.6）：误差 vs 未标定 std 的相关性，
     用来判断这个不确定度到底"可不可信"（r 越高越好，力通常远好于能量）。

标定后：把得到的 alpha_forces（一般关注力）填进 MHCCommitteeCalculator(alpha=...)，
上线的 std 就乘上它，得到与真实误差量级匹配的不确定度。

用法：
    python mhc_calibrate.py \
        --model ./models/mace-omrxn-mhc.model \
        --valid ./datasets/val \          # aselmdb 文件/目录，或 .xyz
        --device cuda --default-dtype float32 \
        --energy-key REF_energy --forces-key REF_forces \
        --max-configs 2000                # 采样上限，验证集很大时用

输出：alpha_energy, alpha_forces, r_energy, r_forces，并存一份 npz 供画图。
"""
import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mhc-calib")


# ----------------------------------------------------------------------- 读验证集
def load_atoms(path, energy_key, forces_key, max_configs=None, seed=0):
    """读验证集为 list[Atoms]，真值放到 .info['ref_energy'] / .arrays['ref_forces']。"""
    atoms_list = []

    def _stash(a):
        # 统一把真值取出来（兼容 REF_ 前缀 / ase 计算器结果）
        e = a.info.get(energy_key, a.info.get("energy"))
        if e is None:
            try:
                e = a.get_potential_energy()
            except Exception:  # pylint: disable=broad-except
                e = None
        f = a.arrays.get(forces_key, a.arrays.get("forces"))
        if f is None:
            try:
                f = a.get_forces()
            except Exception:  # pylint: disable=broad-except
                f = None
        a.info["ref_energy"] = e
        if f is not None:
            a.arrays["ref_forces"] = np.asarray(f, dtype=float)
        a.calc = None
        return a

    if str(path).endswith(".xyz") or str(path).endswith(".extxyz"):
        import ase.io

        for a in ase.io.iread(path, index=":"):
            atoms_list.append(_stash(a))
    else:
        # aselmdb：优先 fairchem，退化 ase.db
        try:
            from fairchem.core.datasets import AseDBDataset

            src = path.split(":") if ":" in path else path
            ds = AseDBDataset(config={"src": src})
            n = len(ds)
            idx = np.arange(n)
            if max_configs and n > max_configs:
                idx = np.random.default_rng(seed).choice(n, max_configs, replace=False)
            for i in idx:
                atoms_list.append(_stash(ds.get_atoms(ds.ids[int(i)])))
            log.info("读入验证集（fairchem）：%d / %d", len(atoms_list), n)
            return atoms_list
        except Exception as e:  # pylint: disable=broad-except
            log.warning("AseDBDataset 不可用（%s），改 ase.db.connect", e)
            import os

            from ase.db import connect

            files = (
                [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".aselmdb")]
                if os.path.isdir(path)
                else [path]
            )
            for p in files:
                with connect(p) as db:
                    for row in db.select():
                        atoms_list.append(_stash(row.toatoms()))

    if max_configs and len(atoms_list) > max_configs:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(atoms_list), max_configs, replace=False)
        atoms_list = [atoms_list[i] for i in sel]
    log.info("读入验证集：%d 个结构", len(atoms_list))
    return atoms_list


# ----------------------------------------------------------------------- 标定核心
def alpha_scale(errors, stds, eps=1e-8):
    """论文 Eq.5：alpha^2 = mean( (Δy)^2 / sigma^2 )。返回 alpha。"""
    errors = np.asarray(errors, dtype=float).ravel()
    stds = np.asarray(stds, dtype=float).ravel()
    m = stds > eps
    return float(np.sqrt(np.mean((errors[m] ** 2) / (stds[m] ** 2))))


def pearson(errors, stds):
    """论文 Eq.6：误差与 std 的 Pearson r。对力用 |误差| 与 std。"""
    a = np.asarray(errors, dtype=float).ravel()
    b = np.asarray(stds, dtype=float).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description="MHC 不确定度标定")
    ap.add_argument("--model", required=True)
    ap.add_argument("--valid", required=True, help="验证集 aselmdb 文件/目录或 .xyz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--default-dtype", default="float32")
    ap.add_argument("--energy-key", default="REF_energy")
    ap.add_argument("--forces-key", default="REF_forces")
    ap.add_argument("--max-configs", type=int, default=2000)
    ap.add_argument("--per-atom-energy", action="store_true",
                    help="能量误差按每原子归一（推荐，和论文一致）")
    ap.add_argument("--out", default="./mhc_calibration.npz")
    ap.add_argument("--enable-cueq", action="store_true", default=True)
    args = ap.parse_args()

    from mhc_calculator import MHCCommitteeCalculator

    calc = MHCCommitteeCalculator(
        model_path=args.model,
        device=args.device,
        default_dtype=args.default_dtype,
        alpha=1.0,  # 标定阶段用未缩放 std
        enable_cueq=args.enable_cueq,
    )

    atoms_list = load_atoms(
        args.valid, args.energy_key, args.forces_key, args.max_configs
    )
    preds = calc.predict_committee(atoms_list)

    # 收集：能量（每结构）、力（每原子分量）
    e_err, e_std = [], []
    f_err, f_std = [], []  # 逐原子标量
    for atoms, pr in zip(atoms_list, preds):
        n = pr["n_atoms"]
        ref_e = atoms.info.get("ref_energy")
        if ref_e is not None:
            de = pr["energy"] - float(ref_e)
            if args.per_atom_energy:
                de /= n
                se = pr["energy_std"] / n
            else:
                se = pr["energy_std"]
            e_err.append(de)
            e_std.append(se)
        ref_f = atoms.arrays.get("ref_forces")
        if ref_f is not None:
            # 每原子：3 分量误差的 RMS  vs  该原子的标量 std
            df_atom = np.sqrt(((pr["forces"] - ref_f) ** 2).mean(axis=1))  # [n]
            f_err.extend(df_atom.tolist())
            f_std.extend(pr["forces_std_atom"].tolist())

    e_err, e_std = np.array(e_err), np.array(e_std)
    f_err, f_std = np.array(f_err), np.array(f_std)

    res = {}
    log.info("=============== 标定结果 ===============")
    if e_err.size:
        a_e = alpha_scale(e_err, e_std)
        r_e = pearson(np.abs(e_err), e_std)
        res.update(alpha_energy=a_e, r_energy=r_e)
        log.info("能量：alpha_energy = %.4g,  Pearson r = %.3f  (n=%d)", a_e, r_e, e_err.size)
        log.info("      能量 RMSE = %.4g, 平均未标定 std = %.4g",
                 np.sqrt(np.mean(e_err ** 2)), e_std.mean())
    if f_err.size:
        a_f = alpha_scale(f_err, f_std)
        r_f = pearson(f_err, f_std)
        res.update(alpha_forces=a_f, r_forces=r_f)
        log.info("力  ：alpha_forces = %.4g,  Pearson r = %.3f  (n=%d atoms)", a_f, r_f, f_err.size)
        log.info("      力 RMSE = %.4g, 平均未标定 std = %.4g",
                 np.sqrt(np.mean(f_err ** 2)), f_std.mean())
        log.info(">> 上线时把 MHCCommitteeCalculator(alpha=%.4g) 用于力的不确定度", a_f)

    np.savez(args.out, e_err=e_err, e_std=e_std, f_err=f_err, f_std=f_std, **res)
    log.info("原始数据已存 %s（可用于画 误差-std 相关图，论文 Fig.2 风格）", args.out)
    if res.get("r_forces", 0) < 0.3:
        log.warning("力的 Pearson r 偏低（<0.3）：可能头训得太趋同。"
                    "建议减小 --frac 或增大头数，让委员会更有分歧。")


if __name__ == "__main__":
    main()
