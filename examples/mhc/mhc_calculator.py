#!/usr/bin/env python
"""
Stage 3 — Multi-Head Committee 推理封装（基于 0.3.16 原生 head_committee）。

自本项目把 MHC 原生集成进 MACE 0.3.16 后（见 mace-0.3.16/MHC_README.md），
推理时不再需要自己加载 N 个模型。原生 MACECalculator 已支持：

    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths="mace-omrxn-mhc.model",
                          head_committee=True)   # 自动选 committee-* 头
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.results["energy"]        # 委员会均值能量
    calc.results["energy_var"]    # 委员会能量方差 -> std = sqrt(var)
    calc.results["forces_comm"]   # [n_heads, n_atoms, 3] 各头力

本文件是这层原生功能之上的 *便利封装*，提供：
  1. MHCCommitteeCalculator：一个 ASE 计算器，直接把不确定度换算成
     energy_std / forces_std / forces_std_atom 并（可选）乘上标定因子 alpha。
  2. predict_committee(atoms_list)：批量返回预测+不确定度，供 Stage 4 标定 /
     主动学习选点使用。

相比自建 N 个 calculator 的旧实现，这里复用原生的“单模型 N 次前向”，
只保留一份主干权重在显存里——对 1024 通道 extra-large 基座更省显存。

代价（务必知晓）：
  - 一次预测把共享主干重复算 N 次（每个头一次前向）。建议只在需要 UQ 的帧
    调用（主动学习选点 / 误差监控），而非每步 MD 都出 std。
  - cueq 主干前向没问题；力的多次反传保持 eager，别叠 torch.compile 推理路径。

用法：
    from mhc_calculator import MHCCommitteeCalculator
    calc = MHCCommitteeCalculator(
        model_path="./models/mace-omrxn-mhc.model",
        device="cuda", default_dtype="float32",
        alpha=1.0,            # Stage 4 标定得到的缩放因子，默认 1.0
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.info["energy_std"]        # 标量，已乘 alpha
    atoms.arrays["forces_std"]      # [N,3]，已乘 alpha
    atoms.arrays["forces_std_atom"] # [N]，每原子 3 分量 std 的 RMS（论文 Eq.4 风格）
"""
import logging
from typing import List, Optional, Union

import numpy as np

log = logging.getLogger("mhc-calc")

try:
    from ase.calculators.calculator import Calculator, all_changes
except Exception as e:  # pragma: no cover
    raise ImportError("需要 ASE 环境") from e


def _sorted_committee_heads(model_heads: List[str]) -> List[str]:
    """从模型 heads 里挑出名字含 'committee' 的头，按尾部编号排序。"""
    committee = [h for h in model_heads if "committee" in str(h).lower()]
    if not committee:
        raise ValueError(
            f"模型 heads 里没有名字含 'committee' 的头，实际 heads = {model_heads}。\n"
            "请确认训练时头命名为 committee-0, committee-1, ...（推理端靠此识别成员）。"
        )

    def _key(h):
        tail = str(h).split("-")[-1]
        return int(tail) if tail.isdigit() else 0

    committee.sort(key=_key)
    return committee


class MHCCommitteeCalculator(Calculator):
    """基于原生 head_committee 的委员会计算器，输出 std（可乘标定 alpha）。"""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        default_dtype: str = "float32",
        alpha: float = 1.0,
        heads: Optional[List[str]] = None,
        enable_cueq: bool = True,
        **mace_kwargs,
    ):
        super().__init__()
        import torch
        from mace.calculators import MACECalculator

        self.alpha = float(alpha)

        # 解析 committee 头集合。
        if heads is None:
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            model_heads = list(getattr(model, "heads", []) or [])
            self.head_names = _sorted_committee_heads(model_heads)
        else:
            self.head_names = list(heads)
        self.n_heads = len(self.head_names)

        # 单个原生 calculator，用 head_committee 让它一次预测跑 N 次前向。
        self._calc = MACECalculator(
            model_paths=model_path,
            device=device,
            default_dtype=default_dtype,
            head_committee=self.head_names,
            enable_cueq=enable_cueq,
            **mace_kwargs,
        )
        log.info(
            "MHCCommitteeCalculator 就绪：%d 个头 %s，alpha=%.4g",
            self.n_heads,
            self.head_names,
            self.alpha,
        )

    # -------------------------------------------------------------- ASE 接口
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        res = self._run(atoms)
        self.results.update(res)
        # 同时挂到 atoms 上，方便 ase.io 写出
        atoms.info["energy_std"] = res["energy_std"]
        atoms.arrays["forces_std"] = res["forces_std"]
        atoms.arrays["forces_std_atom"] = res["forces_std_atom"]
        if "stress" in res:
            atoms.info["stress_std"] = res["stress_std"]

    # -------------------------------------------------------------- 内部
    def _run(self, atoms) -> dict:
        """跑一次原生委员会预测，把 *_comm/*_var 换算成 std 并乘 alpha。"""
        a = atoms.copy()
        a.calc = self._calc
        a.get_potential_energy()  # 触发计算
        r = self._calc.results

        energy = float(r["energy"])
        forces = np.asarray(r["forces"], dtype=float)
        # energy_var / forces_comm 由原生 head_committee 产生
        e_std = float(np.sqrt(max(r["energy_var"], 0.0))) * self.alpha
        f_comm = np.asarray(r["forces_comm"], dtype=float)  # [n_heads, natoms, 3]
        f_std = f_comm.std(axis=0, ddof=0) * self.alpha  # [natoms, 3]
        f_std_atom = np.sqrt((f_std ** 2).mean(axis=1))  # [natoms]

        out = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
            "energy_std": e_std,
            "forces_std": f_std,
            "forces_std_atom": f_std_atom,
        }
        if r.get("stress") is not None:
            out["stress"] = np.asarray(r["stress"], dtype=float)
            if r.get("stress_var") is not None:
                out["stress_std"] = np.sqrt(np.maximum(r["stress_var"], 0.0)) * self.alpha
        return out

    # -------------------------------------------------------------- 批量预测
    def predict_committee(self, atoms_list):
        """
        对一批结构做委员会预测，返回 list[dict]，每个 dict 含：
            energy, energy_std, forces[N,3], forces_std[N,3], forces_std_atom[N], n_atoms
        供 Stage 4 标定 / 主动学习选点使用。
        """
        out = []
        for atoms in atoms_list:
            res = self._run(atoms)
            out.append(
                dict(
                    energy=res["energy"],
                    energy_std=res["energy_std"],
                    forces=res["forces"],
                    forces_std=res["forces_std"],
                    forces_std_atom=res["forces_std_atom"],
                    n_atoms=len(atoms),
                )
            )
        return out
