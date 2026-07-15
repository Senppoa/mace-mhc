"""Tests for the multi-head committee (MHC) uncertainty path.

A single model is trained with two output heads whose names contain
"committee"; the ``MACECalculator(head_committee=...)`` and
``eval_configs --head_committee`` code paths then treat those heads as a
committee and expose the prediction mean plus its spread.

Reference: Beck et al., "Multi-head committees enable direct uncertainty
prediction for atomistic foundation models", J. Chem. Phys. 163, 234103 (2025).
"""

import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms

from mace.calculators.mace import MACECalculator

run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"
eval_configs = Path(__file__).parent.parent / "mace" / "cli" / "eval_configs.py"


@pytest.fixture(scope="module", name="fitting_configs")
def fitting_configs_fixture():
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    fit_configs = [
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3),
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3),
    ]
    fit_configs[0].info["REF_energy"] = 1.0
    fit_configs[0].info["config_type"] = "IsolatedAtom"
    fit_configs[1].info["REF_energy"] = -0.5
    fit_configs[1].info["config_type"] = "IsolatedAtom"

    np.random.seed(5)
    for _ in range(20):
        c = water.copy()
        c.positions += np.random.normal(0.1, size=c.positions.shape)
        c.info["REF_energy"] = np.random.normal(0.1)
        c.new_array("REF_forces", np.random.normal(0.1, size=c.positions.shape))
        fit_configs.append(c)

    return fit_configs


@pytest.fixture(scope="module", name="trained_head_committee")
def trained_head_committee_fixture(tmp_path_factory, fitting_configs):
    """Train a single model with two heads named 'committee-0' / 'committee-1'.

    The two heads see the same structures here (enough to exercise the code
    path); in production the heads would be trained on different (overlapping)
    subsets to induce a meaningful committee spread.
    """
    tmp_path = tmp_path_factory.mktemp("mhc_run_")

    configs_c0 = []
    configs_c1 = []
    for c in fitting_configs:
        c0 = c.copy()
        c0.info["head"] = "committee-0"
        configs_c0.append(c0)
        c1 = c.copy()
        c1.info["head"] = "committee-1"
        configs_c1.append(c1)
    ase.io.write(tmp_path / "fit_c0.xyz", configs_c0)
    ase.io.write(tmp_path / "fit_c1.xyz", configs_c1)

    heads = {
        "committee-0": {"train_file": f"{str(tmp_path)}/fit_c0.xyz"},
        "committee-1": {"train_file": f"{str(tmp_path)}/fit_c1.xyz"},
    }
    yaml_str = "heads:\n"
    for key, value in heads.items():
        yaml_str += f"  {key}:\n"
        for sub_key, sub_value in value.items():
            yaml_str += f"    {sub_key}: {sub_value}\n"
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(yaml_str)

    mace_params = {
        "name": "MACE_MHC",
        "valid_fraction": 0.1,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "model": "MACE",
        "hidden_irreps": "16x0e",
        "r_max": 3.5,
        "batch_size": 2,
        "max_num_epochs": 6,
        "swa": None,
        "ema": None,
        "amsgrad": None,
        "device": "cpu",
        "seed": 5,
        "loss": "weighted",
        "default_dtype": "float64",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "eval_interval": 2,
        "multiheads_finetuning": False,
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "config": str(config_file),
    }

    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)

    cmd = (
        sys.executable
        + " "
        + str(run_train)
        + " "
        + " ".join(
            (f"--{k}={v}" if v is not None else f"--{k}")
            for k, v in mace_params.items()
        )
    )
    p = subprocess.run(cmd.split(), env=run_env, check=True)
    assert p.returncode == 0

    return tmp_path / "MACE_MHC.model"


def test_calculator_head_committee_auto(fitting_configs, trained_head_committee):
    """head_committee=True auto-selects the 'committee-*' heads and emits UQ."""
    calc = MACECalculator(
        model_paths=trained_head_committee,
        device="cpu",
        default_dtype="float64",
        head_committee=True,
    )
    assert calc.head_committee == ["committee-0", "committee-1"]
    assert "energy_var" in calc.implemented_properties
    assert "forces_comm" in calc.implemented_properties

    at = fitting_configs[5].copy()
    at.calc = calc
    energy = at.get_potential_energy()
    results = at.calc.results

    # committee keys present
    for key in ["energy_comm", "energy_var", "forces_comm", "forces_var"]:
        assert key in results, f"missing committee key {key}"

    # committee has one entry per head
    assert results["energy_comm"].shape[0] == 2
    assert results["forces_comm"].shape[0] == 2

    # mean of the committee equals the reported energy
    np.testing.assert_allclose(energy, results["energy_comm"].mean(), atol=1e-8)

    # variances are non-negative and finite
    assert results["energy_var"] >= 0.0
    assert np.all(results["forces_var"] >= 0.0)
    assert np.isfinite(results["energy_var"])


def test_calculator_head_committee_explicit_list(
    fitting_configs, trained_head_committee
):
    """An explicit list of heads is honoured and matches the auto result."""
    calc = MACECalculator(
        model_paths=trained_head_committee,
        device="cpu",
        default_dtype="float64",
        head_committee=["committee-0", "committee-1"],
    )
    at = fitting_configs[5].copy()
    at.calc = calc
    at.get_potential_energy()
    assert calc.head_committee == ["committee-0", "committee-1"]
    assert at.calc.results["energy_var"] >= 0.0


def test_calculator_head_committee_string_forms(trained_head_committee):
    """'auto' and comma-separated strings resolve like True / explicit list."""
    calc_auto = MACECalculator(
        model_paths=trained_head_committee,
        device="cpu",
        default_dtype="float64",
        head_committee="auto",
    )
    assert calc_auto.head_committee == ["committee-0", "committee-1"]
    calc_csv = MACECalculator(
        model_paths=trained_head_committee,
        device="cpu",
        default_dtype="float64",
        head_committee="committee-0, committee-1",
    )
    assert calc_csv.head_committee == ["committee-0", "committee-1"]


def test_calculator_head_committee_disabled(fitting_configs, trained_head_committee):
    """Without head_committee, no committee keys are produced (single head)."""
    calc = MACECalculator(
        model_paths=trained_head_committee,
        device="cpu",
        default_dtype="float64",
        head="committee-0",
    )
    assert calc.head_committee is None
    at = fitting_configs[5].copy()
    at.calc = calc
    at.get_potential_energy()
    assert "energy_var" not in at.calc.results


def test_calculator_head_committee_bad_head(trained_head_committee):
    """Requesting a non-existent head raises."""
    with pytest.raises(ValueError):
        MACECalculator(
            model_paths=trained_head_committee,
            device="cpu",
            default_dtype="float64",
            head_committee=["committee-0", "does-not-exist"],
        )


def test_eval_configs_head_committee(tmp_path, fitting_configs, trained_head_committee):
    """eval_configs --head_committee writes energy_std / forces_std to the xyz."""
    configs = [c.copy() for c in fitting_configs[2:8]]
    in_path = tmp_path / "eval_in.xyz"
    out_path = tmp_path / "eval_out.xyz"
    ase.io.write(in_path, configs)

    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)

    cmd = [
        sys.executable,
        str(eval_configs),
        f"--configs={in_path}",
        f"--model={trained_head_committee}",
        f"--output={out_path}",
        "--device=cpu",
        "--default_dtype=float64",
        "--head_committee=auto",
        "--batch_size=2",
    ]
    p = subprocess.run(cmd, env=run_env, check=True)
    assert p.returncode == 0

    out = ase.io.read(out_path, index=":")
    assert len(out) == len(configs)
    for at in out:
        assert "MACE_energy_std" in at.info
        assert float(at.info["MACE_energy_std"]) >= 0.0
        assert "MACE_forces_std" in at.arrays
        assert at.arrays["MACE_forces_std"].shape == (len(at), 3)
        assert np.all(at.arrays["MACE_forces_std"] >= 0.0)
