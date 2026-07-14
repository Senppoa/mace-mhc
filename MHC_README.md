# Multi-Head Committee (MHC) uncertainty quantification

This adds **direct uncertainty prediction** to a *single* MACE model by treating
a set of its output heads as a committee, following

> H. Beck, P. Simko, L. L. Schaaf, O. Marsalek, C. Schran,
> *Multi-head committees enable direct uncertainty prediction for atomistic
> foundation models*, J. Chem. Phys. **163**, 234103 (2025).

A multi-head committee shares the message-passing backbone (the atomic
environment descriptors) across all members and only duplicates the cheap
readout heads. The **spread between the heads** is used as an uncertainty
estimate for energies and forces — obtained from one model, at inference time,
with no change to training or the model `forward`.

## What this PR changes

Inference / CLI only — the model architecture, `forward`, TorchScript
compilation, cueq/oeq and LAMMPS paths are untouched:

- `mace/calculators/mace.py` — `MACECalculator(head_committee=...)`. When set,
  the calculator runs one forward pass per committee head (sharing
  `self.models[0]`) and reports the committee mean plus spread through the same
  `energy_comm` / `energy_var` / `forces_comm` / `forces_var` / `stress_var`
  result keys already used by a multi-model committee.
- `mace/cli/eval_configs.py` — `--head_committee` flag; writes
  `<prefix>energy_std` and `<prefix>forces_std` into the output xyz.
- `tests/test_mhc.py` — trains a 2-head model and checks both paths.

## Training a committee model (no code changes needed)

Committee heads are just ordinary MACE heads whose names contain `committee`.
Train them with the existing multi-head machinery. To equip a foundation model
(e.g. MACE-OMol) with uncertainty while preserving its accuracy, **freeze the
backbone and train only the heads** (`--freeze 6`), and give each head its own
(overlapping) subset of the data:

```yaml
# config.yaml
foundation_model: "path/to/MACE-omol.model"
foundation_model_elements: yes
multiheads_finetuning: false     # do NOT add the MP replay 'pt_head'
freeze: 6                        # freeze embedding+interactions+products, train readouts only
loss: "weighted"
E0s: "foundation"
enable_cueq: yes                 # accelerates the shared backbone

heads:
  committee-0: { train_file: "subset_0.aselmdb", atomic_numbers: [...] }
  committee-1: { train_file: "subset_1.aselmdb", atomic_numbers: [...] }
  committee-2: { train_file: "subset_2.aselmdb", atomic_numbers: [...] }
  committee-3: { train_file: "subset_3.aselmdb", atomic_numbers: [...] }
```

Notes:
- `.aselmdb` / `.lmdb` / `.h5` / `.xyz` are all accepted per head. LMDB/HDF5 heads
  must declare `atomic_numbers` (they cannot be inferred from the data).
- "overlapping" (each head sampled independently from the full set) keeps a
  stronger committee spread than a disjoint split — recommended for large
  datasets where heads otherwise converge and the spread collapses.

## Inference — Python (ASE calculator)

```python
from mace.calculators.mace import MACECalculator

calc = MACECalculator(
    model_paths="MACE_MHC.model",
    device="cuda",
    default_dtype="float32",
    enable_cueq=True,
    head_committee=True,          # or an explicit list: ["committee-0", ...]
)
atoms.calc = calc
atoms.get_potential_energy()

calc.results["energy"]        # committee mean energy
calc.results["energy_var"]    # committee variance of the energy  -> std = sqrt(var)
calc.results["forces"]        # committee mean forces
calc.results["forces_comm"]   # [n_heads, n_atoms, 3] per-head forces
# per-atom force std:
import numpy as np
forces_std = np.sqrt(np.var(calc.results["forces_comm"], axis=0))
```

`head_committee=True` auto-selects every head whose name contains `committee`;
pass an explicit list to control the members. It is rejected for a multi-model
committee (`model_paths=` with a wildcard) — combine at most one committee type.

**Cost:** the shared backbone is evaluated once per head, so a prediction with
`N` heads costs ≈ `N` forward passes. Only request uncertainty on the frames
where you need it (active learning / error monitoring), not every MD step.
Keep the force-uncertainty path in eager mode (do not stack `torch.compile`).

## Inference — CLI (batch evaluation of an xyz)

```bash
mace_eval_configs \
    --configs data.xyz \
    --model MACE_MHC.model \
    --output data_out.xyz \
    --device cuda --default_dtype float32 \
    --head_committee auto        # or "committee-0,committee-1,committee-2,committee-3"
```

Writes `MACE_energy_std` (per structure) and `MACE_forces_std` (per atom) into
`data_out.xyz`.

## Calibrating the uncertainty

The raw committee spread is typically an underestimate of the true error. Fit a
single scaling factor `alpha` on a held-out validation set (Beck et al. Eq. 5):

```
alpha^2 = mean_i ( (y_i - y_ref,i)^2 / sigma_i^2 )
```

and check the Pearson correlation between error and `sigma` (Eq. 6) — for forces
this is usually strong, for energies weaker. Multiply the reported std by
`alpha` before use.
```
