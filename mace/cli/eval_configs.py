###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import logging
from typing import Dict

import ase.data
import ase.io
import numpy as np
import torch
from e3nn import o3

from mace import data
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--enable_cueq",
        help="enable cuequivariance acceleration",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compute_bec",
        help="compute BEC",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_contributions",
        help="model outputs energy contributions for each body order, only supported for MACE, not ScaleShiftMACE",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_descriptors",
        help="model outputs MACE descriptors",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--descriptor_num_layers",
        help="number of layers to take descriptors from",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--descriptor_aggregation_method",
        help="method for aggregating node features. None saves descriptors for each atom.",
        choices=["mean", "per_element_mean", None],
        default=None,
    )
    parser.add_argument(
        "--descriptor_invariants_only",
        help="save invariant (l=0) descriptors only",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--return_node_energies",
        help="model outputs MACE node energies",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--head",
        help="Model head used for evaluation",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--head_committee",
        help=(
            "Treat several heads of the model as a committee (multi-head committee, "
            "MHC) and write the committee mean prediction plus its spread "
            "(<prefix>energy_std, <prefix>forces_std). Pass a comma-separated list "
            "of head names, or 'auto' to use every head whose name contains "
            "'committee'. See Beck et al., J. Chem. Phys. 163, 234103 (2025)."
        ),
        type=str,
        required=False,
        default=None,
    )
    return parser.parse_args()


def get_model_output(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    compute_stress: bool,
    compute_bec: bool,
) -> Dict[str, torch.Tensor]:
    forward_args = {
        "compute_stress": compute_stress,
    }
    if compute_bec:
        # Only add `compute_bec` if it is requested
        # We check if the model is MACELES at the start of the run function
        forward_args["compute_bec"] = compute_bec
    return model(batch, **forward_args)


def get_committee_output(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    compute_stress: bool,
    head_indices: list,
) -> Dict[str, torch.Tensor]:
    """Run one forward pass per committee head (overriding the per-graph head
    index each time) and return the mean energy/forces/stress plus the committee
    standard deviation in ``energy_std`` and ``forces_std``.
    """
    energies, forces, stresses = [], [], []
    for h in head_indices:
        # Fresh clone per pass: the forward mutates the dict in place
        # (requires_grad_ on positions, symmetric displacement for stress),
        # so tensors must not be shared between committee passes.
        batch_h = {
            k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        batch_h["head"] = torch.full_like(batch_h["head"], h)
        out = model(batch_h, compute_stress=compute_stress)
        energies.append(out["energy"].detach())
        forces.append(out["forces"].detach())
        if compute_stress and out.get("stress") is not None:
            stresses.append(out["stress"].detach())
    energy = torch.stack(energies, dim=0)  # [n_heads, n_graphs]
    force = torch.stack(forces, dim=0)  # [n_heads, n_atoms, 3]
    result = {
        "energy": energy.mean(dim=0),
        "energy_std": energy.std(dim=0, unbiased=False),
        "forces": force.mean(dim=0),
        "forces_std": force.std(dim=0, unbiased=False),
    }
    if stresses:
        stress = torch.stack(stresses, dim=0)
        result["stress"] = stress.mean(dim=0)
        result["stress_std"] = stress.std(dim=0, unbiased=False)
    return result


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    if model.__class__.__name__ != "MACELES" and args.compute_bec:
        raise ValueError("BEC can only be computed with MACELES model. ")
    if args.enable_cueq:
        print("Converting models to CuEq for acceleration")
        model = run_e3nn_to_cueq(model, device=device)
    model = model.to(
        args.device
    )  # shouldn't be necessary but seems to help with CUDA problems

    for param in model.parameters():
        param.requires_grad = False

    try:
        model_heads = list(model.heads)
    except AttributeError:
        model_heads = None

    # Resolve multi-head committee members, if requested.
    committee_heads = None
    if args.head_committee is not None:
        if model_heads is None:
            raise ValueError("--head_committee requires a model with named heads")
        if args.head_committee.strip().lower() == "auto":
            committee_heads = [h for h in model_heads if "committee" in str(h).lower()]
            if not committee_heads:
                raise ValueError(
                    f"--head_committee auto found no 'committee' head in {model_heads}"
                )
        else:
            committee_heads = [h.strip() for h in args.head_committee.split(",")]
            missing = [h for h in committee_heads if h not in model_heads]
            if missing:
                raise ValueError(
                    f"--head_committee heads {missing} not in model heads {model_heads}"
                )
        if len(committee_heads) < 2:
            raise ValueError(
                f"--head_committee needs >=2 heads, got {committee_heads}"
            )
        unsupported = [
            flag
            for flag, active in [
                ("--compute_bec", args.compute_bec),
                ("--return_contributions", args.return_contributions),
                ("--return_descriptors", args.return_descriptors),
                ("--return_node_energies", args.return_node_energies),
            ]
            if active
        ]
        if unsupported:
            raise ValueError(
                f"--head_committee does not support {', '.join(unsupported)}; "
                "run them in a separate pass without --head_committee."
            )
        logging.info(f"Multi-head committee over heads: {committee_heads}")

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, index=":")
    if args.head is not None:
        assert args.head in model.heads
        head_name = args.head
    elif committee_heads is not None:
        head_name = committee_heads[0]
    else:
        head_name = "Default"
    configs = [
        data.config_from_atoms(atoms, head_name=head_name) for atoms in atoms_list
    ]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max), heads=heads
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    contributions_list = []
    descriptors_list = []
    node_energies_list = []
    stresses_list = []
    bec_list = []
    qs_list = []
    forces_collection = []
    energy_std_list = []
    forces_std_collection = []

    committee_indices = (
        [heads.index(h) for h in committee_heads]
        if committee_heads is not None
        else None
    )

    for batch in data_loader:
        batch = batch.to(device)
        if committee_indices is not None:
            output = get_committee_output(
                model, batch.to_dict(), args.compute_stress, committee_indices
            )
            energy_std_list.append(torch_tools.to_numpy(output["energy_std"]))
            forces_std = np.split(
                torch_tools.to_numpy(output["forces_std"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            forces_std_collection.append(forces_std[:-1])  # drop last as its empty
        else:
            output = get_model_output(
                model, batch.to_dict(), args.compute_stress, args.compute_bec
            )
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if args.compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))
        if args.compute_bec:
            becs = np.split(
                torch_tools.to_numpy(output["BEC"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            bec_list.append(becs[:-1])  # drop last as its empty

            qs = np.split(
                torch_tools.to_numpy(output["latent_charges"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            qs_list.append(qs[:-1])  # drop last as its empty

        if args.return_contributions:
            contributions_list.append(torch_tools.to_numpy(output["contributions"]))

        if args.return_descriptors:
            num_layers = args.descriptor_num_layers
            if num_layers == -1:
                num_layers = int(model.num_interactions)
            irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
            l_max = irreps_out.lmax
            num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
            per_layer_features = [
                irreps_out.dim for _ in range(int(model.num_interactions))
            ]
            per_layer_features[-1] = (
                num_invariant_features  # Equivariant features not created for the last layer
            )

            descriptors = output["node_feats"]

            if args.descriptor_invariants_only:
                descriptors = extract_invariant(
                    descriptors,
                    num_layers=num_layers,
                    num_features=num_invariant_features,
                    l_max=l_max,
                )

            to_keep = np.sum(per_layer_features[:num_layers])
            descriptors = descriptors[:, :to_keep].detach().cpu().numpy()

            descriptors = np.split(
                descriptors,
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            descriptors_list.extend(descriptors[:-1])  # drop last as its empty

        if args.return_node_energies:
            node_energies_list.append(
                np.split(
                    torch_tools.to_numpy(output["node_energy"]),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                )[
                    :-1
                ]  # drop last as its empty
            )

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        forces_collection.append(forces[:-1])  # drop last as its empty

    energies = np.concatenate(energies_list, axis=0)
    forces_list = [
        forces for forces_list in forces_collection for forces in forces_list
    ]
    assert len(atoms_list) == len(energies) == len(forces_list)
    if args.compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(atoms_list) == stresses.shape[0]

    if args.compute_bec:
        bec_list = [becs for sublist in bec_list for becs in sublist]
        qs_list = [qs for sublist in qs_list for qs in sublist]

    if args.return_contributions:
        contributions = np.concatenate(contributions_list, axis=0)
        assert len(atoms_list) == contributions.shape[0]

    if args.return_descriptors:
        # no concatentation  - elements of descriptors_list have non-uniform shapes
        assert len(atoms_list) == len(descriptors_list)

    if args.return_node_energies:
        node_energies = np.concatenate(node_energies_list, axis=0)
        assert len(atoms_list) == node_energies.shape[0]

    if committee_indices is not None:
        energy_stds = np.concatenate(energy_std_list, axis=0)
        forces_std_list = [
            fstd for sublist in forces_std_collection for fstd in sublist
        ]
        assert len(atoms_list) == len(energy_stds) == len(forces_std_list)

    # Store data in atoms objects
    for i, (atoms, energy, forces) in enumerate(zip(atoms_list, energies, forces_list)):
        atoms.calc = None  # crucial
        atoms.info[args.info_prefix + "energy"] = energy
        atoms.arrays[args.info_prefix + "forces"] = forces

        if committee_indices is not None:
            atoms.info[args.info_prefix + "energy_std"] = energy_stds[i]
            atoms.arrays[args.info_prefix + "forces_std"] = forces_std_list[i]

        if args.compute_stress:
            atoms.info[args.info_prefix + "stress"] = stresses[i]

        if args.compute_bec:
            atoms.arrays[args.info_prefix + "BEC"] = bec_list[i].reshape(-1, 9)
            atoms.arrays[args.info_prefix + "latent_charges"] = qs_list[i]

        if args.return_contributions:
            atoms.info[args.info_prefix + "BO_contributions"] = contributions[i]

        if args.return_descriptors:
            descriptors = descriptors_list[i]
            if args.descriptor_aggregation_method:
                if args.descriptor_aggregation_method == "mean":
                    descriptors = np.mean(descriptors, axis=0)
                elif args.descriptor_aggregation_method == "per_element_mean":
                    descriptors = {
                        element: np.mean(
                            descriptors[atoms.symbols == element], axis=0
                        ).tolist()
                        for element in np.unique(atoms.symbols)
                    }
                atoms.info[args.info_prefix + "descriptors"] = descriptors
            else:  # args.descriptor_aggregation_method is None
                # Save descriptors for each atom (default behavior)
                atoms.arrays[args.info_prefix + "descriptors"] = np.array(descriptors)

        if args.return_node_energies:
            atoms.arrays[args.info_prefix + "node_energies"] = node_energies[i]

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
