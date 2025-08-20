import torch
from omegaconf import DictConfig
from torch.nn import Parameter, ReLU, Sigmoid
from torch_geometric.data import Batch
from torch_geometric.nn import Linear, Sequential
from torch_scatter import scatter

from . import physics
from .pignet import PIGNet


class PIGNetMorse(PIGNet):
    def __init__(
        self,
        config: DictConfig,
        in_features: int = -1,
        **kwargs,
    ):
        super().__init__(config=config)
        self.reset_log()
        self.config = config
        dim_gnn = config.model.dim_gnn
        dim_mlp = config.model.dim_mlp

        self.embed = Linear(in_features, dim_gnn, bias=False)

        self.nn_vdw_epsilon = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                Sigmoid(),
            ],
        )
        self.nn_vdw_width = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                Sigmoid(),
            ],
        )
        self.nn_vdw_radius = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                ReLU(),
            ],
        )
        # Born radii heads
        self.gb_head_mode = config.model.get("gb_head_mode", "shared")
        self.nn_born_radii = Sequential(
            "x",
            [
                (Linear(dim_gnn, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                Sigmoid(),
            ],
        )
        if self.gb_head_mode == "split_sharestem":
            self.nn_born_radii_stem = Sequential(
                "x",
                [
                    (Linear(dim_gnn, dim_mlp), "x -> x"),
                    ReLU(),
                ],
            )
            self.nn_born_radii_head_complex = Sequential(
                "h",
                [
                    (Linear(dim_mlp, 1), "h -> h"),
                    Sigmoid(),
                ],
            )
            self.nn_born_radii_head_iso = Sequential(
                "h",
                [
                    (Linear(dim_mlp, 1), "h -> h"),
                    Sigmoid(),
                ],
            )

        self.hbond_coeff = Parameter(torch.tensor([0.714]))
        self.metal_ligand_coeff = Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = Parameter(torch.tensor([0.216]))
        self.rotor_coeff = Parameter(torch.tensor([0.102]))
        self.ionic_coeff = Parameter(torch.tensor([1.0]))  # NOT USED

    def forward(self, sample: Batch):
        cfg = self.config.model

        # Initial embedding
        x0 = self.embed(sample.x)

        # Graph convolutions (full: intra + inter)
        x = self.conv(x0, sample.edge_index, sample.edge_index_c)

        # Ligand-to-target uni-directional edges
        # to compute pairwise interactions: (2, pairs)
        edge_index_i = physics.interaction_edges(sample.is_ligand, sample.batch)

        # Pairwise distances: (pairs,)
        D = physics.distances(sample.pos, edge_index_i)

        # Limit the interaction distance.
        _mask = (cfg.interaction_range[0] <= D) & (D <= cfg.interaction_range[1])
        edge_index_i = edge_index_i[:, _mask]
        D = D[_mask]

        # Pairwise node features: (pairs, 2*features)
        x_cat = torch.cat((x[edge_index_i[0]], x[edge_index_i[1]]), -1)

        # Pairwise vdW-radii deviations: (pairs,)
        dvdw_radii = self.nn_dvdw(x_cat).view(-1)
        dvdw_radii = dvdw_radii * cfg.dev_vdw_radii_coeff

        # Pairwise vdW radii: (pairs,)
        R = (
            sample.vdw_radii[edge_index_i[0]]
            + sample.vdw_radii[edge_index_i[1]]
            + dvdw_radii
        )

        # Predict born radii for each atom
        born_radii_full = None
        born_radii_iso = None
        if cfg.get("include_gb", False):
            # Complex (full) radii from full conv representation
            if getattr(cfg, "gb_head_mode", "shared") == "split_sharestem":
                h_full = self.nn_born_radii_stem(x)
                u = self.nn_born_radii_head_complex(h_full).squeeze(-1)
            else:
                u = self.nn_born_radii(x).squeeze(-1)

            dev = x.device
            dtype = u.dtype

            # Per-element lower bounds (Å). Fallback is global min.
            try:
                Z = sample.atomic_numbers  # shape: (num_nodes,)
                global_min = torch.as_tensor(cfg.gb_radii_scale[0], device=dev, dtype=dtype)
                global_max = torch.as_tensor(cfg.gb_radii_scale[1], device=dev, dtype=dtype)

                per_atom_min = torch.full_like(u, global_min)

                # Vectorised assignment of element minima
                element_minima = {
                    1: 1.20,  # H
                    6: 1.70,  # C
                    7: 1.55,  # N
                    8: 1.50,  # O
                    9: 1.50,  # F
                    14: 2.10, # Si
                    15: 1.85, # P
                    16: 1.80, # S
                    17: 1.70, # Cl
                }

                # Build a per-atom minima tensor without Python loops over atoms
                for z_val, min_val in element_minima.items():
                    mask = (Z == z_val)
                    if mask.any():
                        per_atom_min = torch.where(
                            mask,
                            torch.as_tensor(min_val, device=dev, dtype=dtype),
                            per_atom_min
                        )

                # Span is per-atom: [per_atom_min, global_max]
                span = global_max - per_atom_min
                # Avoid negative or zero span if config is odd
                span = torch.clamp(span, min=1e-6)

                # Final radii, differentiable inside the interval
                born_radii_full = u * span + per_atom_min

                # Numerical safety upper clamp to global_max
                born_radii_full = torch.minimum(born_radii_full, global_max)

            except AttributeError:
                # No atomic numbers available - fall back to global scaling
                span = cfg.gb_radii_scale[1] - cfg.gb_radii_scale[0]
                born_radii_full = u * span + cfg.gb_radii_scale[0]

            # Isolated radii from intraconv-only representation (faithful delta mode)
            if getattr(cfg, "gb_mode", "complex") == "delta_full":
                x_iso = self.intraconv_only(x0, sample.edge_index)
                if getattr(cfg, "gb_head_mode", "shared") == "split_sharestem":
                    h_iso = self.nn_born_radii_stem(x_iso)
                    u_iso = self.nn_born_radii_head_iso(h_iso).squeeze(-1)
                else:
                    u_iso = self.nn_born_radii(x_iso).squeeze(-1)
                try:
                    Z = sample.atomic_numbers
                    global_min = torch.as_tensor(cfg.gb_radii_scale[0], device=x_iso.device, dtype=u_iso.dtype)
                    global_max = torch.as_tensor(cfg.gb_radii_scale[1], device=x_iso.device, dtype=u_iso.dtype)
                    per_atom_min = torch.full_like(u_iso, global_min)
                    element_minima = {
                        1: 1.20,
                        6: 1.70,
                        7: 1.55,
                        8: 1.50,
                        9: 1.50,
                        14: 2.10,
                        15: 1.85,
                        16: 1.80,
                        17: 1.70,
                    }
                    for z_val, min_val in element_minima.items():
                        mask = (Z == z_val)
                        if mask.any():
                            per_atom_min = torch.where(
                                mask,
                                torch.as_tensor(min_val, device=x_iso.device, dtype=u_iso.dtype),
                                per_atom_min,
                            )
                    span = torch.clamp(global_max - per_atom_min, min=1e-6)
                    born_radii_iso = u_iso * span + per_atom_min
                    born_radii_iso = torch.minimum(born_radii_iso, global_max)
                except AttributeError:
                    span = cfg.gb_radii_scale[1] - cfg.gb_radii_scale[0]
                    born_radii_iso = u_iso * span + cfg.gb_radii_scale[0]

        # Prepare a pair-energies container: (energy_types, pairs)
        # Only inter-molecular terms are included here.
        # Base: vdW (Morse) + H-bond + Metal-Ligand + Hydrophobic = 4
        num_energy_types = 4
        if cfg.get("include_ionic", False):
            num_energy_types += 1
        energies_pairs = torch.empty(num_energy_types, D.numel()).to(self.device)
        energy_idx = 0

        # vdW energy minima (well depths): (pairs,)
        vdw_epsilon = self.nn_vdw_epsilon(x_cat).squeeze(-1)

        # Scale the minima as done in AutoDock Vina.
        vdw_epsilon = (
            vdw_epsilon * (cfg.vdw_epsilon_scale[1] - cfg.vdw_epsilon_scale[0])
            + cfg.vdw_epsilon_scale[0]
        )

        vdw_width = self.nn_vdw_width(x_cat).squeeze(-1)
        vdw_width = (
            vdw_width * (cfg.vdw_width_scale[1] - cfg.vdw_width_scale[0])
            + cfg.vdw_width_scale[0]
        )
        energies_pairs[energy_idx] = physics.morse_potential(
            D,
            R,
            vdw_epsilon,
            vdw_width,
            cfg.short_range_A,
        )
        energy_idx += 1

        minima_hbond = -(self.hbond_coeff**2)
        minima_metal_ligand = -(self.metal_ligand_coeff**2)
        minima_hydrophobic = -(self.hydrophobic_coeff**2)
        energies_pairs[energy_idx] = physics.linear_potential(
            D, R, minima_hbond, *cfg.hydrogen_bond_cutoffs
        )
        energy_idx += 1
        energies_pairs[energy_idx] = physics.linear_potential(
            D, R, minima_metal_ligand, *cfg.metal_ligand_cutoffs
        )
        energy_idx += 1
        energies_pairs[energy_idx] = physics.linear_potential(
            D, R, minima_hydrophobic, *cfg.hydrophobic_cutoffs
        )
        energy_idx += 1

        # Include the ionic interaction if required.
        if cfg.get("include_ionic", False):
            # Note the sign of `minima_ionic`
            minima_ionic = self.ionic_coeff**2 * (
                sample.partial_charges[edge_index_i[0]]
                * sample.partial_charges[edge_index_i[1]]
            )
            energies_pairs[energy_idx] = physics.linear_potential(
                D, R, minima_ionic, *cfg.ionic_cutoffs
            )
            energy_idx += 1

        # Include Generalized Born terms (pairwise: intra+inter; self)
        gb_pairwise_per_graph = None
        gb_self_per_graph = None
        gb_pairwise_delta = None
        gb_self_delta = None
        if cfg.get("include_gb", False):
            # All unique pairs within each graph
            edge_index_all = physics.all_pairs_edges(sample.batch)
            D_all = physics.distances(sample.pos, edge_index_all)
            _mask_all = (cfg.interaction_range[0] <= D_all) & (D_all <= cfg.interaction_range[1])
            edge_index_all = edge_index_all[:, _mask_all]
            D_all = D_all[_mask_all]

            gb_pairwise_all = physics.generalised_born_energy(
                D_all,
                born_radii_full,
                sample.partial_charges,
                edge_index_all,
                cfg.gb_dielectric_in,
                cfg.gb_dielectric_out,
            )
            gb_pairwise_per_graph = scatter(gb_pairwise_all, sample.batch[edge_index_all[0]])

            gb_self_energy = physics.self_born_energy(
                born_radii_full,
                sample.partial_charges,
                cfg.gb_dielectric_in,
                cfg.gb_dielectric_out,
            )
            gb_self_per_graph = scatter(gb_self_energy, sample.batch, dim=0)

            # Faithful delta mode: subtract isolated ligand and protein contributions computed with intraconv-only radii
            if getattr(cfg, "gb_mode", "complex") == "delta_full":
                ligand_mask = sample.is_ligand
                protein_mask = ~ligand_mask

                lig_pairs_mask = ligand_mask[edge_index_all[0]] & ligand_mask[edge_index_all[1]]
                edge_index_lig = edge_index_all[:, lig_pairs_mask]
                if edge_index_lig.numel() > 0:
                    D_lig = physics.distances(sample.pos, edge_index_lig)
                    _mask_lig = (cfg.interaction_range[0] <= D_lig) & (D_lig <= cfg.interaction_range[1])
                    edge_index_lig = edge_index_lig[:, _mask_lig]
                    D_lig = D_lig[_mask_lig]
                    gb_pair_lig = physics.generalised_born_energy(
                        D_lig, born_radii_iso, sample.partial_charges, edge_index_lig,
                        cfg.gb_dielectric_in, cfg.gb_dielectric_out,
                    )
                    gb_pair_lig_sum = scatter(gb_pair_lig, sample.batch[edge_index_lig[0]])
                else:
                    gb_pair_lig_sum = torch.zeros_like(gb_pairwise_per_graph)

                pro_pairs_mask = protein_mask[edge_index_all[0]] & protein_mask[edge_index_all[1]]
                edge_index_pro = edge_index_all[:, pro_pairs_mask]
                if edge_index_pro.numel() > 0:
                    D_pro = physics.distances(sample.pos, edge_index_pro)
                    _mask_pro = (cfg.interaction_range[0] <= D_pro) & (D_pro <= cfg.interaction_range[1])
                    edge_index_pro = edge_index_pro[:, _mask_pro]
                    D_pro = D_pro[_mask_pro]
                    gb_pair_pro = physics.generalised_born_energy(
                        D_pro, born_radii_iso, sample.partial_charges, edge_index_pro,
                        cfg.gb_dielectric_in, cfg.gb_dielectric_out,
                    )
                    gb_pair_pro_sum = scatter(gb_pair_pro, sample.batch[edge_index_pro[0]])
                else:
                    gb_pair_pro_sum = torch.zeros_like(gb_pairwise_per_graph)

                gb_self_iso = physics.self_born_energy(
                    born_radii_iso, sample.partial_charges,
                    cfg.gb_dielectric_in, cfg.gb_dielectric_out,
                )
                gb_self_lig_sum = scatter(gb_self_iso[ligand_mask], sample.batch[ligand_mask], dim=0)
                gb_self_pro_sum = scatter(gb_self_iso[protein_mask], sample.batch[protein_mask], dim=0)

                gb_pairwise_delta = gb_pairwise_per_graph - gb_pair_lig_sum - gb_pair_pro_sum
                gb_self_delta = gb_self_per_graph - gb_self_lig_sum - gb_self_pro_sum

        # Interaction masks according to atom types: (energy_types, pairs)
        masks = physics.interaction_masks(
            sample.is_metal,
            sample.is_h_donor,
            sample.is_h_acceptor,
            sample.is_hydrophobic,
            edge_index_i,
            include_ionic=cfg.get("include_ionic", False),
        )

        energies_pairs = energies_pairs * masks
        # Per-graph sum -> (energy_types, batch)
        energies = scatter(energies_pairs, sample.batch[edge_index_i[0]], dim=1)
        # Reshape -> (batch, energy_types)
        energies = energies.t().contiguous()

        # Append GB pairwise and self energies per graph as separate columns
        if cfg.get("include_gb", False):
            if getattr(cfg, "gb_mode", "complex") == "delta_full":
                energies = torch.cat([
                    energies,
                    gb_pairwise_delta.unsqueeze(1),
                    gb_self_delta.unsqueeze(1),
                ], dim=1)
            else:
                energies = torch.cat([
                    energies,
                    gb_pairwise_per_graph.unsqueeze(1),
                    gb_self_per_graph.unsqueeze(1),
                ], dim=1)

        # Rotor penalty
        if cfg.rotor_penalty:
            penalty = 1 + self.rotor_coeff**2 * sample.rotor
            # -> (batch, 1)
            energies = energies / penalty

        # Expose last per-atom Born radii for inspection after inference
        self.last_born_radii = born_radii_full

        return energies, dvdw_radii