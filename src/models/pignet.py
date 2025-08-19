from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import Dropout, Module, ModuleList, Parameter, ReLU, Sigmoid, Tanh
from torch.nn.parameter import UninitializedParameter
from torch.optim import AdamW
from torch_geometric.data import Batch
from torch_geometric.nn import Linear, Sequential
from torch_scatter import scatter

from . import physics
from .layers import GatedGAT, InteractionNet


class PIGNet(Module):
    def __init__(
        self,
        config: DictConfig,
        in_features: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.reset_log()
        self.config = config
        n_gnn = config.model.n_gnn
        dim_gnn = config.model.dim_gnn
        dim_mlp = config.model.dim_mlp
        dropout_rate = config.run.dropout_rate

        self.embed = Linear(in_features, dim_gnn, bias=False)

        self.intraconv = ModuleList()
        for _ in range(n_gnn):
            self.intraconv.append(
                Sequential(
                    "x, edge_index",
                    [
                        (GatedGAT(dim_gnn, dim_gnn), "x, edge_index -> x"),
                        (Dropout(dropout_rate), "x -> x"),
                    ],
                )
            )

        self.interconv = ModuleList()
        if config.model.interconv:
            for _ in range(n_gnn):
                self.interconv.append(
                    Sequential(
                        "x, edge_index",
                        [
                            (InteractionNet(dim_gnn), "x, edge_index -> x"),
                            (Dropout(dropout_rate), "x -> x"),
                        ],
                    )
                )

        self.nn_vdw_epsilon = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                Sigmoid(),
            ],
        )

        self.nn_dvdw = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                Tanh(),
            ],
        )

        # Born radii prediction MLP 
        self.nn_born_radii = Sequential(
            "x",
            [
                (Linear(dim_gnn, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                ReLU(),  # Ensure positive radii
            ],
        )

        self.hbond_coeff = Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = Parameter(torch.tensor([0.5]))
        self.rotor_coeff = Parameter(torch.tensor([0.5]))
                
        if config.model.get("include_ionic", False):
            self.ionic_coeff = Parameter(torch.tensor([1.0]))

    @property
    def size(self) -> Tuple[int, int]:
        """Get the number of all learnable parameters.

        Returns: (num_parameters, num_uninitialized_parameters)
        """
        num_params = 0
        num_uninitialized = 0

        for param in self.parameters():
            if isinstance(param, UninitializedParameter):
                num_uninitialized += 1
            elif param.requires_grad:
                num_params += param.numel()

        return num_params, num_uninitialized

    @property
    def in_features(self) -> int:
        """Get the number of input features."""
        try:
            return self.embed.in_channels
        except AttributeError:
            return self.embed.in_features

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def conv(self, x, edge_index_1, edge_index_2):
        for conv in self.intraconv:
            x = conv(x, edge_index_1)

        for conv in self.interconv:
            x = conv(x, edge_index_2)
        return x

    def intraconv_only(self, x, edge_index_1):
        """Apply only intra-molecular convolutions (no inter-molecular pass)."""
        for conv in self.intraconv:
            x = conv(x, edge_index_1)
        return x

    def forward(self, sample: Batch):
        cfg = self.config.model

        # Initial embedding
        x0 = self.embed(sample.x)

        # Graph convolutions (full: intra + inter)
        x = self.conv(x0, sample.edge_index, sample.edge_index_c)

        # Predict Born radii for each atom (per-atom prediction)
        born_radii_full = None
        born_radii_iso = None
        if cfg.get("include_gb", False):
            # Complex (full) radii from full conv representation
            born_radii_full = self.nn_born_radii(x).view(-1)
            born_radii_full = born_radii_full * (cfg.gb_radii_scale[1] - cfg.gb_radii_scale[0]) + cfg.gb_radii_scale[0]
            # Isolated radii from intraconv-only representation (faithful delta mode)
            if getattr(cfg, "gb_mode", "complex") == "delta_full":
                x_iso = self.intraconv_only(x0, sample.edge_index)
                born_radii_iso = self.nn_born_radii(x_iso).view(-1)
                born_radii_iso = born_radii_iso * (cfg.gb_radii_scale[1] - cfg.gb_radii_scale[0]) + cfg.gb_radii_scale[0]

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

        # Prepare a pair-energies container: (energy_types, pairs)
        # Only inter-molecular terms are included here.
        num_energy_types = 4
        if cfg.get("include_ionic", False):
            num_energy_types += 1
        
        energies_pairs = torch.empty(num_energy_types, D.numel()).to(self.device)
        energy_idx = 0

        # vdW energy minima (well depths): (pairs,)
        vdw_epsilon = self.nn_vdw_epsilon(x_cat).view(-1)
        # Scale the minima as done in AutoDock Vina.
        vdw_epsilon = (
            vdw_epsilon * (cfg.vdw_epsilon_scale[1] - cfg.vdw_epsilon_scale[0])
            + cfg.vdw_epsilon_scale[0]
        )
        # vdW interaction
        energies_pairs[energy_idx] = physics.lennard_jones_potential(
            D, R, vdw_epsilon, cfg.vdw_N_short, cfg.vdw_N_long
        )
        energy_idx += 1

        # Hydrogen-bond, metal-ligand, hydrophobic interactions
        minima_hbond = -(self.hbond_coeff**2)
        minima_hydrophobic = -(self.hydrophobic_coeff**2)
        energies_pairs[energy_idx] = physics.linear_potential(
            D, R, minima_hbond, *cfg.hydrogen_bond_cutoffs
        )
        energy_idx += 1
        energies_pairs[energy_idx] = physics.linear_potential(
            D, R, minima_hbond, *cfg.metal_ligand_cutoffs
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
            # All unique pairs within each graph (intra-ligand, intra-protein, and inter)
            edge_index_all = physics.all_pairs_edges(sample.batch)
            D_all = physics.distances(sample.pos, edge_index_all)
            _mask_all = (cfg.interaction_range[0] <= D_all) & (D_all <= cfg.interaction_range[1])
            edge_index_all = edge_index_all[:, _mask_all]
            D_all = D_all[_mask_all]

            gb_pairwise_all = physics.generalised_born_energy(
                D_all, born_radii_full, sample.partial_charges, edge_index_all,
                cfg.gb_dielectric_in, cfg.gb_dielectric_out
            )
            # Sum pairwise GB per graph
            gb_pairwise_per_graph = scatter(gb_pairwise_all, sample.batch[edge_index_all[0]])

            # GB self-energy (per atom) -> per-graph sum
            gb_self_energy = physics.self_born_energy(
                born_radii_full, sample.partial_charges,
                cfg.gb_dielectric_in, cfg.gb_dielectric_out
            )
            gb_self_per_graph = scatter(gb_self_energy, sample.batch, dim=0)

            # Faithful delta mode: subtract isolated ligand and protein contributions computed with intraconv-only radii
            if getattr(cfg, "gb_mode", "complex") == "delta_full":
                ligand_mask = sample.is_ligand
                protein_mask = ~ligand_mask

                # Ligand-only pairs
                lig_pairs_mask = ligand_mask[edge_index_all[0]] & ligand_mask[edge_index_all[1]]
                edge_index_lig = edge_index_all[:, lig_pairs_mask]
                if edge_index_lig.numel() > 0:
                    D_lig = physics.distances(sample.pos, edge_index_lig)
                    _mask_lig = (cfg.interaction_range[0] <= D_lig) & (D_lig <= cfg.interaction_range[1])
                    edge_index_lig = edge_index_lig[:, _mask_lig]
                    D_lig = D_lig[_mask_lig]
                    gb_pair_lig = physics.generalised_born_energy(
                        D_lig, born_radii_iso, sample.partial_charges, edge_index_lig,
                        cfg.gb_dielectric_in, cfg.gb_dielectric_out
                    )
                    gb_pair_lig_sum = scatter(gb_pair_lig, sample.batch[edge_index_lig[0]])
                else:
                    gb_pair_lig_sum = torch.zeros_like(gb_pairwise_per_graph)

                # Protein-only pairs
                pro_pairs_mask = protein_mask[edge_index_all[0]] & protein_mask[edge_index_all[1]]
                edge_index_pro = edge_index_all[:, pro_pairs_mask]
                if edge_index_pro.numel() > 0:
                    D_pro = physics.distances(sample.pos, edge_index_pro)
                    _mask_pro = (cfg.interaction_range[0] <= D_pro) & (D_pro <= cfg.interaction_range[1])
                    edge_index_pro = edge_index_pro[:, _mask_pro]
                    D_pro = D_pro[_mask_pro]
                    gb_pair_pro = physics.generalised_born_energy(
                        D_pro, born_radii_iso, sample.partial_charges, edge_index_pro,
                        cfg.gb_dielectric_in, cfg.gb_dielectric_out
                    )
                    gb_pair_pro_sum = scatter(gb_pair_pro, sample.batch[edge_index_pro[0]])
                else:
                    gb_pair_pro_sum = torch.zeros_like(gb_pairwise_per_graph)

                # Self-energy isolated sums
                gb_self_iso = physics.self_born_energy(
                    born_radii_iso, sample.partial_charges,
                    cfg.gb_dielectric_in, cfg.gb_dielectric_out
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
        
        # Apply masks (GB handled separately per-graph)
        
        energies_pairs = energies_pairs * masks

        # Per-graph sum -> (energy_types, batch)
        energies = scatter(energies_pairs, sample.batch[edge_index_i[0]], dim=1)
        # Reshape -> (batch, energy_types)
        energies = energies.t().contiguous()
        
        # Append GB energies per graph
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

        return energies, dvdw_radii

    def loss_dvdw(self, dvdw_radii: torch.Tensor):
        loss = dvdw_radii.pow(2).mean()
        return loss

    def loss_born_radii(self, born_radii: torch.Tensor, sample: Batch):
        """Default Born radii regularizer (base model): simple margin-from-bounds penalty.

        Subclasses can override for element-specific priors or different behavior.
        """
        if born_radii is None:
            return torch.tensor(0.0, device=self.device)

        cfg = self.config.model
        rmin, rmax = cfg.gb_radii_scale[0], cfg.gb_radii_scale[1]

        # Determine absolute margin
        margin = cfg.get("gb_bound_margin", None)
        if margin is None:
            margin_frac = cfg.get("gb_bound_margin_fraction", 0.1)
            margin = float(margin_frac) * (float(rmax) - float(rmin))

        lower_bound = float(rmin) + margin
        upper_bound = float(rmax) - margin

        # Hinge penalties if radii hug the bounds
        lower_violation = torch.relu(lower_bound - born_radii)
        upper_violation = torch.relu(born_radii - upper_bound)
        loss = (lower_violation + upper_violation).mean()
        return loss

    def loss_regression(
        self,
        energies: torch.Tensor,
        true: torch.Tensor,
    ):
        return F.mse_loss(energies.sum(-1, True), true)

    def loss_augment(
        self,
        energies: torch.Tensor,
        true: torch.Tensor,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ):
        """Loss functions for docking, random & cross screening.

        Args:
            sample
            task: 'docking' | 'random' | 'cross'
        """
        loss_energy = true - energies.sum(-1, True)
        loss_energy = loss_energy.clamp(min, max)
        loss_energy = loss_energy.mean()
        return loss_energy

    def training_step(self, batch: Dict[str, Batch]):
        loss_total = torch.tensor(0.0, device=self.device)

        for task, sample in batch.items():
            task_config = self.config.data[task]

            # Updated forward pass returns born_radii as well
            forward_result = self(sample)
            if len(forward_result) == 3:
                energies, dvdw_radii, born_radii = forward_result
            else:
                energies, dvdw_radii = forward_result
                born_radii = None

            loss_dvdw = self.loss_dvdw(dvdw_radii)
            loss_born = self.loss_born_radii(born_radii, sample)
            
            if task_config.objective == "regression":
                loss_energy = self.loss_regression(energies, sample.y)
            elif task_config.objective == "augment":
                loss_energy = self.loss_augment(
                    energies, sample.y, *task_config.loss_range
                )
            else:
                raise NotImplementedError(
                    "Current loss functions only support regression and augment."
                )

            loss_total += loss_energy * task_config.loss_ratio
            loss_total += loss_dvdw * self.config.run.loss_dvdw_ratio
            
            # Add GB loss if GB is enabled
            if self.config.model.get("include_gb", False):
                loss_total += loss_born * self.config.run.get("loss_gb_ratio", 0.1)

            # Update log
            self.losses["energy"][task].append(loss_energy.item())
            self.losses["dvdw"][task].append(loss_dvdw.item())
            if self.config.model.get("include_gb", False):
                self.losses["gb"][task].append(loss_born.item())
            
            for key, pred, true in zip(sample.key, energies, sample.y):
                self.predictions[task][key] = pred.tolist()
                self.labels[task][key] = true.item()

        return loss_total

    def validation_step(self, batch: Dict[str, Batch]):
        return self.training_step(batch)

    def test_step(self, batch: Batch):
        sample = batch
        task = next(iter(self.config.data))

        energies, dvdw_radii = self(sample)
        loss_energy = self.loss_regression(energies, sample.y)
        loss_dvdw = self.loss_dvdw(dvdw_radii)

        # Update log
        self.losses["energy"][task].append(loss_energy.item())
        self.losses["dvdw"][task].append(loss_dvdw.item())
        for key, pred, true in zip(sample.key, energies, sample.y):
            self.predictions[task][key] = pred.tolist()
            self.labels[task][key] = true.item()

    def predict_step(self, batch: Batch):
        sample = batch
        task = next(iter(self.config.data))
        energies, dvdw_radii = self(sample)
        for key, pred in zip(sample.key, energies):
            self.predictions[task][key] = pred.tolist()

    def configure_optimizers(self):
        lr = float(self.config.run.lr)
        weight_decay = float(self.config.run.weight_decay)

        decay_params = []
        no_decay_params = []

        physics_coeff_names = {
            "hbond_coeff",
            "hydrophobic_coeff",
            "rotor_coeff",
            "ionic_coeff",
        }

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            is_bias = name.endswith(".bias") or name == "bias"
            is_physics_coeff = any(name.endswith(n) for n in physics_coeff_names)
            is_norm_like = param.dim() == 1

            if (not is_bias) and (not is_norm_like) and (not is_physics_coeff):
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

        return AdamW(param_groups, lr=lr)

    def reset_log(self):
        """Reset logs. Intended to be called every epoch.

        Attributes:
            losses: Dict[str, Dict[str, List[float]]]
                losses[loss_type][task] -> loss_values
                where
                    loss_type: 'energy' | 'dvdw'
                    task: 'scoring' | 'docking' | 'random' | 'cross' | ...
                    loss_values: List[float] of shape (batches,)

            predictions: Dict[str, Dict[str, Tuple[float, ...]]]
                predictions[task][key] -> energies
                where
                    energies: List[float] of shape (4,)

            labels: Dict[str, Dict[str, float]]
                labels[task][key] -> energy (float)
        """
        self.losses = defaultdict(lambda: defaultdict(list))
        self.predictions = defaultdict(dict)
        self.labels = defaultdict(dict)
