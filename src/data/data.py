import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdmolops
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse

from . import chem


def one_hot_encode(
    x: Any,
    kinds: List[Any],
    handle_unknown: str = "error",
) -> List[bool]:
    """
    Make a one-hot vector.

    Args:
        handle_unknown: 'error' | 'ignore' | 'last'
            If `x` not in `kinds`:
                'error' -> raise ValueError
                'ignore' -> return zero vector
                'last' -> use the last kind.
    """
    onehot = [False] * len(kinds)
    try:
        onehot[kinds.index(x)] = True

    except ValueError:
        if handle_unknown == "error":
            msg = f"input {x} not in the allowed set {kinds}"
            raise ValueError(msg)
        elif handle_unknown == "ignore":
            pass
        elif handle_unknown == "last":
            onehot[-1] = True
        else:
            raise NotImplementedError

    return onehot


def get_period_group(atom: Chem.Atom) -> List[bool]:
    period, group = chem.PERIODIC_TABLE[atom.GetSymbol().upper()]
    period_vec = one_hot_encode(period, chem.PERIODS)
    group_vec = one_hot_encode(group, chem.GROUPS)
    total_vec = period_vec + group_vec
    return total_vec


def get_vdw_radius(atom: Chem.Atom) -> float:
    atomic_number = atom.GetAtomicNum()
    try:
        radius = chem.VDW_RADII[atomic_number]
    except KeyError:
        radius = Chem.GetPeriodicTable().GetRvdw(atomic_number)
    return radius


def get_atom_charges(mol: Chem.Mol) -> List[float]:
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    return charges


def get_metals(mol: Chem.Mol) -> List[bool]:
    mask = [atom.GetSymbol() in chem.METALS for atom in mol.GetAtoms()]
    return mask


def get_smarts_matches(mol: Chem.Mol, smarts: str) -> List[bool]:
    # Get the matching atom indices.
    pattern = Chem.MolFromSmarts(smarts)
    matches = {idx for match in mol.GetSubstructMatches(pattern) for idx in match}

    # Convert to a mask vector.
    mask = [idx in matches for idx in range(mol.GetNumAtoms())]
    return mask


def get_hydrophobes(mol: Chem.Mol) -> List[bool]:
    mask = []

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol().upper()
        if symbol in chem.HYDROPHOBES:
            # Check if all neighbors are also in `hyd_atoms`.
            neighbor_symbols = {
                neighbor.GetSymbol().upper() for neighbor in atom.GetNeighbors()
            }
            neighbors_not_hyd = neighbor_symbols - chem.HYDROPHOBES
            mask.append(len(neighbors_not_hyd) == 0)
        else:
            mask.append(False)

    return mask


def atom_to_features(atom: Chem.Atom) -> List[bool]:
    # Total 47, currently.
    features = [
        # Symbol (10)
        one_hot_encode(atom.GetSymbol(), chem.ATOM_SYMBOLS, "last"),
        # Degree (6)
        one_hot_encode(atom.GetDegree(), chem.ATOM_DEGREES, "last"),
        # Hybridization (7)
        one_hot_encode(atom.GetHybridization(), chem.HYBRIDIZATIONS, "last"),
        # Period & group (23)
        get_period_group(atom),
        # Aromaticity (1)
        [atom.GetIsAromatic()],
    ]

    # Flatten
    features = [value for row in features for value in row]
    return features


def mol_to_data(
    mol: Chem.Mol,
    remove_hydrogens: bool = True,
    pos_noise_std: float = 0.0,
    pos_noise_max: float = 0.0,
) -> Data:
    """Convert a RDKit mol to PyG data.
    Every numerical attributes are converted to torch.tensor.
    Note that label `y` is not set here.

    Data attributes:
        x: (num_atoms, num_atom_features), float
        edge_index: (2, num_bonds), long
        pos: (num_atoms, 3), float
        vdw_radii: (num_atoms,), float
        is_metal: (num_atoms,), bool
        is_h_donor: (num_atoms,), bool
        is_h_acceptor: (num_atoms,), bool
        is_hydrophobic: (num_atoms,), bool
    """
    if remove_hydrogens:
        mol = Chem.RemoveAllHs(mol)

    # Get all atoms at once to avoid repeated calls
    atoms = list(mol.GetAtoms())
    num_atoms = len(atoms)

    # Pre-allocate tensors for better memory efficiency
    data = Data()
    
    # Node features - vectorized operation
    x = torch.tensor([atom_to_features(atom) for atom in atoms], dtype=torch.float)
    data.x = x

    # Adjacency matrix - single operation
    adj = torch.tensor(rdmolops.GetAdjacencyMatrix(mol))
    edge_index, _ = dense_to_sparse(adj)
    data.edge_index = edge_index

    # Cartesian coordinates
    try:
        pos = mol.GetConformers()[0].GetPositions()
    except IndexError:
        msg = "No position in the `Chem.Mol` data!"
        raise RuntimeError(msg)
    data.pos = torch.tensor(pos, dtype=torch.float)

    # Add noise - MODIFIED SECTION TO MATCH CODE 2 LOGIC
    noise = torch.zeros_like(data.pos)
    if pos_noise_std and pos_noise_max:
        noise += torch.normal(0, pos_noise_std, size=noise.shape)
        noise.clamp_(-pos_noise_max, pos_noise_max)
    elif pos_noise_std:
        noise += torch.normal(0, pos_noise_std, size=noise.shape)
    elif pos_noise_max:
        noise += (pos_noise_max * 2) * torch.rand(noise.shape) - pos_noise_max
    data.pos += noise
    # END OF MODIFIED SECTION

    # Pre-compute all atom properties in parallel
    vdw_radii = torch.tensor([get_vdw_radius(atom) for atom in atoms], dtype=torch.float)
    atom_charges = torch.tensor([atom.GetFormalCharge() for atom in atoms], dtype=torch.float)
    metals = torch.tensor([atom.GetSymbol() in chem.METALS for atom in atoms], dtype=torch.bool)
    
    # SMARTS patterns - compute once and reuse
    pattern = Chem.MolFromSmarts(chem.H_DONOR_SMARTS)
    h_donor_matches = {idx for match in mol.GetSubstructMatches(pattern) for idx in match}
    pattern = Chem.MolFromSmarts(chem.H_ACCEPTOR_SMARTS)
    h_acceptor_matches = {idx for match in mol.GetSubstructMatches(pattern) for idx in match}
    
    # Convert matches to masks
    h_donors = torch.tensor([idx in h_donor_matches for idx in range(num_atoms)], dtype=torch.bool)
    h_acceptors = torch.tensor([idx in h_acceptor_matches for idx in range(num_atoms)], dtype=torch.bool)
    
    # Hydrophobes - optimized version
    hydrophobes = []
    for atom in atoms:
        symbol = atom.GetSymbol().upper()
        if symbol in chem.HYDROPHOBES:
            neighbor_symbols = {neighbor.GetSymbol().upper() for neighbor in atom.GetNeighbors()}
            hydrophobes.append(len(neighbor_symbols - chem.HYDROPHOBES) == 0)
        else:
            hydrophobes.append(False)
    hydrophobes = torch.tensor(hydrophobes, dtype=torch.bool)

    # Assign all properties at once
    data.vdw_radii = vdw_radii
    data.atom_charges = atom_charges
    data.is_metal = metals
    data.is_h_donor = h_donors
    data.is_h_acceptor = h_acceptors
    data.is_hydrophobic = hydrophobes

    return data


def get_complex_edges(
    pos1: torch.Tensor,
    pos2: torch.Tensor,
    min_distance: float,
    max_distance: float,
) -> torch.LongTensor:
    """\
    Args:
        pos1: (num_atoms1, 3)
        pos2: (num_atoms2, 3)
        min_distance, max_distance:
            Atoms a_i and a_j are deemed connected if:
                min_distance <= d_ij <= max_distance
    """
    # Optimized distance calculation using cdist
    D = torch.cdist(pos1, pos2)
    
    # Create mask for distances within range
    mask = (min_distance <= D) & (D <= max_distance)
    
    # Get indices of connected atoms
    i, j = torch.nonzero(mask, as_tuple=True)
    
    # Shift j indices by pos1.size(0)
    j_shifted = j + pos1.size(0)
    
    # Create bidirectional edges
    edge_index = torch.stack([
        torch.cat([i, j_shifted]),
        torch.cat([j_shifted, i])
    ])
    
    # Handle case with no edges
    if edge_index.numel() == 0:
        edge_index = edge_index.view(2, 0).long()
    else:
        edge_index = edge_index.long()
    
    return edge_index


def complex_to_data(
    mol_ligand: Chem.Mol,
    mol_target: Chem.Mol,
    label: Optional[float] = None,
    key: Optional[str] = None,
    conv_range: Tuple[float, float] = None,
    remove_hydrogens: bool = True,
    pos_noise_std: float = 0.0,
    pos_noise_max: float = 0.0,
) -> Data:
    """\
    Data attributs (additional to `mol_to_data`):
        y: (1, 1), float
        key: str
        rotor: (1, 1), float
        is_ligand: (num_ligand_atoms + num_target_atoms,), bool
        edge_index_c: (2, num_edges), long
            Intermolecular edges for graph convolution.
        mol_ligand: Chem.Mol
            Ligand Mol object used for docking.
        mol_target: Chem.Mol
            Target Mol object used for docking.
    """
    ligand = mol_to_data(mol_ligand, remove_hydrogens, pos_noise_std, pos_noise_max)
    target = mol_to_data(mol_target, remove_hydrogens, pos_noise_std, pos_noise_max)
    data = Data()

    if remove_hydrogens:
        mol_ligand = Chem.RemoveAllHs(mol_ligand)
        mol_target = Chem.RemoveAllHs(mol_target)

    # Combine the values.
    assert set(ligand.keys()) == set(target.keys())
    for attr in ligand.keys():
        ligand_value = ligand[attr]
        target_value = target[attr]

        # Shift atom indices for some attributes.
        if attr in ("edge_index",):
            target_value = target_value + ligand.num_nodes

        # Dimension to concatenate over.
        cat_dim = ligand.__cat_dim__(attr, None)
        value = torch.cat((ligand_value, target_value), cat_dim)
        data[attr] = value

    if label is not None:
        data.y = torch.tensor(label, dtype=torch.float).view(1, 1)

    if key is not None:
        data.key = key

    rotor = rdMolDescriptors.CalcNumRotatableBonds(mol_ligand)
    data.rotor = torch.tensor(rotor, dtype=torch.float).view(1, 1)

    # Ligand mask
    is_ligand = [True] * ligand.num_nodes + [False] * target.num_nodes
    data.is_ligand = torch.tensor(is_ligand)

    # Intermolecular edges
    if conv_range is not None:
        data.edge_index_c = get_complex_edges(ligand.pos, target.pos, *conv_range)

    # Save the Mol objects; used for docking.
    data.mol_ligand = mol_ligand
    data.mol_target = mol_target

    return data


class ComplexDataset(Dataset):
    def __init__(
        self,
        keys: List[str],
        data_dir: Optional[str] = None,
        id_to_y: Optional[Dict[str, float]] = None,
        conv_range: Optional[Tuple[float, float]] = None,
        processed_data_dir: Optional[str] = None,
        pos_noise_std: float = 0.0,
        pos_noise_max: float = 0.0,
        num_workers: int = 0,
        cache_size: int = 1000,  # Add cache size parameter
    ):
        assert data_dir is not None or processed_data_dir is not None

        super().__init__()
        self.keys = keys
        self.data_dir = data_dir
        self.id_to_y = id_to_y
        self.conv_range = conv_range
        self.processed_data_dir = processed_data_dir
        self.pos_noise_std = pos_noise_std
        self.pos_noise_max = pos_noise_max
        self.num_workers = num_workers
        self.cache_size = cache_size
        
        # Initialize cache
        self._cache = {}
        self._cache_order = []
        
        # Pre-compute labels if available
        if id_to_y is not None:
            self.labels = {k: v * -1.36 for k, v in id_to_y.items()}
        else:
            self.labels = None

    def _update_cache(self, key: str, data: Data):
        """Update the cache with LRU policy."""
        if key in self._cache:
            self._cache_order.remove(key)
        elif len(self._cache) >= self.cache_size:
            # Remove least recently used item
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = data
        self._cache_order.append(key)

    def len(self) -> int:
        return len(self.keys)

    def get(self, idx) -> Data:
        key = self.keys[idx]
        
        # Check cache first
        if key in self._cache:
            return self._cache[key]

        # Setting 'processed_data_dir' takes priority than 'data_dir'.
        if self.processed_data_dir is not None:
            data_path = os.path.join(self.processed_data_dir, key + ".pt")
            try:
                data = torch.load(data_path, map_location='cpu')
                self._update_cache(key, data)
                return data
            except Exception as e:
                print(f"Error loading {data_path}: {e}")
                return None

        elif self.data_dir is not None:
            data_path = os.path.join(self.data_dir, key)
            try:
                # Load data with error handling
                with open(data_path, "rb") as f:
                    data_mol_info = pickle.load(f) # Renamed to avoid conflict with 'data' PyG object
                    try:
                        mol_ligand, _, mol_target, _ = data_mol_info
                    except ValueError:
                        try:
                            mol_ligand, mol_target = data_mol_info
                        except ValueError: # Added to catch if data_mol_info is already a PyG-like object with .mol_ligand
                            mol_ligand, mol_target = data_mol_info.mol_ligand, data_mol_info.mol_target


                # Get label if available
                label = self.labels.get(key) if self.labels is not None else None

                data = complex_to_data(
                    mol_ligand,
                    mol_target,
                    label,
                    key,
                    self.conv_range,
                    # Pass the dataset's noise parameters to mol_to_data
                    pos_noise_std=self.pos_noise_std, 
                    pos_noise_max=self.pos_noise_max,
                )
                self._update_cache(key, data)
                return data
            except Exception as e:
                print(f"Error processing {data_path}: {e}")
                return None

        return None

    def process(self):
        """Pre-process all data and save to processed_data_dir if specified."""
        if self.processed_data_dir is None:
            return

        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Process data in parallel if num_workers > 0
        if self.num_workers > 0:
            from concurrent.futures import ProcessPoolExecutor
            
            # Helper function to pass instance methods for ProcessPoolExecutor
            def process_item(item_key):
                # When loading from data_dir, self.get will call complex_to_data,
                # which calls mol_to_data with self.pos_noise_std and self.pos_noise_max.
                # The data is then saved.
                processed_data = self.get(self.keys.index(item_key)) 
                if processed_data is not None:
                    # Saving logic is now inside _process_single, called below if not parallel
                    # For parallel, we need to ensure _process_single is called or its logic replicated
                    save_path = os.path.join(self.processed_data_dir, item_key + ".pt")
                    if not os.path.exists(save_path): # Avoid re-saving if get() already loaded it from processed
                         torch.save(processed_data, save_path)


            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # We map over keys, as _process_single takes a key
                # The list(executor.map(...)) ensures all tasks complete
                list(executor.map(self._process_single, self.keys))
        else:
            for key_idx in range(len(self.keys)): # Iterate by index to use self.get(idx)
                # self._process_single will call self.get(idx_of_key)
                self._process_single(self.keys[key_idx])


    def _process_single(self, key: str):
        """Process and save a single data point."""
        if self.processed_data_dir is None:
            return

        data_path = os.path.join(self.processed_data_dir, key + ".pt")
        if os.path.exists(data_path):
            return # Already processed

        # Get data using the key. This will use the dataset's pos_noise_std/max.
        # self.get needs an index, so find the index of the key.
        try:
            idx = self.keys.index(key)
            data = self.get(idx) # This call will use self.pos_noise_std and self.pos_noise_max from the dataset instance
        except ValueError:
            print(f"Key {key} not found in dataset keys during processing.")
            return
        
        if data is not None:
            torch.save(data, data_path)