#!/usr/bin/env python
import argparse
import inspect
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from protonate import protonate_ligand, protonate_pdb
from pymol import cmd, chempy
from rdkit import Chem
from rdkit.Chem import AllChem

PathLike = Union[str, os.PathLike]

MAX_NUM_RETRIALS = 20

def extract(
    receptor_pdb: Path,
    ligand_sdf: Path,
    ligand_idx: int = 1,
    distance: float = 5.0,
    tmp_ext: str = ".sdf",
    correct_oxygens: bool = False,
    unbond_metals: bool = True,
    remove_h_in_tmp_save: bool = True,
    remove_h_in_final_save: bool = True,
    ligand_sdf_is_large: bool = False,
    retry_with_obabel: bool = True,
    retry_with_cutoff: bool = True,
    connect_cutoff: Optional[float] = None,
    tmp_dir: Optional[Union[str, Path]] = None,
    silent: bool = False,
    pqr_path: Optional[Union[str, Path]] = None,
    charge_tolerance: float = 0.001,
    include_pqr_hydrogens: bool = False,
) -> Optional[AllChem.Mol]:
    cmd.reinitialize()
    cmd.set("max_threads", 1)
    if connect_cutoff is not None:
        cmd.set("connect_cutoff", connect_cutoff)

    # Load and clean the protein.
    cmd.load(receptor_pdb, "prot")
    if unbond_metals:
        cmd.unbond("metals", "*")
    if remove_h_in_tmp_save:
        cmd.remove("h.")
    assert cmd.count_atoms("prot"), f"No protein found in {receptor_pdb}"

    # Load the ligand as name 'lig'.
    if ligand_sdf_is_large:
        tmp_sdf = extract_ith_mol(ligand_sdf, ligand_idx)
        cmd.load(tmp_sdf, "lig")
        os.remove(tmp_sdf)
    else:
        cmd.load(ligand_sdf, "lig_all")
        cmd.create("lig", f"%lig_all and state {ligand_idx}")
        cmd.delete("%lig_all")
    assert cmd.count_atoms("lig"), f"No ligand found in {ligand_sdf}"

    # Extract the pocket.
    cmd.create("pocket", f"br. (%prot and not h.) w. {distance} of (%lig and not h.)")
    cmd.delete("%lig")
    if not cmd.count_atoms("%pocket"):
        return

    # Clean the pocket.
    if correct_oxygens:
        _neutralize_pi_oxygens()
        if cmd.count_atoms("het"):
            _rebond_monovalent_oxygens()

    # Transfer the pocket into `AllChem.Mol`.
    tmp_ext = "." + tmp_ext.lstrip(".")
    fd, tmp_path = tempfile.mkstemp(suffix=tmp_ext, dir=tmp_dir)
    os.close(fd)
    cmd.save(tmp_path, "%pocket")
    (pocket,) = read_mols(tmp_path, removeHs=remove_h_in_final_save)
    # Assign partial charges from PQR if provided
    if pocket is not None and pqr_path is not None:
        try:
            pqr_models = _read_pqr_models(pqr_path, include_hydrogens=include_pqr_hydrogens)
            if pqr_models:
                _assign_partial_charges_by_coords(pocket, pqr_models[0], tolerance=charge_tolerance)
        except Exception as e:
            if not silent:
                print(f"Warning: failed assigning PQR charges to pocket: {e}", file=sys.stderr)

    # If failed, retry after obabel re-save.
    if pocket is None and retry_with_obabel:
        if not silent:
            print(
                f"{receptor_pdb}, {ligand_sdf}, {tmp_path}:",
                "Retrying with obabel",
                file=sys.stderr,
            )
        from openbabel import pybel

        fd, tmp_path2 = tempfile.mkstemp(suffix=tmp_ext, dir=tmp_dir)
        os.close(fd)
        pybel_mol = next(pybel.readfile(tmp_ext.lstrip("."), tmp_path))
        pybel_mol.write(tmp_ext.lstrip("."), tmp_path2, overwrite=True)

        (pocket,) = read_mols(tmp_path2, removeHs=remove_h_in_final_save)
        # Assign partial charges from PQR if provided
        if pocket is not None and pqr_path is not None:
            try:
                pqr_models = _read_pqr_models(pqr_path, include_hydrogens=include_pqr_hydrogens)
                if pqr_models:
                    _assign_partial_charges_by_coords(pocket, pqr_models[0], tolerance=charge_tolerance)
            except Exception as e:
                if not silent:
                    print(f"Warning: failed assigning PQR charges to pocket: {e}", file=sys.stderr)
        os.remove(tmp_path2)

    # os.remove(tmp_path)

    # If failed, retry by reducing `connect_cutoff`.
    if pocket is None and retry_with_cutoff:
        frame = inspect.currentframe()
        _args, _varargs, _keywords, _locals = inspect.getargvalues(frame)
        args_to_pass = {arg: _locals[arg] for arg in _args}
        args_to_pass["retry_with_cutoff"] = False

        cutoffs = [0.35 - 0.01 * i for i in range(1, MAX_NUM_RETRIALS + 1)]
        for cutoff in cutoffs:
            if not silent:
                print(
                    f"{receptor_pdb}, {ligand_sdf}:",
                    f"Retrying with cutoff {cutoff}",
                    file=sys.stderr,
                )
            args_to_pass["connect_cutoff"] = cutoff
            pocket = extract(**args_to_pass)
            if pocket:
                break

    return pocket


def _rebond_monovalent_oxygens(selection: str = "het"):
    model: chempy.models.Indexed = cmd.get_model(selection)
    monovalent_oxygens: List[List[chempy.Atom]] = []

    # Get monovalent oxygens and their bonded atoms as pairs.
    for i, atom in enumerate(model.atom):
        if atom.symbol == "O":
            bonds: List[chempy.Bond] = [bond for bond in model.bond if i in bond.index]
            if len(bonds) == 1:
                a, b = bonds[0].index
                begin = a if i == a else b
                end = b if begin == a else a
                assert begin == i
                monovalent_oxygens.append([model.atom[begin], model.atom[end]])

    # Rebond oxygen bonds.
    for oxygen, other in monovalent_oxygens:
        cmd.unbond(f"id {oxygen.id}", f"id {other.id}")
        # [O]=X
        if oxygen.formal_charge == 0:
            bond_order = 2
        # [O-]-X
        elif oxygen.formal_charge == -1:
            bond_order = 1
        cmd.bond(f"id {oxygen.id}", f"id {other.id}", bond_order)

def _neutralize_pi_oxygens(selection: str = "polymer"):
    """Neutralize '=[O-]', which somtimes appears in ASP and GLU."""
    model: chempy.models.Indexed = cmd.get_model(selection)
    # Scan "=O" oxygens.
    for i, atom in enumerate(model.atom):
        if atom.symbol == "O":
            bonds: List[chempy.Bond] = [bond for bond in model.bond if i in bond.index]
            if len(bonds) == 1 and bonds[0].order == 2:
                # If not neutral.
                if atom.formal_charge:
                    cmd.alter(f"index {atom.index}", "formal_charge=0")


def extract_ith_mol(
    sdf: Path,
    idx: int,
    tmp_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Extract i-th record from an SDF file (i >= 1)."""
    fd, tmp_path = tempfile.mkstemp(suffix=sdf.suffix, dir=tmp_dir)
    os.close(fd)
    mol = AllChem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)[idx - 1]
    # Old RDKit versions don't support context manager for `SDWriter`.
    f = AllChem.SDWriter(tmp_path)
    f.write(mol)
    f.close()
    return Path(tmp_path)


def read_mols(
    file_path: Union[str, Path],
    sanitize: bool = True,
    removeHs: bool = True,
    rebond: bool = True,
    pqr_path: Optional[Union[str, Path]] = None,
    charge_tolerance: float = 0.001,
    include_pqr_hydrogens: bool = False,
) -> List[Optional[Chem.Mol]]:
    kwargs = {"sanitize": sanitize, "removeHs": removeHs}
    path = Path(file_path)

    if path.suffix in (".sdf", ".mol"):
        mols = Chem.SDMolSupplier(str(path), **kwargs)
    elif path.suffix == ".mol2":
        mols = mols_from_mol2_file(path, **kwargs)
    elif path.suffix == ".pdb":
        mols = mols_from_pdb_file(
            path,
            rebond=rebond,
            pqr_path=pqr_path,
            charge_tolerance=charge_tolerance,
            include_pqr_hydrogens=include_pqr_hydrogens,
            **kwargs,
        )
    elif path.suffix == ".smi":
        mols = mols_from_smi_file(path, sanitize=sanitize)
    else:
        raise NotImplementedError

    return mols


def mols_from_mol2_file(
    file_path: Union[str, Path],
    sanitize: bool = True,
    removeHs: bool = True,
) -> List[Chem.Mol]:
    """Read molecules from a Mol2 file.

    A multi-mol version of `rdkit.Chem.MolFromMol2File`."""
    # For the .mol2 case, the delimiter line should be included as the beginning
    # of a block. So we read line-by-line unlike the .pdb case.
    delimiter = "@<TRIPOS>MOLECULE"
    blocks = []
    with Path(file_path).open() as f:
        for line in f:
            if line.startswith(delimiter):
                blocks.append(line)
            # Not meeting the first molecule yet.
            elif not blocks:
                continue
            else:
                blocks[-1] += line

    mols = [
        Chem.MolFromMol2Block(block, sanitize=sanitize, removeHs=removeHs)
        for block in blocks
        if block.strip()
    ]
    return mols


def mols_from_pdb_file(
    file_path: Union[str, Path],
    sanitize: bool = True,
    removeHs: bool = True,
    rebond: bool = True,
    pqr_path: Optional[Union[str, Path]] = None,
    charge_tolerance: float = 0.001,
    include_pqr_hydrogens: bool = False,
) -> List[Chem.Mol]:
    """Read molecules from a PDB file.

    A multi-model version of `rdkit.Chem.MolFromPDBFile`."""
    delimiter = "ENDMDL"
    with Path(file_path).open() as f:
        blocks = f.read().split(delimiter)

    mols = [
        Chem.MolFromPDBBlock(
            block, sanitize=sanitize, removeHs=removeHs, proximityBonding=rebond
        )
        for block in blocks
        if block.strip() and block.strip() != "END"
    ]
    # Assign partial charges from companion PQR, if provided
    if pqr_path is not None:
        try:
            pqr_models = _read_pqr_models(pqr_path, include_hydrogens=include_pqr_hydrogens)
            for i, mol in enumerate(mols):
                if mol is None:
                    continue
                pqr_atoms = pqr_models[i] if i < len(pqr_models) else (pqr_models[0] if pqr_models else [])
                if pqr_atoms:
                    _assign_partial_charges_by_coords(mol, pqr_atoms, tolerance=charge_tolerance)
        except Exception as e:
            print(f"Warning: failed assigning PQR charges: {e}", file=sys.stderr)
    return mols


def _read_pqr_models(
    pqr_path: Union[str, Path],
    include_hydrogens: bool = False,
) -> List[List[Tuple[float, float, float, float]]]:
    """Parse a PQR file into models of (x, y, z, charge).

    Tries fixed-width fields; falls back to float token extraction.
    """
    pqr_path = Path(pqr_path)
    with pqr_path.open() as f:
        text = f.read()

    blocks = text.split("ENDMDL") if "ENDMDL" in text else [text]

    def is_hydrogen_name(name: str) -> bool:
        n = name.strip().upper()
        return n.startswith("H") or n.startswith("D")

    models: List[List[Tuple[float, float, float, float]]] = []
    for block in blocks:
        atoms: List[Tuple[float, float, float, float]] = []
        for line in block.splitlines():
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            # Try fixed-width parse
            parsed = False
            try:
                if len(line) >= 62:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    charge = float(line[54:62].replace("D", "E"))
                    atom_name = line[12:16]
                    if (include_hydrogens or (not is_hydrogen_name(atom_name))):
                        atoms.append((x, y, z, charge))
                        parsed = True
            except Exception:
                parsed = False
            if parsed:
                continue
            # Fallback: float token parse
            parts = line.split()
            atom_name = parts[2] if len(parts) > 2 else ""
            if (not include_hydrogens) and is_hydrogen_name(atom_name):
                continue
            floats: List[float] = []
            for tok in parts:
                try:
                    floats.append(float(tok.replace("D", "E")))
                except ValueError:
                    continue
            if len(floats) >= 4:
                x, y, z, charge = floats[0], floats[1], floats[2], floats[3]
                atoms.append((x, y, z, charge))
        if atoms:
            models.append(atoms)
    return models


def _assign_partial_charges_by_coords(
    mol: Chem.Mol,
    pqr_atoms: List[Tuple[float, float, float, float]],
    tolerance: float = 0.1,
) -> int:
    """Assign PartialCharge to RDKit mol atoms by nearest coordinate match.

    Returns number of atoms assigned.
    """
    if mol is None or mol.GetNumAtoms() == 0 or not pqr_atoms:
        return 0
    conf = mol.GetConformer()
    used = [False] * len(pqr_atoms)
    assigned = 0
    for atom_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        best_j = -1
        best_d2 = float("inf")
        for j, (x, y, z, charge) in enumerate(pqr_atoms):
            if used[j]:
                continue
            dx = pos.x - x
            dy = pos.y - y
            dz = pos.z - z
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best_j = j
        if best_j >= 0 and best_d2 <= tolerance * tolerance:
            charge_val = float(pqr_atoms[best_j][3])
            mol.GetAtomWithIdx(atom_idx).SetDoubleProp("PartialCharge", charge_val)
            used[best_j] = True
            assigned += 1
    return assigned


def mols_from_smi_file(
    file_path: Union[str, Path],
    sanitize: bool = True,
) -> List[Chem.Mol]:
    """Read molecules from a SMILES file.

    Substitute for `rdkit.Chem.SmilesMolSupplier`."""
    mols = []
    with Path(file_path).open() as f:
        for line in f:
            try:
                # `maxsplit=1` allows names with whitespaces.
                smiles, name = line.split(maxsplit=1)
                name = name.strip()
            except ValueError:
                smiles = line.strip()
                name = None

            mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
            if name is not None:
                mol.SetProp("_Name", name)
            mols.append(mol)

    return mols


def extract_binding_pocket(
    ligand_mol: Chem.Mol,
    pdb_path: PathLike,
    distance: float = 5.0,
    pqr_path: Optional[Union[str, Path]] = None,
    charge_tolerance: float = 0.001,
    include_pqr_hydrogens: bool = False,
) -> Optional[Chem.Mol]:
    with tempfile.NamedTemporaryFile(suffix=".sdf") as ligand_file:
        writer = Chem.SDWriter(ligand_file.name)
        try:
            writer.write(ligand_mol)
        finally:
            writer.close()
        return extract(
            pdb_path,
            ligand_file.name,
            distance=distance,
            pqr_path=pqr_path,
            charge_tolerance=charge_tolerance,
            include_pqr_hydrogens=include_pqr_hydrogens,
        )


def main(args: argparse.Namespace):
    # Protonate the PDB file if required.
    if args.no_prot_pdb:
        pdb_file: Path = args.pdb_file
    else:
        pdb_file = protonate_pdb(args.pdb_file)

    if len(list(read_mols(args.ligand_file))) > 1:
        for mol_idx, mol_ligand in enumerate(read_mols(args.ligand_file)):
            # mol_ligand = read_mols(args.ligand_file)[0]
            if not args.no_prot_sdf:
                mol_ligand = protonate_ligand(mol_ligand)
                if not mol_ligand:
                    print("protonate_ligand failed:", f"{args.ligand_file.stem}_{mol_idx}", file=sys.stderr)
                    exit()

            mol_target = extract_binding_pocket(
                mol_ligand,
                pdb_file,
                pqr_path=getattr(args, "pqr_file", None),
                charge_tolerance=getattr(args, "pqr_tolerance", 0.001),
            )
            args.save_file_path.mkdir(parents=True, exist_ok=True)

            filename = (
                f"{args.prefix}_{args.ligand_file.stem}_{mol_idx}"
                if args.prefix
                else f"{args.ligand_file.stem}_{mol_idx}"
            )
            with open(
                args.save_file_path / filename, "wb"
            ) as f:
                pickle.dump((mol_ligand, mol_target), f)
    else:
        mol_ligand = read_mols(args.ligand_file)[0]
        if not args.no_prot_sdf:
            mol_ligand = protonate_ligand(mol_ligand)
            if not mol_ligand:
                print("protonate_ligand failed:", args.ligand_file.stem, file=sys.stderr)
                exit()

        mol_target = extract_binding_pocket(
            mol_ligand,
            pdb_file,
            pqr_path=getattr(args, "pqr_file", None),
            charge_tolerance=getattr(args, "pqr_tolerance", 0.001),
        )
        args.save_file_path.mkdir(parents=True, exist_ok=True)

        filename = (
            f"{args.prefix}_{args.ligand_file.stem}"
            if args.prefix
            else args.ligand_file.stem
        )
        with open(args.save_file_path / filename, "wb") as f:
            pickle.dump((mol_ligand, mol_target), f)

    if not args.no_prot_pdb:
        # pdb_file = protonate_pdb(args.pdb_file)
        pdb_file.unlink()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-p", "--pdb_file", type=Path, help="protein .pdb file")
    parser.add_argument(
        "-l",
        "--ligand_file",
        type=Path,
        help="ligand files (.pdb | .mol2 | .sdf)",
    )
    parser.add_argument(
        "--no-prot-pdb", action="store_true", help="don't protonate the input PDB"
    )
    parser.add_argument(
        "--no-prot-sdf", action="store_true", help="don't protonate the input SDF"
    )
    parser.add_argument(
        "-s",
        "--save_file_path",
        type=Path,
        help="save files (.pt) directory",
        default="./data",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="filename prefix",
        default="",
    )
    parser.add_argument(
        "--pqr_file",
        type=Path,
        help="optional companion protein .pqr file for partial charges",
        default=None,
    )
    parser.add_argument(
        "--pqr_tolerance",
        type=float,
        help="distance tolerance (Å) for PQR coordinate matching",
        default=0.001,
    )
    args = parser.parse_args()

    main(args)
