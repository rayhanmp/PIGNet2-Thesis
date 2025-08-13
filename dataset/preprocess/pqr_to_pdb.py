#!/usr/bin/env python3
"""
pqr_to_pdb.py

Lightweight utility to convert a PQR (PDB-like) file to PDB while
setting atom occupancy and B-factor (temperature factor) to fixed values.

Done using Biopython's PDBParser which is permissive enough to read many PQR files,
treating them as PDB with extra columns that it ignores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from Bio.PDB import PDBParser, PDBIO


def convert_pqr_to_pdb(
    input_pqr: str | Path,
    output_pdb: str | Path,
    occupancy: float = 1.0,
    bfactor: float = 0.0,
) -> Tuple[int, int]:
    """
    Convert a PQR file to PDB and set occupancy/B-factor for all atoms.

    Parameters
    ----------
    input_pqr : str | Path
        Path to the input PQR file. Although the extension is ".pqr",
        the parser treats it as a PDB-like file and will ignore PQR-only
        columns (charge, radius) rather than failing.
    output_pdb : str | Path
        Path for the output PDB file to be written.
    occupancy : float, default 1.0
        Value to assign to the occupancy field for every atom.
    bfactor : float, default 0.0
        Value to assign to the B-factor (a.k.a. temperature factor) for every atom.

    Returns
    -------
    Tuple[int, int]
        A pair (n_models, n_atoms) describing how many models and atoms were written.

    Notes
    -----
    - This function does not attempt to preserve PQR charge/radius columns because
      standard PDB format has no canonical columns for them.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If no atoms are found after parsing (likely a malformed file).
    """
    input_pqr = Path(input_pqr)
    output_pdb = Path(output_pdb)

    if not input_pqr.exists():
        raise FileNotFoundError(f"Input file not found: {input_pqr}")

    # Parse as a PDB-like structure. PQR has similar record layout;
    # extra columns (charge, radius) are ignored by the parser.
    parser = PDBParser(PERMISSIVE=True)
    structure = parser.get_structure("structure", str(input_pqr))

    # Walk all atoms across all models/chains/residues and set fields.
    atom_count = 0
    for atom in structure.get_atoms():
        atom.set_occupancy(float(occupancy))
        atom.set_bfactor(float(bfactor))
        atom_count += 1

    if atom_count == 0:
        raise ValueError("Parsed structure contains zero atoms; is the input valid?")

    # Write out a clean PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb))

    # Count models for convenience in return value.
    n_models = sum(1 for _ in structure.get_models())
    return n_models, atom_count


# Optional CLI wrapper, so this module can be used as a script.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PQR to PDB and set occupancy/B-factor."
    )
    parser.add_argument("input_pqr", help="Path to input .pqr file")
    parser.add_argument("output_pdb", help="Path to output .pdb file")
    parser.add_argument(
        "--occupancy",
        type=float,
        default=1.0,
        help="Occupancy value to assign (default: 1.0)",
    )
    parser.add_argument(
        "--bfactor",
        type=float,
        default=0.0,
        help="B-factor (temperature factor) to assign (default: 0.0)",
    )

    args = parser.parse_args()
    models, atoms = convert_pqr_to_pdb(
        args.input_pqr, args.output_pdb, args.occupancy, args.bfactor
    )
    print(f"Wrote {args.output_pdb} with {atoms} atoms across {models} model(s).")
