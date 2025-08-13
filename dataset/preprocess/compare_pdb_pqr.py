#!/usr/bin/env python3
"""
compare_pdb_pqr.py

Compare PDB vs PQR after stripping hydrogens/deuteriums using Biopython.

- Strips H/D logically via filters.
- Compares by key (chain,resName,resSeq,atomName) with tolerance on per-axis deltas.
- Handles altLocs, waters, residue and chain filters, HETATM ignoring, and optional serial checks.
- Line-level outputs and exit code:
    0 = perfect match under chosen filters and tolerance
    1 = differences found or parsing/conversion error
"""

import argparse
import sys
import re
from typing import Dict, Tuple, Set, Optional

import numpy as np
from Bio.PDB import PDBParser

# -------- CLI --------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Compare PDB vs PQR after stripping hydrogens/deuteriums using Biopython."
    )
    p.add_argument("pdb_in")
    p.add_argument("pqr_in")
    p.add_argument("-t", dest="tol", type=float, default=0.001, help="Tolerance in Å (default 0.001)")
    p.add_argument("--altloc", default="A", help="Keep altLoc X plus blank; 'none'=only blank; 'any'=keep all")
    p.add_argument("--ignore-chain", action="store_true", help="Ignore chain in the identity key")
    p.add_argument("--check-serial", action="store_true", help="Check serial numbers; report SERIAL_MISMATCH")
    p.add_argument("--no-waters", action="store_true", help="Exclude waters (HOH,WAT,TIP3,SOL)")
    p.add_argument("--exclude-resn", default="", help="CSV of residue names to exclude")
    p.add_argument("--only-chains", default="", help="CSV of chain IDs to keep")
    p.add_argument("--ignore-hetatm", action="store_true", help="Ignore HETATM records entirely")
    p.add_argument("--debug", action="store_true", help="Print counts and sample lines")
    return p.parse_args()

# -------- Core helpers --------

WATERS = {"HOH", "WAT", "TIP3", "SOL"}

def keep_altloc(atom_altloc: str, mode: str) -> bool:
    """Implement altloc selection: 'any', 'none' for blank only, else blank or letter X."""
    if mode.lower() == "any":
        return True
    if mode.lower() == "none":
        return atom_altloc in ("", " ", None)
    return atom_altloc in ("", " ", mode)

def is_hydrogen(atom_name: str, element: Optional[str]) -> bool:
    """Conservative H/D test. Prefer element if present, else name prefix."""
    if element:
        e = element.strip().upper()
        if e in ("H", "D"):
            return True
    return atom_name.strip().upper().startswith(("H", "D"))

def resi_string(residue) -> str:
    """
    Build residue index string as in PDB fixed columns: resseq + optional insertion code.
    Biopython residue.id is (hetflag, resseq, icode).
    """
    hetflag, resseq, icode = residue.id
    icode = icode.strip() if isinstance(icode, str) else ""
    return f"{resseq}{icode}"

def identity_key(chain_id: str, resn: str, resi: str, atom_name: str, ignore_chain: bool) -> str:
    chain = "" if ignore_chain else chain_id
    return f"{chain}:{resn}:{resi}:{atom_name}"

def want_chain(chain_id: str, only: Set[str]) -> bool:
    return True if not only else chain_id in only

def should_filter(resn: str, chain_id: str, recname: str,
                  no_waters: bool, exclude_resn: Set[str], only_chains: Set[str],
                  ignore_hetatm: bool) -> bool:
    if ignore_hetatm and recname == "HETATM":
        return True
    if no_waters and resn in WATERS:
        return True
    if exclude_resn and resn in exclude_resn:
        return True
    if not want_chain(chain_id, only_chains):
        return True
    return False

# -------- Parsing with Biopython (PDB) --------

def parse_pdb_with_biopython(
    pdb_path: str,
    altloc_mode: str,
    ignore_chain: bool,
    no_waters: bool,
    exclude_resn: Set[str],
    only_chains: Set[str],
    ignore_hetatm: bool,
    debug: bool = False,
):
    """
    Return:
      coords: dict[key] -> np.array([x,y,z], dtype=float)
      serials: dict[key] -> serial as str
      parsed_any: bool
    """
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure("pdb", pdb_path)

    coords: Dict[str, np.ndarray] = {}
    serials: Dict[str, str] = {}
    parsed_any = False
    head_lines = []

    # Use only first model by default to match your original behaviour
    model = next(structure.get_models())

    for chain in model:
        chain_id = chain.id.strip() if chain.id is not None else ""
        for residue in chain:
            # Skip water etc
            resn = residue.resname.strip() if hasattr(residue, "resname") else ""
            if should_filter(resn, chain_id, "ATOM", no_waters, exclude_resn, only_chains, ignore_hetatm=False):
                continue

            for atom in residue:
                # HETATM vs ATOM
                recname = "HETATM" if residue.id[0].strip() != "" else "ATOM"
                if should_filter(resn, chain_id, recname, no_waters, exclude_resn, only_chains, ignore_hetatm):
                    continue

                alt = atom.get_altloc()
                alt = "" if alt is None else alt
                if not keep_altloc(alt, altloc_mode):
                    continue

                # Hydrogens
                if is_hydrogen(atom.get_name(), atom.element if hasattr(atom, "element") else None):
                    continue

                resi = resi_string(residue)
                key = identity_key(chain_id, resn, resi, atom.get_name().strip(), ignore_chain)

                if key not in coords:
                    coords[key] = atom.get_coord().astype(float)
                    # Biopython exposes serial via atom.get_serial_number() in recent versions
                    try:
                        serials[key] = str(atom.get_serial_number())
                    except Exception:
                        serials[key] = ""
                    parsed_any = True
                    if debug and len(head_lines) < 5:
                        head_lines.append(f"PDB {recname} key={key} coord={coords[key].tolist()} serial={serials[key]}")

    if debug:
        print(f"[PDB] atom count: {len(coords)}")
        print("[PDB] head:")
        for ln in head_lines:
            print(ln)
        print()
    return coords, serials, parsed_any

# -------- Minimal PQR reader (whitespace-tolerant) --------

_int = re.compile(r"^-?\d+$")

def parse_pqr_minimal(
    pqr_path: str,
    altloc_mode: str,
    ignore_chain: bool,
    no_waters: bool,
    exclude_resn: Set[str],
    only_chains: Set[str],
    ignore_hetatm: bool,
    debug: bool = False,
):
    """
    Parse ATOM/HETATM lines from PQR. Treat like PDB without occupancy/tempfactor,
    allow records with or without chain column.

    Returns coords, serials, parsed_any (same contract as PDB parser).
    """
    coords: Dict[str, np.ndarray] = {}
    serials: Dict[str, str] = {}
    parsed_any = False
    head_lines = []

    with open(pqr_path, "r", errors="replace") as fh:
        nline = 0
        for line in fh:
            nline += 1
            parts = line.split()
            if not parts:
                continue
            rec = parts[0]
            if rec not in ("ATOM", "HETATM"):
                continue
            if ignore_hetatm and rec == "HETATM":
                continue

            # Columns in PQR vary. Try two common layouts:
            # 1) ATOM serial name resn chain resi x y z charge radius
            # 2) ATOM serial name resn resi x y z charge radius
            try:
                serial = parts[1]
                atom = parts[2]
                resn = parts[3]
                if len(parts) >= 10 and _int.fullmatch(parts[5]):
                    chain = parts[4]
                    resi = parts[5]
                    xi, yi, zi = 6, 7, 8
                elif len(parts) >= 9 and _int.fullmatch(parts[4]):
                    chain = ""
                    resi = parts[4]
                    xi, yi, zi = 5, 6, 7
                else:
                    # Very odd formatting, skip
                    continue

                # altLoc not present in PQR; treat as blank
                if not keep_altloc("", altloc_mode):
                    continue

                if no_waters and resn in WATERS:
                    continue
                if exclude_resn and resn in exclude_resn:
                    continue
                if not want_chain(chain, only_chains):
                    continue

                # Hydrogens by atom name
                if is_hydrogen(atom, None):
                    continue

                x = float(parts[xi]); y = float(parts[yi]); z = float(parts[zi])
                key = identity_key(chain, resn, str(int(resi)), atom, ignore_chain)

                if key not in coords:
                    coords[key] = np.array([x, y, z], dtype=float)
                    serials[key] = serial if _int.fullmatch(serial) else ""
                    parsed_any = True
                    if debug and len(head_lines) < 5:
                        head_lines.append(f"PQR {rec} key={key} coord={coords[key].tolist()} serial={serials[key]}")

            except Exception:
                # Skip malformed lines silently to mirror robust AWK behaviour
                continue

    if debug:
        print(f"[PQR] atom count: {len(coords)}")
        print("[PQR] head:")
        for ln in head_lines:
            print(ln)
        print()
    return coords, serials, parsed_any

# -------- Comparison --------

def compare_and_report(
    pdb_coords: Dict[str, np.ndarray],
    pdb_serials: Dict[str, str],
    pqr_coords: Dict[str, np.ndarray],
    pqr_serials: Dict[str, str],
    tol: float,
    check_serial: bool,
) -> int:
    mismatches = 0
    serial_mismatches = 0
    missing_in_2 = 0
    extra_in_2 = 0
    common = 0

    # Compare common keys
    for k, v1 in pdb_coords.items():
        if k in pqr_coords:
            common += 1
            v2 = pqr_coords[k]
            delta = v1 - v2
            adelta = np.abs(delta)
            if np.any(adelta > tol):
                print(
                    "MISMATCH\t{}\tPDB({:.6f},{:.6f},{:.6f})\tPQR->PDB({:.6f},{:.6f},{:.6f})\t|Δ|=({:.6f},{:.6f},{:.6f})"
                    .format(k, v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], adelta[0], adelta[1], adelta[2])
                )
                mismatches += 1
            if check_serial and pdb_serials.get(k, "") != pqr_serials.get(k, ""):
                print(f"SERIAL_MISMATCH\t{k}\tPDB#{pdb_serials.get(k,'')}\tPQR#{pqr_serials.get(k,'')}")
                serial_mismatches += 1
        else:
            print(f"MISSING_IN_PQR\t{k}")
            missing_in_2 += 1

    # Extras in PQR
    for k in pqr_coords.keys():
        if k not in pdb_coords:
            print(f"EXTRA_IN_PQR\t{k}")
            extra_in_2 += 1

    print(
        f"\nSummary: stripped H/D | tol={tol:g} Å | common={common} | "
        f"mismatches={mismatches} | serial_mismatches={serial_mismatches} | "
        f"missing_in_pqr={missing_in_2} | extra_in_pqr={extra_in_2}"
    )
    return 1 if (mismatches + serial_mismatches + missing_in_2 + extra_in_2) > 0 else 0

# -------- Main --------

def main():
    args = parse_args()

    exclude_resn = {s.strip() for s in args.exclude_resn.split(",") if s.strip()}
    only_chains = {s.strip() for s in args.only_chains.split(",") if s.strip()}

    # PDB via Biopython
    try:
        pdb_coords, pdb_serials, parsed1 = parse_pdb_with_biopython(
            args.pdb_in, args.altloc, args.ignore_chain,
            args.no_waters, exclude_resn, only_chains, args.ignore_hetatm, debug=args.debug
        )
    except Exception as e:
        print(f"ERROR: Failed to parse PDB input: {e}", file=sys.stderr)
        sys.exit(1)

    if not parsed1:
        print("ERROR: No atoms parsed from PDB input after stripping and filtering", file=sys.stderr)
        sys.exit(1)

    # PQR via minimal reader
    try:
        pqr_coords, pqr_serials, parsed2 = parse_pqr_minimal(
            args.pqr_in, args.altloc, args.ignore_chain,
            args.no_waters, exclude_resn, only_chains, args.ignore_hetatm, debug=args.debug
        )
    except Exception as e:
        print(f"ERROR: Failed to parse PQR input: {e}", file=sys.stderr)
        sys.exit(1)

    if not parsed2:
        print("ERROR: No atoms parsed from PQR input after stripping and filtering", file=sys.stderr)
        sys.exit(1)

    rc = compare_and_report(
        pdb_coords, pdb_serials, pqr_coords, pqr_serials, args.tol, args.check_serial
    )
    sys.exit(rc)

if __name__ == "__main__":
    main()