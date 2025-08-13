from pathlib import Path
from generate_data import _read_pqr_models

pqr_file = Path("1a9m.pqr")

models = _read_pqr_models(pqr_file, include_hydrogens=True)

print(f"Parsed {len(models)} model(s)")
for i, atoms in enumerate(models, start=1):
    print(f"Model {i}: {len(atoms)} atoms parsed")
    # peek at first few
    for (x, y, z, q) in atoms[:10]:
        print(f"  x={x:.3f} y={y:.3f} z={z:.3f} charge={q:.10f}")
