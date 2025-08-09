import sys, pathlib
def centroid(pdbqt):
    xs= []; ys= []; zs= []
    with open(pdbqt) as f:
        for line in f:
            if line.startswith(('ATOM','HETATM')):
                xs.append(float(line[30:38])); ys.append(float(line[38:46])); zs.append(float(line[46:54]))
    cx=sum(xs)/len(xs); cy=sum(ys)/len(ys); cz=sum(zs)/len(zs)
    print(f"{cx:.3f} {cy:.3f} {cz:.3f}")

def split_models(src, outdir):
    src = pathlib.Path(src); outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    parts=[]; chunk=[]
    for line in src.read_text().splitlines(keepends=True):
        if line.startswith("MODEL "):
            if chunk: parts.append("".join(chunk))
            chunk=[line]
        else:
            chunk.append(line)
    if chunk: parts.append("".join(chunk))
    for i, part in enumerate(parts, 1):
        (outdir / f"pose_{i:03d}.pdbqt").write_text(part)
    print(len(parts))

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "centroid": centroid(sys.argv[2])
    elif cmd == "split-models": split_models(sys.argv[2], sys.argv[3])
    else: sys.exit("unknown subcommand")