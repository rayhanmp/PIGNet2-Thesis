# Negative Data Augmentation Code

All codes in this directory works with a **single** PDB or file.
You need to install Smina, OpenBabel, and DockRMSD.

Note that you should add PATH for `DockRMSD`, to execute them directly.

Here, we assume that the PDBbind data are downloaded at `$PDBBIND_DIR` directory.
So, ligand and protein files for certain `$PDB` are `$PDBBIND_DIR/$PDB/${PDB}_ligand.mol2` and `$PDBBIND_DIR/$PDB/${PDB}_protein.pdb`, respectively.

First, define the location of your files by:

```console
export PDBBIND_DIR=/path/to/pdbbind
```

Use either absolute path or relative path from where you execute the script from.

You can then run the script to generate the NDA by running:

```console
./main.sh
```
