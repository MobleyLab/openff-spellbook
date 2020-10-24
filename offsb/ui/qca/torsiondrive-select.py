import offsb.ui.qcasb

# def torsiondrive_select(
#     self,
#     smarts="[*:1]~[*:2]~[*:3]~[*:3]",
#     exact=True,
#     wbo=False,
#     mbo=False,
#     ffname=None,
#     bond_param=None,
#     torsion_param=None,
#     energy="None",
#     mm_minimize=False,
#     out_fname=None,
#     collapse_atom_tuples=True,
# ):

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="The OpenForceField Spellbook TorsionDrive parser"
    )

    parser.add_argument("--out_file_name", type=str)
    parser.add_argument("--datasets", type=str)

    parser.add_argument("--qm-energy", action="store_true")
    parser.add_argument("--wbo", action="store_true")
    parser.add_argument("--mbo", action="store_true")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--smarts", type=str, default="[*:1]~[*:2]~[*:3]~[*:4]")

    mm_choices = ["None", "all", "vdw", "bonds", "angles", "dihedrals", "outofplanes"]

    parser.add_argument("--mm-energy", choices=mm_choices, default="all")
    parser.add_argument("--mm-minimize", action="store_true")
    parser.add_argument("--mm-constrain", action="store_true")
    parser.add_argument("--mm-geometric", action="store_true")

    parser.add_argument("--openff-name", type=str)
    parser.add_argument("--bond-parameter", type=str)
    parser.add_argument("--torsion-parameter", type=str)

    args = parser.parse_args()

    datasets = None
    if args.datasets is not None:
        datasets = offsb.ui.qcasb.load_dataset_input(args.datasets)

    obj = offsb.ui.qcasb.QCArchiveSpellBook(datasets=datasets)

    obj.torsiondrive_select(
        smarts=args.smarts,
        exact=args.exact,
        wbo=args.wbo,
        mbo=args.mbo,
        energy=args.mm_energy,
        bond_param=args.bond_parameter,
        torsion_param=args.torsion_parameter,
        ffname=args.openff_name,
        mm_minimize=args.mm_minimize,
        mm_geometric=args.mm_geometric,
        mm_constrain=args.mm_constrain,
        out_fname=args.out_file_name,
    )

    # if out is not None:
    # obj.plot_torsiondrive_groupby_openff_param(out, oldparamfile=args.openff_previous)
