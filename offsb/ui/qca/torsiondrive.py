#!/usr/bin/env python3

import offsb.ui.qcasb


def select_mode_func(obj, mode):
    if mode == "torsiondrive_groupby_openff":
        return obj.torsiondrive_groupby_openff_param


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="The OpenForceField Spellbook TorsionDrive parser"
    )

    mode_choices = ["torsiondrive_groupby_openff"]
    parser.add_argument("mode", choices=mode_choices, type=str)
    parser.add_argument("--out_file_name", type=str)
    parser.add_argument("--datasets", type=str)

    parser.add_argument("--qm-energy", action="store_true")

    mm_choices = ["None", "all", "vdw", "bonds", "angles", "dihedrals", "outofplanes"]

    parser.add_argument("--mm-energy", choices=mm_choices, default="all")

    parser.add_argument("--openff-name", type=str)
    parser.add_argument("--openff-parameter", type=str)
    parser.add_argument("--openff-previous", type=str)

    args = parser.parse_args()

    datasets = None
    if args.datasets is not None:
        datasets = offsb.ui.qcasb.load_dataset_input(args.datasets)

    obj = offsb.ui.qcasb.QCArchiveSpellBook(datasets=datasets)

    if args.openff_name is not None and args.openff_parameter is not None:
        name = args.openff_name
        param = args.openff_parameter
        ene = args.mm_energy
        out = args.out_file_name

        obj.torsiondrive_groupby_openff_param(name, param, energy=ene,
                out_fname=out)
        if out is not None:
            obj.plot_torsiondrive_groupby_openff_param(out, oldparamfile=args.openff_previous)
    else:
        raise Exception("Not implemented yet")
        obj.torsiondrive_print(energy=args.qm_energy, out=args.out_file_name)
