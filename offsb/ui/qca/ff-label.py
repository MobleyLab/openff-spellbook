#!/usr/bin/env python3

import offsb.ui.qcasb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The OpenForceField Spellbook parameter labeler for QCArchive")

    mode_choices = ['bonds', 'angles', 'dihedrals', 'outofplanes']
    parser.add_argument('openff_name', type=str)
    parser.add_argument('--out_file_name', type=str)
    parser.add_argument('--datasets', type=str)

    args = vars(parser.parse_args())

    datasets = None
    if args.datasets is not None:
        datasets = offsb.ui.qcasb.load_dataset_input(args.datasets)

    obj = offsb.ui.qcasb.QCArchiveSpellBook(datasets=datasets)


    if args.out_file_name is None:
        out = args.openff_name

    obj.assign_labels_from_openff(args.openff_name, out)

