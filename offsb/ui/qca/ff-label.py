#!/usr/bin/env python3

import offsb.ui.qcasb
import pprint


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The OpenForceField Spellbook parameter labeler for QCArchive")

    mode_choices = ['bonds', 'angles', 'dihedrals', 'outofplanes']
    parser.add_argument('openff_name', type=str)
    parser.add_argument('--out_file_name', type=str)
    parser.add_argument('--datasets', type=str)
    parser.add_argument('--cache', type=str)

    args = parser.parse_args()

    datasets = None
    if args.datasets is not None:
        datasets = offsb.ui.qcasb.load_dataset_input(args.datasets)

    obj = offsb.ui.qcasb.QCArchiveSpellBook(datasets=datasets, QCA=args.cache)


    out = args.out_file_name
    if args.out_file_name is None:
        out = args.openff_name

    labeler = obj.assign_labels_from_openff(args.openff_name, out)
    
    pprint.pprint(labeler.db)
