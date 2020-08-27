#!/usr/bin/env python3

import offsb.ui.qcasb

def select_mode_func(obj, mode):
    return getattr(obj, "measure_"+mode)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The OpenForceField Spellbook measuring tool for QCA")

    mode_choices = ['bonds', 'angles', 'dihedrals', 'outofplanes']
    parser.add_argument('mode', choices=mode_choices, type=str)
    parser.add_argument('--out_file_name', type=str)
    parser.add_argument('--datasets', type=str)

    args = parser.parse_args()

    datasets = None
    if args.datasets is not None:
        datasets = offsb.ui.qcasb.load_dataset_input(args.datasets)

    obj = offsb.ui.qcasb.QCArchiveSpellBook(datasets=datasets)

    mode = args.mode

    if args.out_file_name is None:
        out = mode

    measure = select_mode_func(obj, mode)
    measure(out)
