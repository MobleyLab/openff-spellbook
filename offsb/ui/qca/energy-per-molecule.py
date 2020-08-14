#!/usr/bin/env python3

import sys
import offsb.ui.qcasb

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The OpenForceField Spellbook energy extractor from QCArchive")

    parser.add_argument('--report-out', type=str)
    parser.add_argument('--datasets', type=str)
    # parser.add_argument('--full-report', action="store_true")
    args = parser.parse_args()

    if args.datasets is not None:
        datasets = offsb.ui.qcasb.load_dataset_input(args.datasets)

    obj = offsb.ui.qcasb.QCArchiveSpellBook(datasets=datasets)
    enes_per_mol= obj.energy_minimum_per_molecule()

    if args.report_out == "":
        fid = sys.stdout
    else:
        fid = open(args.report_out, 'w')

    for cmiles, enes in enes_per_mol.items():
        fid.write("# {:s}\n".format(cmiles))
        for ene in enes:
            fid.write("    {:12.8f}\n".format(ene))

    if args.report_out == "":
        fid.close()
