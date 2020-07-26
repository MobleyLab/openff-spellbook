#!/usr/bin/env python3

import sys
from ..qcasb import QCArchiveSpellBook

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The OpenForceField Spellbook energy extractor from QCArchive")

    parser.add_argument('--report-out', type=str)
    # parser.add_argument('--full-report', action="store_true")
    args = parser.parse_args()

    qcasb = QCArchiveSpellBook()
    enes_per_mol= qcasb.energy_minimum_per_molecule()

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
