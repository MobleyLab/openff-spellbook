#!/usr/bin/env python3

from ..qcasb import QCArchiveSpellBook

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The OpenForceField Spellbook error scanner for QCArchive")

    parser.add_argument('--save-xyz', action="store_true")
    parser.add_argument('--report-out', type=str)
    parser.add_argument('--full-report', action="store_true")
    args = parser.parse_args()

    qcasb = QCArchiveSpellBook()
    qcasb.error_report_per_dataset(
            save_xyz=args.save_xyz,
            out_fnm=args.report_out,
            full_report=args.full_report)

