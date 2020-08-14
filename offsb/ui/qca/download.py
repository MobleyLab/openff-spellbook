
#!/usr/bin/env python3

import offsb.ui.qcasb

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The OpenForceField Spellbook QCArchive dataset downloader")

    parser.add_argument('--datasets', type=str)

    args = parser.parse_args()

    if args.datasets is not None:
        datasets = offsb.ui.qcasb.load_dataset_input(args.datasets)

    offsb.ui.qcasb.QCArchiveSpellBook(datasets=datasets)
