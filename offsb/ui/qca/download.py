#!/usr/bin/env python3

import offsb.ui.qcasb


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="The OpenForceField Spellbook QCArchive dataset downloader"
    )

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--filter-from-file", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()

    if args.datasets is not None:
        datasets = offsb.ui.qcasb.load_dataset_input(args.datasets)

    drop = []
    if args.filter_from_file is not None:
        with open(args.filter_from_file) as fid:
            for line in fid:
                line = line.rstrip('\n')
                if line.lstrip().startswith('#'):
                    continue
                line = line.split('#')[0]
                drop.append(line)

    offsb.ui.qcasb.QCArchiveSpellBook(
        datasets=datasets, start=args.start, limit=args.limit, drop=drop
    )


if __name__ == "__main__":
    main()
