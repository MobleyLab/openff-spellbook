#!/usr/bin/env python3

import offsb.ui.qcasb
import offsb.qcarchive.qcatree as qca

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="The OpenForceField Spellbook QCArchive dataset downloader"
    )

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--filter-from-file", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache-initial", action="store_true")
    parser.add_argument("--cache-final", action="store_true")
    parser.add_argument("--cache-molecules", action="store_true")
    parser.add_argument("--cache-gradients", action="store_true")
    parser.add_argument("--cache-minimum-molecules", action="store_true")
    parser.add_argument("--cache-minimum-gradients", action="store_true")

    args = parser.parse_args()

    if args.datasets is not None:
        datasets = offsb.ui.qcasb.load_dataset_input(args.datasets)

    drop = None
    if args.filter_from_file is not None:
        drop = qca.QCAFilter.from_file(args.filter_from_file)

    obj = offsb.ui.qcasb.QCArchiveSpellBook(
        datasets=datasets, start=args.start, limit=args.limit, drop=drop
    )

    cached = False

    if args.cache_initial:
        obj.QCA.cache_initial_molecules()
        cached = True

    if args.cache_final:
        obj.QCA.cache_final_molecules()
        cached = True

    if args.cache_molecules:
        cached += obj.QCA.cache_optimization_minimum_molecules()
        cached = True

    if args.cache_gradients:
        cached += obj.QCA.cache_optimization_minimum_gradients()
        cached = True

    if args.cache_minimum_molecules:
        cached += obj.QCA.cache_torsiondriverecord_minimum_molecules()
        cached = True

    if args.cache_minimum_gradients:
        cached += obj.QCA.cache_torsiondriverecord_minimum_gradients()
        cached = True

    if cached:
        obj.save()


if __name__ == "__main__":
    main()
