#!/usr/bin/env python3

import contextlib
import io
import json
import logging
import sys
from multiprocessing import Pool

import msgpack
import numpy as np
import tqdm

import lzma
import bz2

import openforcefield
from openforcefield.topology.molecule import Molecule
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.rdMolAlign import AlignMol

logger = logging.getLogger(__name__)
# TODO: convert output to use the logger


def detect_file_total_data_lines(filename, skip_rows=0):
    N = -skip_rows
    for line in open(filename, "r"):
        N += 1
    return N


def parse_file(filename, N, line_start=0, line_end=None, skip_rows=0):
    """
    start and end is inclusive
    """

    if line_end is None:
        line_end = N

    smi = []

    with open(filename, "r") as fd:
        for lineno, line in enumerate(fd, -skip_rows):
            if lineno < line_start:
                continue
            if lineno > line_end:
                continue
            spline = line.split()
            smi.append(spline[0])

    return smi


def process_smiles_to_qcschema(lineno, N, header, filename, **kwargs):

    out_lines = ""
    out_mols = {}
    smi_list = parse_file(
        filename, N, line_start=lineno, line_end=lineno, skip_rows=header
    )
    out_lines += "{:8d} / {:8d} ENTRY: {:s}\n".format(lineno + 1, N, smi_list[0])
    mols = expand_smiles_to_qcschema(smi_list[0], **kwargs)

    isomers = 0
    conformations = 0
    entries = 1
    for i, (smi, mol_list) in enumerate(mols.items(), 1):
        for mol in mol_list:
            isomers += 1
            conformations += len(mol.conformers)
            out_lines += "{:22s}ISOMER {:3d}/{:3d} CONFS: {} SMILES: {:s}\n".format(
                "", i, len(mols), len(mol.conformers), smi
            )

    out_mols.update(mols)
    # out_lines += out

    return out_mols, out_lines, (entries, isomers, conformations)


def expand_smiles_to_qcschema(
    smi,
    cutoff=None,
    n_confs=1,
    unique_smiles=True,
    isomer_max=-1,
):
    """
    Load a file containing smiles strings, and generate stereoisomers and
    conformers for each stereoisomer.

    Parameters
    ----------
    input_fnm : str
        The input filename to read SMILES from
    cutoff : float
        During the all-pairwise RMSD calculation, remove
        molecules that are less than this cutoff value apart
    n_confs : int
        The number of conformations to attempt generating
    unique_smiles : bool
        If stereoisomers are generated, organize molecules by
        their unambiguous SMILES string
    isomers : int
        The number of stereoisomers to keep if multiple are found.
        The default of -1 means keep all found.
    line_start : int
        The line in the input file to start processing
    line_end : int
        The line in the input file to stop processing (not inclusive)
    skip_rows : int
        The number of lines at the top of the file to skip before
        data begins
    output_fid : FileHandle
        the file object to write to. Must support the write function

    Returns
    -------
    mols : dict
        Keys are the smiles from the input file, and the value is a
        list of OpenFF molecules with conformers attached.
    output : str
        The contents of what was written to output_fid
    """

    # TODO: unique_smiles=False is broken as it repeats isomers for some reason
    unique_smiles = True

    # Initializing
    i = 0
    rmsd_cutoff = cutoff

    # this is the main object returned
    molecule_set = {}

    ref_smi = smi

    try:
        # If this fails, probably due to stereochemistry. Catch the
        # exception, then enumerate the variations on the SMILES.
        mol = Molecule.from_smiles(smi).to_rdkit()

        smi_list = [mol]

    except openforcefield.utils.toolkits.UndefinedStereochemistryError:

        smi_list = list(EnumerateStereoisomers(Chem.MolFromSmiles(smi)))

        # Clip the isomers here if a limit was specified
        if isomer_max > 0:
            smi_list = smi_list[:isomer_max]

    for i, mol in enumerate(smi_list):
        smi_list[i] = Chem.AddHs(mol)
        for atom in smi_list[i].GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)

    smi_list = [
        smi
        for smi in sorted(
            Chem.MolToSmiles(
                x,
                isomericSmiles=True,
                allHsExplicit=True,
                canonical=True,
                allBondsExplicit=False,
            )
            for x in smi_list
        )
    ]

    if unique_smiles:
        # we are collecting molecules by their specific stereoisomer SMILES
        for smi in smi_list:
            try:
                # this is ridiculous; we enumerated stereoisomers previously,
                # but we still fail to build the molecule. Silently allow...
                # note that this is likely because there is still bond 
                # stereochemistry
                lvl = logging.getLogger("openforcefield").getEffectiveLevel()
                logging.getLogger("openforcefield").setLevel(logging.ERROR)
                molecule_set[smi] = [Molecule.from_smiles(smi, allow_undefined_stereo=True)]
                logging.getLogger("openforcefield").setLevel(lvl)
            except openforcefield.utils.toolkits.UndefinedStereochemistryError:
                # RDKit was unable to determine chirality? Skip...
                pass

    else:
        mols = []
        for smi in smi_list:
            mols.append(Molecule.from_smiles(smi))
        molecule_set[ref_smi] = mols

    for smi in smi_list:

        # Some book keeping to make sure that the stereoisomer SMILES
        # is always printed to the log, but the returned data structure
        # follows the request input settings
        if unique_smiles:
            out_smi = smi
        else:
            out_smi = smi
            smi = ref_smi

        if smi not in molecule_set:
            continue

        for mol in molecule_set[smi]:

            # Not obvious, but i is the number of unique SMILES strings
            # generated (so far) from the input SMILES
            i += 1

            # attempt to generate n_confs, but the actual number could be
            # smaller
            f = io.StringIO()
            with contextlib.redirect_stderr(f):
                with contextlib.redirect_stdout(f):
                    try:
                        mol.generate_conformers(n_conformers=n_confs)
                    except TypeError:
                        pass

            rdmol = mol.to_rdkit()

            L = len(mol.conformers)
            # This will be used to determined whether it should be pruned
            # from the RMSD calculations. If we find it should be pruned
            # just once, it is sufficient to avoid it later in the pairwise
            # processing.
            uniq = list([True] * L)

            # This begins the pairwise RMSD pruner
            if L > 1:

                # The reference conformer for RMSD calculation
                for j in range(L - 1):

                    # A previous loop has determine this specific conformer
                    # is too close to another, so we can entirely skip it
                    if not uniq[j]:
                        continue

                    # since k starts from j+1, we are only looking at the
                    # upper triangle of the comparisons (j < k)
                    for k in range(j + 1, L):

                        rmsd_i = AlignMol(rdmol, rdmol, k, j)
                        r = np.linalg.norm(
                            mol.conformers[k] - mol.conformers[j], axis=1
                        )
                        rmsd_i = r.mean()

                        # Flag this conformer for pruning, and also
                        # prevent it from being used as a reference in the
                        # future comparisons
                        if rmsd_i < rmsd_cutoff:
                            uniq[k] = False

                # hack? how to set conformers explicity if different number than
                # currently stored?
                confs = [
                    mol.conformers[j] for j, add_bool in enumerate(uniq) if add_bool
                ]
                mol._conformers = confs.copy()

    if len(molecule_set) == 0:
       molecule_set[ref_smi] = []

    return molecule_set


def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="A tool to transform a SMILES string into a QCSchema.  Enumerates stereoisomers if the SMILES is ambiguous, and generates conformers."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input file containing smiles strings. Assumes that the file is CSV-like, splits on spaces, and the SMILES is the first column",
    )

    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=9999.0,
        help="Prune conformers less than this cutoff using all pairwise RMSD comparisons (in Angstroms)",
    )

    parser.add_argument(
        "-n",
        "--max-conformers",
        type=int,
        default=1,
        help="The number of conformations to attempt generating",
    )

    parser.add_argument(
        "-s",
        "--line-start",
        type=int,
        default=0,
        help="The line in the input file to start processing",
    )

    parser.add_argument(
        "-e",
        "--line-end",
        type=int,
        help="The line in the input file to stop processing (not inclusive)",
    )

    parser.add_argument(
        "-H",
        "--header-lines",
        type=int,
        default=0,
        help=""" The number of lines at the top of the file to skip before data
        begins""",
    )

    parser.add_argument(
        "-u",
        "--unique-smiles",
        action="store_true",
        help="""If stereoisomers are generated, organize molecules by their
        unambiguous SMILES string""",
    )

    parser.add_argument(
        "-i",
        "--isomers",
        type=int,
        default=-1,
        help="""The number of stereoisomers to keep if multiple are found""",
    )

    parser.add_argument(
        "-o", "--output-file", type=str, help="The file to write the output log to"
    )

    parser.add_argument(
        "-f",
        "--formatted-out",
        type=str,
        help="Write all molecules to a formatted output as qc_schema molecules.  Assumes singlets! Choose either --json, --qcsubmit, or --msgpack as the  the format",
    )

    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Write the formatted output to qc_schema (json) format.",
    )
    parser.add_argument(
        "--qcsubmit",
        action="store_true",
        help="Create and ",
    )

    parser.add_argument(
        "-m",
        "--msgpack",
        action="store_true",
        help="Write the formatted output to qc_schema binary message pack (msgpack).",
    )

    parser.add_argument(
        "--ncpus",
        type=int,
        help="Number of processes to use.",
    )

    args = parser.parse_args()

    if args.output_file is not None:
        fid = open(args.output_file, "w")
    else:
        fid = sys.stdout

    start = args.line_start

    end = args.line_end
    N = detect_file_total_data_lines(args.input, args.header_lines)
    if end is None:
        end = N

    #############################################################################

    # Output settings and general description to the log
    output = ""
    out_line = "# Running entries {:d} to {:d}\n".format(start + 1, end)
    output += out_line

    out_line = "# Generating max {:d} conformers, prune RMSD {:6.2f}\n".format(
        args.max_conformers, args.cutoff
    )
    output += out_line

    if args.unique_smiles:
        out_line = "# Collecting molecules using unique SMILES\n"
    else:
        out_line = "# Collecting molecules by their input SMILES\n"
    output += out_line

    if args.isomers < 0:
        out_line = "# Collecting all stereoisomers found\n"
    else:
        out_line = "# Collecting at most {:d} stereoisomers\n".format(args.isomers)

    output += out_line
    fid.write(out_line)
    fid.flush()
    #############################################################################

    kwargs = dict(
        cutoff=args.cutoff,
        n_confs=args.max_conformers,
        unique_smiles=args.unique_smiles,
        isomer_max=args.isomers,
    )

    mols = {}
    entries, isomers, conformations = 0, 0, 0

    no_mol = []
    no_conf = []
    if args.ncpus == 1:
        for i in tqdm.tqdm(
            range(start, end), total=end - start, ncols=80, disable=True
        ):
            fn_args = (i, end, args.header_lines, args.input)
            mol, out_lines, counts = process_smiles_to_qcschema(*fn_args, **kwargs)
            mols.update(mol)
            fid.write(out_lines)
            entries += counts[0]
            isomers += counts[1]
            conformations += counts[2]
            if counts[1] == 0:
                fid.write("{:22s}Error: Could not build molecule\n".format(""))
                no_mol.extend(mol.keys())
            elif counts[2] == 0:
                fid.write("{:22s}Error: Could not generate any conformers\n".format(""))
                no_conf.extend(mol.keys())
            fid.write(
                "{:22s}Inputs: {:10d} Isomers: {:10d} Conformations: {:10d}\n".format(
                    "", entries, isomers, conformations
                )
            )

    else:
        pool = Pool(processes=args.ncpus)

        work = []

        for i in range(start, end):
            fn_args = (i, end, args.header_lines, args.input)
            unit = pool.apply_async(process_smiles_to_qcschema, fn_args, kwargs)
            work.append(unit)

        for i, unit in tqdm.tqdm(
            enumerate(work), total=len(work), ncols=80, disable=True
        ):
            mol, out_lines, counts = unit.get()
            mols.update(mol)
            fid.write(out_lines)
            entries += counts[0]
            isomers += counts[1]
            conformations += counts[2]
            if counts[1] == 0:
                fid.write("{:22s}Error: Could not build molecule\n".format(""))
                no_mol.extend(mol.keys())
            elif counts[2] == 0:
                fid.write("{:22s}Error: Could not generate any conformers\n".format(""))
                no_conf.extend(mol.keys())

            fid.write(
                "{:22s}Inputs: {:10d} Isomers: {:10d} Conformations: {:10d}\n".format(
                    "", entries, isomers, conformations
                )
            )

        pool.close()

    fid.write("Totals:\n")
    fid.write("  Inputs:        {:8d}\n".format(entries))
    fid.write("  Isomers:       {:8d}\n".format(isomers))
    fid.write("  Conformations: {:8d}\n".format(conformations))

    if len(no_mol) > 0:
        fid.write("\nEntries that could not be built:\n")
        for i, m in enumerate(no_mol, 1):
            fid.write("  {:8d} {:s}\n".format(i, m))

    if len(no_conf) > 0:
        fid.write("\nEntries that could not generate conformers:\n")
        for i, m in enumerate(no_conf, 1):
            fid.write("  {:8d} {:s}\n".format(i, m))


    if args.output_file is not None and fid is not sys.stdout:
        fid.close()

    serialize_method = "json"
    serializer = json
    if args.msgpack:
        serialize_method = "msgpack-ext"
        serializer = msgpack

    if args.json is not None:
        json_mol = {}
        for smi in mols:
            json_mol[smi] = [
                serializer.loads(
                    mol.to_qcschema(conformer=i).serialize(serialize_method)
                )
                for mol in mols[smi]
                for i in range(mol.n_conformers)
            ]

        if args.formatted_out:
            if args.msgpack:
                with open(args.formatted_out, "wb") as fid:
                    msgpack.dump(json_mol, fid)
            elif args.json:
                opener = open
                if args.formatted_out.endswith("bz2"):
                    opener = bz2.open
                elif args.formatted_out.endswith("xz"):
                    opener = lzma.open
                elif args.formatted_out.endswith("lzma"):
                    opener = lzma.open
                with opener(args.formatted_out, "wt") as fid:
                    json.dump(json_mol, fid, indent=2)


if __name__ == "__main__":
    main()
