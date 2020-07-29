#!/usr/bin/env python3

import sys
import json
import numpy as np
import msgpack

import openforcefield
from openforcefield.topology.molecule import Molecule

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions


def expand_smiles_to_qcschema(
        input_fnm,
        cutoff=None,
        n_confs=1,
        unique_smiles=False,
        isomer_max=-1,
        line_start=0,
        line_end=None,
        skip_rows=0,
        output_fid=None,
        debug=False):
    """
    Load a file containing smiles strings, and generate stereoisomers and
    conformers for each stereoisomer.

    Parameters
    ----------
    input_fnm : str, The input filename to read SMILES from
    cutoff : float, During the all-pairwise RMSD calculation, remove
        molecules that are less than this cutoff value apart
    n_confs : int, The number of conformations to attempt generating
    unique_smiles : bool, If stereoisomers are generated, organize molecules by
        their unambiguous SMILES string
    isomers : int, The number of stereoisomers to keep if multiple are found.
        The default of -1 means keep all found.
    line_start : int, The line in the input file to start processing
    line_end : int, The line in the input file to stop processing
        (not inclusive)
    skip_rows : int, The number of lines at the top of the file to skip before
        data begins
    output_fid : FileHandle, the file object to write to. Must support the
        write function

    Returns
    -------
    mols : dict, keys are the smiles from the input file, and the value is a
        list of OpenFF molecules with conformers attached.
    output : str, the contents of what was written to output_fid
    """

    # Initializing
    skip_rows = 1
    i = 0
    rmsd_cutoff = cutoff

    # this is the main object returned
    molecule_set = {}
    total_mol = 0

    N = -skip_rows
    for line in open(input_fnm, 'r'):
        N += 1
    if line_end is None:
        line_end = N

    start = line_start 
    stop = line_end

    fid = output_fid
    dowrite = fid is not None
    #############################################################################

    # Output settings and general description to the log
    output = ""
    out_line = "# Running entries {:d} to {:d}\n".format(start, stop)
    if dowrite:
        fid.write(out_line)
    output += out_line

    out_line = "# Generating max {:d} conformers, prune RMSD {:6.2f}\n".format(
        n_confs, rmsd_cutoff)
    if dowrite:
        fid.write(out_line)
    output += out_line

    if unique_smiles:
        out_line = "# Collecting molecules using unique SMILES\n"
    else:
        out_line = "# Collecting molecules by their input SMILES\n"
    if dowrite:
        fid.write(out_line)
    output += out_line

    if isomer_max < 0:
        out_line = "# Collection all stereoisomers found\n"
    else:
        out_line = "# Collecting at most {:d} stereoisomers\n".format(isomer_max)
    if dowrite:
        fid.write(out_line)
    output += out_line
    #############################################################################

    for n, line in enumerate(open(input_fnm, 'r')):
        spline = line.split()
        if n < skip_rows or n < start:
            n += 1
            continue
        if n == stop:
            break

        smi = spline[0]
        ref_smi = spline[0]

        out_line = "{:8d} / {:8d} SMILES: {:s}".format(
            n - skip_rows+1, N, smi)
        if dowrite:
            fid.write(out_line)
        output += out_line

        try:
            # If this fails, probably due to stereochemistry. Catch the
            # exception, then enumerate the variations on the SMILES.
            mol = Molecule.from_smiles(smi)
            i += 1
            smi_list = [smi]

        except openforcefield.utils.toolkits.UndefinedStereochemistryError:

            isomers = tuple(EnumerateStereoisomers(Chem.MolFromSmiles(smi)))
            smi_list = [smi for smi in sorted(
                Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers)]

            # Clip the isomers here if a limit was specified
            if isomer_max > 0:
                smi_list = smi_list[:isomer_max]

            out_line = "\n"
            if dowrite:
                fid.write(out_line)
            output += out_line

        if not unique_smiles:
            # if we are aggregating under the same SMILES, collect the mols
            # under the reference SMILES
            mols = [Molecule.from_smiles(smi) for smi in smi_list]
            molecule_set[ref_smi] = mols
        else:
            for smi in smi_list:
                molecule_set[smi] = [Molecule.from_smiles(smi)]

        for smi in smi_list:
            initial_mol_N = 0

            # Some book keeping to make sure that the stereoisomer SMILES
            # is always printed to the log, but the returned data structure
            # follows the request input settings
            if not unique_smiles:
                out_smi = smi
                smi = ref_smi
            else:
                out_smi = smi

            for mol in molecule_set[smi]:

                # Not obvious, but i is the number of unique SMILES strings
                # generated (so far) from the input SMILES
                i += 1
                out_line = "{:8d}  ISOMER: {:s}\n".format(i, out_smi)
                if dowrite:
                    fid.write(out_line)
                    fid.flush()
                output += out_line

                # attempt to generate n_confs, but the actual number could be
                # smaller
                mol.generate_conformers(n_conformers=n_confs)
                L = len(mol.conformers)

                # This will be used to determined whether it should be pruned
                # from the RMSD calculations. If we find it should be pruned
                # just once, it is sufficient to avoid it later in the pairwise
                # processing.
                uniq = list([True] * L)

                out_str = ""

                if L > 1:
                    if debug:
                        out_str += "        RMSD: "

                    # The reference conformer for RMSD calculation
                    for j in range(L - 1):
                        if debug:
                            out_str += "Ref_{:03d} ".format(j)

                        # A previous loop has determine this specific conformer
                        # is too close to another, so we can entirely skip it
                        if not uniq[j]:
                            if debug:
                                out_str += "X "
                            continue

                        # Rather than print every rmsd values, print the min,
                        # max, and mean at the end as a debugging measure
                        rmsd = []

                        # since k starts from j+1, we are only looking at the
                        # upper triangle of the comparisons (j < k)
                        for k in range(j + 1, L):

                            r = np.linalg.norm(
                                mol.conformers[k] - mol.conformers[j], axis=1)
                            rmsd_i = r.mean()
                            rmsd.append(rmsd_i)

                            # Flag this conformer for pruning, and also
                            # prevent it from being used as a reference in the
                            # future comparisons
                            if rmsd_i < rmsd_cutoff:
                                uniq[k] = False

                        min_rmsd = min(rmsd)
                        max_rmsd = max(rmsd)
                        mean_rmsd = sum(rmsd) / len(rmsd)
                        if debug:
                            out_str_tmp = "min={:6.2f} max={:6.2} mean={:6.2} "
                            out_str += out_str_tmp.format(min_rmsd, max_rmsd,
                                mean_rmsd)

                    # hack? how to set conformers explicity if different number?
                    confs = [mol.conformers[j] for j, add_bool
                        in enumerate(uniq) if add_bool]
                    mol._conformers = confs.copy()

                initial_mol_N += len(mol.conformers)
                total_mol += initial_mol_N

                # output the report str on the RMSD calcs
                out_str_tmp = "{:9s} Kept: {:4d} / {:4d} {:s} Total: {:12d}\n"
                out_str = out_str_tmp.format("", len(mol.conformers),
                    L, out_str, total_mol)
                output += out_str
                if dowrite:
                    fid.write(out_str)
                    fid.flush()

    if dowrite:
        fid.close()

    return molecule_set, output

def main():

    import argparse
    parser = argparse.ArgumentParser(
        description="""
        A tool to transform a SMILES string into a QCSchema.
        Enumerates stereoisomers if the SMILES is ambiguous, and generates
        conformers.
        """
    )
    parser.add_argument("input", type=str,
        help="""Input file containing smiles strings. Assumes that the file is
        CSV-like, splits on spaces, and the SMILES is the first column""")

    parser.add_argument("-c", "--cutoff", type=float,
        help=""" Prune conformers less than this cutoff using all pairwise RMSD
        comparisons (in Angstroms)""")

    parser.add_argument("-n", "--max-conformers", type=int,
        help="The number of conformations to attempt generating")

    parser.add_argument("-s", "--line-start", type=int, default=0,
        help="The line in the input file to start processing")

    parser.add_argument("-e", "--line-end", type=int,
        help="The line in the input file to stop processing (not inclusive)")

    parser.add_argument("-H", "--header-lines", type=int,
        help=""" The number of lines at the top of the file to skip before data
        begins""")

    parser.add_argument("-u", "--unique-smiles", action="store_true",
        help="""If stereoisomers are generated, organize molecules by their
        unambiguous SMILES string""")

    parser.add_argument("-i", "--isomers", type=int, default=-1,
        help="""The number of stereoisomers to keep if multiple are found""")

    parser.add_argument("-o", "--output-file", type=str,
        help="The file to write the output log to")

    parser.add_argument("-f", "--formatted-out", type=str,
        help="""
        Write all molecules to a formatted output as qc_schema molecules.
        Assumes singlets!
        Only choose one option: --json or --msgpack""")

    parser.add_argument("-j", "--json", action="store_true",
        help="Write the formatted output to qc_schema (json) format.")

    parser.add_argument("-m", "--msgpack", action="store_true",
        help="""Write the formatted output to qc_schema binary message pack
        (msgpack)""")

    args = parser.parse_args()

    if args.output_file is not None:
        fid = open(args.output_file, 'w')
    else:
        fid = None

    mols, out = expand_smiles_to_qcschema(
        args.input,
        cutoff=args.cutoff,
        n_confs=args.max_conformers,
        unique_smiles=args.unique_smiles,
        isomer_max=args.isomers,
        line_start=args.line_start,
        line_end=args.line_end,
        skip_rows=args.header_lines,
        output_fid=fid)

    if args.output_file is not None:
        fid.close()
    else:
        print(out)

    serialize_method = "json"
    if args.msgpack:
        serialize_method = "msgpack-ext"

    if args.json is not None:
        json_mol = {}
        for smi in mols:
            json_mol[smi] = [
                mol.to_qcschema(conformer=i).serialize(serialize_method)
                for mol in mols[smi]
                for i in range(mol.n_conformers)
            ]

        if args.formatted_out:
            if args.msgpack:
                with open(args.formatted_out, 'wb') as fid:
                    msgpack.dump(json_mol, fid)
            elif args.json:
                with open(args.formatted_out, 'w') as fid:
                    json.dump(json_mol, fid)


if __name__ == "__main__":
    main()
