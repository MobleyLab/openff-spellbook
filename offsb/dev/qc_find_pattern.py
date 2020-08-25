#!/usr/bin/env python3

import sys
import os
import qcfractal.interface as ptl
from rdkit import Chem
from rdkit.Chem import FragmentMatcher


def get_frag_matches(frag, ds):
    """
    inputs:
    frag: str
        The pattern to search for
    ds: QCArchive dataset
        The dataset to search in
            
    returns:
    A dictionary with QC record keys as keys, value is a 2D list of the matches
    in each molecule, e.g. { 'cccc': [[0, 1, 2], [4, 5, 6]] means the pattern was
    in the record key 'cccc' 2 times, with the values of the corresponding indices
    produced by, in this case, RDkit. 
    """

    targets = ds.data.records.keys()
    p = FragmentMatcher.FragmentMatcher()
    p.Init(frag)
    matches = {}
    for mol_smiles in targets:
        smiles_pattern = ds.data.records[mol_smiles].attributes['canonical_isomeric_explicit_hydrogen_mapped_smiles']
        mol = Chem.MolFromSmiles(smiles_pattern)
        mol = Chem.AddHs(mol)
        if(p.HasMatch(mol)):

            # uniquify=True returns nonredundant matches
            for match in p.GetMatches(mol, uniquify=True):
                if(mol_smiles not in matches):
                    matches[mol_smiles] = []
                matches[mol_smiles].append(list(match))
    return matches


def get_ds(client, key, init_cache=True):

    ds = client.get_collection(*key)

    # not sure if this is actually useful here
    if(init_cache):
        ds.status(["default"], status="COMPLETE")

    return ds

def get_qc_attr(ds, keys, attrs):
    """
    input:
    ds: QCArchive dataset
    keys: list of str
    list of records in the dataset to search
    attrs: list of str
    list of keys in the attributes section. Examples are:
       canonical_smiles 
       standard_inchi
    """
    return {key: [ds.data.records[key].attributes[i] for i in attrs] \
            for key in keys}

def main(out=sys.stdout):

    # search pattern, here anything with boron
    pattern = "[#5]"

    client = ptl.FractalClient()

    # list databases to review what to search
    #print(client.list_collections())

    # databases we want to search
    ds_keys = [['OptimizationDataset', 'OpenFF VEHICLe Set 1'], \
               ['TorsionDriveDataset', 'OpenFF Group1 Torsions'], \
               ['OptimizationDataset', 'OpenFF NCI250K Boron 1'], \
               ['OptimizationDataset', 'OpenFF Discrepancy Benchmark 1']]

    match_smiles = set()
    out.write("Searching for {:s}\n".format(pattern))
    for key in ds_keys:
        out.write("Loading dataset {:s} {:s} ... \n".format(*key))
        ds = get_ds(client, key, init_cache=False)

        out.write("Searching .... ")
        matches = get_frag_matches(pattern, ds)
        hits = sum([len(x) for x in matches.values()])
        print("Found", len(matches.keys()), "molecules with", hits, "hits") 

        smiles = get_qc_attr(ds, matches.keys(), ['canonical_smiles'])

        # get_qc_attrs returns a list, so get the first attr we asked for
        [match_smiles.add(i[0]) for i in smiles.values()]

    if(len(match_smiles) > 0):
        out.write("----Results:\n")
        [out.write("{:s}\n".format(smiles)) for smiles in \
            sorted(match_smiles, key=len)]

if(__name__ == "__main__"):
    out = sys.stdout
    if(len(sys.argv) > 1):
        out = open(sys.argv[1], 'w')

    main(out)
    out.close()
