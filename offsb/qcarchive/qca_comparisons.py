#!/usr/bin/env python3

def match_canonical_isomeric_explicit_hydrogen_smiles( entryA, entryB):
    """
        Compare equality of two entries by comparing their canonical isomeric
        explicit hydrogen smiles

        Input: QCArchive entry types (TDEntry, OptEntry, etc.)
    """

    key = 'canonical_isomeric_explicit_hydrogen_smiles'
    #if entryA is None or entryB is None:
    #    return False
    A = entryA.attributes[ key]
    B = entryB.attributes[ key]
    if A == B:
        return True

    return False
