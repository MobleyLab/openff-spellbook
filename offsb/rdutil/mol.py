#!/usr/bin/env python3
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import FragmentMatcher
from rdkit import Geometry as RDGeom
from ..tools import const
from rdkit.Chem.rdchem import Conformer


def build_from_smiles( smiles_pattern):
    mol = Chem.MolFromSmiles( smiles_pattern, sanitize=False)
    Chem.SanitizeMol( mol, Chem.SanitizeFlags.SANITIZE_ALL ^ \
            Chem.SanitizeFlags.SANITIZE_ADJUSTHS ^ \
            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    Chem.SetAromaticity( mol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.SanitizeMol( mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    return mol

def atom_map_invert( map_idx):
    inv = [ ( map_idx[ i] - 1) for i in range(len(map_idx))]
    return inv

def atom_map( mol):
    map_idx = { a.GetIdx() : a.GetAtomMapNum() for a in mol.GetAtoms()}
    valid = True
    for v in map_idx.values():
        if v == 0:
            return None
    return map_idx

def embed_qcmol_3d( mol, qcmol):
    map_idx = atom_map( mol)
    assert map_idx is not None
    xyz = qcmol.get("geometry") * const.bohr2angstrom
    coordMap = {i : RDGeom.Point3D( *xyz[ map_idx[ i]-1]) for i in map_idx}

    n = mol.GetNumAtoms()
    conf = Conformer(n)
    conf.Set3D(True)
    for i in range(n):
        conf.SetAtomPosition(i, coordMap[i])
    ret = mol.AddConformer(conf, assignId=True)
    # not sure if this can fail, so just accept anything 
    return ret

    #ret = AllChem.EmbedMolecule( mol, coordMap=coordMap, forceTol=.01, 
    #        ignoreSmoothingFailures=True, maxAttempts=100000,
    #        useBasicKnowledge=False )
    #return ret
