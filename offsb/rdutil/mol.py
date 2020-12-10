#!/usr/bin/env python3
import rdkit.Chem.Draw
from rdkit import Chem
from rdkit import Geometry as RDGeom
from rdkit.Chem import AllChem, FragmentMatcher
from rdkit.Chem.rdchem import Conformer

from ..tools import const


def atom_map(mol):
    map_idx = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
    for v in map_idx.values():
        if v == 0:
            return None
    return map_idx


def atom_map_invert(map_idx):
    inv = {v - 1: k for k,v in map_idx.items()}
    return inv


def build_from_smiles(smiles_pattern):
    mol = Chem.MolFromSmiles(smiles_pattern, sanitize=False)
    Chem.SanitizeMol(
        mol,
        Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
        ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
    )
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    return mol


def embed_qcmol_3d(mol, qcmol):

    map_idx = atom_map(mol)
    assert map_idx is not None
    xyz = qcmol.geometry * const.bohr2angstrom
    coordMap = {i: RDGeom.Point3D(*xyz[map_idx[i] - 1]) for i in map_idx}

    n = mol.GetNumAtoms()
    conf = Conformer(n)
    conf.Set3D(True)
    for i in range(n):
        conf.SetAtomPosition(i, coordMap[i])
    ret = mol.AddConformer(conf, assignId=True)
    # not sure if this can fail, so just accept anything
    return ret


def rdmol_from_smiles_and_qcmol(smiles_pattern, qcmol):

    mol = build_from_smiles(smiles_pattern)
    id = embed_qcmol_3d(mol, qcmol)
    if id < 0:
        raise Exception()

    Chem.rdmolops.AssignStereochemistryFrom3D(mol, id, replaceExistingTags=True)
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol, id, replaceExistingTags=True)

    return mol

def save2d(rdmol, fname=None, indices=False, rdkwargs=None, rdoption_properties=None):
    AllChem.Compute2DCoords(rdmol)
    # rdkit.Chem.Draw.PrepareMolForDrawing(rdmol)
    options = rdkit.Chem.Draw.MolDrawOptions()
    if indices:
        options.addAtomIndices = True
        # get the map; if there is one, use it, otherwise make one
        mapnum = [m.GetAtomMapNum() for m in rdmol.GetAtoms()]
        if not all(mapnum):
            for atom in rdmol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx()+1)
    if rdkwargs is None:
        rdkwargs = {}
    size = rdkwargs.pop("size", (500, 500))

    # if rdoption_properties is not None:
    #     [setattr(options, k, v) for k,v in rdoption_properties.items()]

    # options.fixedBondLength = 20.0
    options.explictMethyl = True
    options.fixedScale = .1
    options.flagCloseContactsDist = 3
    image = rdkit.Chem.Draw.MolToImage(
        rdmol,
        size=size,
        drawingOptions=options,
        **rdkwargs
    )
    if fname is None:
        return image
    else:
        image.save(fname)
