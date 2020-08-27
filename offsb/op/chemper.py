#!/usr/bin/env python3
import os
import pdb
import treedi
import treedi.tree
import simtk.unit
import simtk.unit as unit
from simtk import openmm
from simtk.openmm.app.simulation import Simulation as OpenMMSimulation
import openforcefield as oFF
from openforcefield.typing.engines.smirnoff.parameters import (
    UnassignedProperTorsionParameterException,
)
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
import smirnoff99frosst as ff
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import FragmentMatcher
from rdkit import Geometry as RDGeom
from ..tools import const
from .. import rdutil
from .. import qcarchive as qca
import offsb.op.geometry
import copy
import numpy as np
from multiprocessing import Pool
import threading
import sys
import tempfile

from chemper.mol_toolkits.cp_rdk import Mol
from chemper.smirksify import SMIRKSifier, print_smirks
import contextlib
import re


class ChemperOperation(treedi.tree.TreeOperation):
    """ Given a set of indices, find the SMARTS representation

    """

    _LABEL_TYPES = {
        "Bonds": 2,
        "Angles": 3,
        "vdW": 1,
        "ProperTorsions": 4,
        "ImproperTorsions": 4,
    }
    _LABEL_LENS = {"b": 2, "a": 3, "n": 1, "t": 4, "i": 4}

    _OUT_STR = {
        "n": "{:16s}",
        "b": "{:16s} {:6s} {:16s}",
        "a": "{:16s} {:6s} {:16s} {:6s} {:16s}",
        "t": "{:16s} {:6s} {:16s} {:6s} {:16s} {:6s} {:16s}",
        "i": "{:16s} {:6s} {:16s} ( {:6s} {:16s} ) {:6s} {:16s}",
    }

    def __init__(self, source, name):
        super().__init__(source, name)
        self._select = "Entry"
        print("My db is", self.db)

    @staticmethod
    def _smirks_splitter(smirks, atoms=2):
        """
        """

        atom = r"\[([^[.]*):[0-9]*\]"
        bond = r"([^[.]*)"
        smirks = smirks.strip("()")
        if atoms <= 0:
            return tuple()
        pat_str = atom
        for i in range(1, atoms):
            pat_str += bond + atom
        pat = re.compile(pat_str)
        ret = pat.match(smirks)
        return ret.groups()

    def _unpack_result(self, val):
        self.db.update(val)

    def _generate_apply_kwargs(self, i, target, kwargs={}):

        # labels = self.source.db[target.payload]["data"]
        entry = self.source.source.db[target.payload]["data"]

        out_str = ""
        smi = entry.attributes["canonical_isomeric_explicit_hydrogen_mapped_smiles"]
        if "initial_molecule" in entry.dict():
            qcid = entry.dict()["initial_molecule"]
        elif "initial_molecules" in entry.dict():
            qcid = entry.dict()["initial_molecules"]
        else:
            out_str += "{:d} initial mol was empty: {:s}".format(i, str(qcid))
            return {"error": out_str}

        if isinstance(qcid, set):
            qcid = list(qcid)
        if isinstance(qcid, list):
            qcid = str(qcid[0])

        qcmolid = "QCM-" + qcid

        if qcmolid not in self.source.source.db:
            out_str += "{:d} initial mol was empty: {:s}".format(i, str(qcmolid))
            return {"error": out_str}

        if "data" in self.source.source.db.get(qcmolid):
            qcmol = self.source.source.db.get(qcmolid).get("data")
        else:
            out_str += "{:d} initial mol was empty: {:s}".format(i, str(qcmolid))
            return {"error": out_str}

        mol = offsb.rdutil.mol.rdmol_from_smiles_and_qcmol(smi, qcmol)

        obj = self.source.db[self.source[target.index].payload]
        masks = obj["data"]

        kwargs.update({
            "masks": masks,
            "mol": mol,
            "name": self.name,
            "entry": str(entry)
        })
        return kwargs

    @staticmethod
    def apply_single(i, target, kwargs):
        """
        From an entry, builds the initial molecule (for e.g. stereochemistry)
        i : int
            the identifying number of job
        target : EntryNode
            A node that 
        """

        if "error" in kwargs:
            return {
                target.payload: "Could not run Chemper:\n" + kwargs["error"],
                "return": {"data": {}},
            }

        mol = kwargs["mol"]
        masks = kwargs["masks"]

        # mol = Mol(mol)
        # values will be 1-index
        map_idx = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}

        # chemper uses 0-indexing
        map_inv = {v - 1: k for k, v in map_idx.items()}
        chemper = {}
        for mask in map(tuple, masks):
            mapped_bond = tuple([map_inv[i] for i in mask])
            pat = ("name", [[mapped_bond]])
            # with open("/dev/null", "w") as f:
            #     with contextlib.redirect_stdout(f):
            fier = SMIRKSifier([mol], [pat], verbose=False, max_layers=3)
            chemper[mask] = ChemperOperation._smirks_splitter(
                fier.current_smirks[0][1], len(mask)
            )

        return {target.payload: None, "return": {target.payload: {"data": chemper}}}

        # for label_type in self._LABEL_TYPES:
        #     chemper[label_type] = {}
        #     for bond, lbl in labels[label_type].items():
        #         mapped_bond = tuple([map_inv[i] for i in bond])
        #         pat = ("name", [[mapped_bond]])
        #         with open("/dev/null", "w") as f:
        #             with contextlib.redirect_stdout(f):
        #                 fier = SMIRKSifier([mol], [pat], verbose=True, max_layers=3)
        #         chemper[label_type][bond] = (fier.current_smirks[0][1], lbl) return {target.payload: out_str, "return": {"data": chemper}}

    def apply(self, targets=None):
        super().apply(targets=targets, select=self._select)

    def op(self, mol, idx):
        return self.apply(mol, idx)

    def report(self):

        chemper_inv = {}

        for term in "nbait":
            LABELS = sorted(
                self.db.keys(), key=lambda x: int(re.search(r"[0-9]+", x).group(0))
            )
            for lbl in LABELS:
                if term not in lbl:
                    continue
                smirks_list = self.db[lbl]
                print(lbl, len(smirks_list))
                uniq_sorted = {}
                for smirks in smirks_list:
                    n_uniq = len([x for x in smirks_list if smirks == x])
                    tokens = self._smirks_splitter(
                        smirks, atoms=self._LABEL_LENS[lbl[0]]
                    )
                    if lbl[0] == "a":
                        if tokens[4] < tokens[0]:
                            tokens = tokens[::-1]
                    elif lbl[0] == "b":
                        if tokens[2] < tokens[0]:
                            tokens = tokens[::-1]
                    elif lbl[0] == "t":
                        if tokens[6] < tokens[0]:
                            tokens = tokens[::-1]
                    elif lbl[0] in "i":
                        atoms = tokens[::2]
                        if atoms[2] < atoms[0]:
                            if atoms[2] < atoms[3]:
                                if atoms[0] < atoms[3]:
                                    # 2 1 0 3
                                    idx = [4, 3, 2, 1, 0, 5, 6]
                                else:
                                    # 2 1 3 0
                                    idx = [4, 3, 2, 5, 6, 1, 0]
                            else:
                                # 3 1 2 0
                                idx = [6, 5, 2, 3, 4, 1, 0]
                        else:
                            if atoms[2] < atoms[3]:
                                # 0 1 2 3
                                idx = [0, 1, 2, 3, 4, 5, 6]
                            else:
                                # 0 1 3 2
                                idx = [0, 1, 2, 5, 6, 3, 4]
                        tokens = [tokens[i] for i in idx]
                    tokens = tuple(tokens)

                    # tokens = tuple(sorted(tokens))
                    if tokens not in uniq_sorted:
                        uniq_sorted[tokens] = 0
                    uniq_sorted[tokens] += 1

                lines = []
                for smirks, n_uniq in uniq_sorted.items():
                    out_str = self._OUT_STR[lbl[0]]
                    line = "    {: 8d}  {:s}".format(n_uniq, out_str.format(*smirks))
                    lines.append(line)
                    for item in smirks:
                        if item not in chemper_inv:
                            chemper_inv[item] = []
                        chemper_inv[item].append(lbl)
                for line in sorted(lines, key=lambda x: "".join(x.split()[1:])):
                    print(line)

    ########################################
