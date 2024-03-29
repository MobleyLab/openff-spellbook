#!/usr/bin/env python3
import re

import offsb.chem.types
import offsb.op.geometry
import offsb.treedi
import offsb.treedi.tree

from chemper.smirksify import SMIRKSifier


class ChemperOperation(offsb.treedi.tree.TreeOperation):
    """Given a set of indices, find the SMARTS representation"""

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
        self._processes = None
        self.chembit = False
        self.chemgraph = False

        self.chemgraph_distinguish_hydrogen = True  # likely needed for vdW?
        self.chemgraph_explicit_hydrogen = False

        self.chemgraph_depth_limit = None
        self.chemgraph_min_depth = 0

        self.openff_compat = True

    @property
    def processes(self):
        return self._processes

    @processes.setter
    def processes(self, n):

        if n is None:
            self._processes = n
        else:
            self._processes = int(n)

    @staticmethod
    def _smirks_splitter(smirks, atoms=2):
        """"""

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

    def _generate_apply_kwargs(self, i, target, kwargs=None):

        # labels = self.source.db[target.payload]["data"]
        entry = self.source.source.db[target.payload]["data"]

        if kwargs is None:
            kwargs = {}

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

        kwargs.update(
            {
                "masks": masks,
                "mol": mol,
                "smi": smi,
                "name": self.name,
                "entry": str(entry),
                "openff_compat": self.openff_compat,
                "chembit": self.chembit,
                "chemgraph": self.chemgraph,
                "chemgraph_distinguish_hydrogen": self.chemgraph_distinguish_hydrogen,
                "chemgraph_depth_limit": self.chemgraph_depth_limit,
                "chemgraph_min_depth": self.chemgraph_min_depth,
                "chemgraph_explicit_hydrogen": self.chemgraph_explicit_hydrogen,
            }
        )
        return kwargs

    @staticmethod
    def apply_single(i, target, **kwargs):
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
        chembit = kwargs.get("chembit", False)
        chemgraph = kwargs.get("chemgraph", False)
        chemgraph_distinguish_hydrogen = kwargs.get(
            "chemgraph_distinguish_hydrogen", True
        )
        chemgraph_depth_limit = kwargs.get("chemgraph_depth_limit", None)
        chemgraph_min_depth = kwargs.get("chemgraph_min_depth", 0)
        chemgraph_explicit_hydrogen = kwargs.get(
            "chemgraph_explicit_hydrogen", False
        )  # likely needed for vdW terms

        openff_compat = kwargs.get("openff_compat", True)
        smi = kwargs['smi']

        prim_to_graph = None
        if chembit:
            import offsb.chem.types

            prim_to_graph = {
                "n": offsb.chem.types.AtomType,
                "b": offsb.chem.types.BondGraph,
                "a": offsb.chem.types.AngleGraph,
                "i": offsb.chem.types.OutOfPlaneGraph,
                "t": offsb.chem.types.TorsionGraph,
            }

        mol = kwargs["mol"]
        all_masks = kwargs["masks"]

        # mol = Mol(mol)
        # values will be 1-index
        # map_idx = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}

        # chemper uses 0-indexing
        # map_inv = {v - 1: k for k, v in map_idx.items()}
        chemper = {}
        # for query, masks in all_masks.items():
        #     for mask in map(tuple, masks):
        #         # mapped_bond = tuple([map_inv[i] for i in mask])
        #         # pat = ("name", [[mapped_bond]])
        #         pat = ("name", [[mask]])
        #         # with open("/dev/null", "w") as f:
        #         #     with contextlib.redirect_stdout(f):

        #         # If this fails, likely the indices were messed up
        #         try:
        #             fier = SMIRKSifier([mol], [pat], verbose=False, max_layers=1)
        #             chemper[mask] = ChemperOperation._smirks_splitter(
        #                 fier.current_smirks[0][1], len(mask)
        #             )
        #             if chembit:

        #                 ic_type = 0
        #                 if len(mask) == 1:
        #                     ic_type = "n"
        #                 if len(mask) == 2:
        #                     ic_type = "b"
        #                 elif len(mask) == 3:
        #                     ic_type = "a"
        #                 elif len(mask) == 4:
        #                     if "(" in fier.current_smirks[0][1]:
        #                         ic_type = "i"
        #                     else:
        #                         ic_type = "t"

        #                 chemper[mask] = prim_to_graph[ic_type].from_string_list(
        #                     chemper[mask], sorted=True
        #                 )

        #         except Exception as e:
        #             breakpoint()
        #             print(e)

        if chemgraph:
            import offsb.chem.graph

            M = offsb.chem.graph.MoleculeGraph.from_smiles(
                smi,
                distinguish_hydrogen=chemgraph_distinguish_hydrogen,
                depth_limit=chemgraph_depth_limit,
                min_depth=chemgraph_min_depth,
                explicit_hydrogen=chemgraph_explicit_hydrogen,
                openff_compat=openff_compat
            )

            # M = offsb.chem.graph.MoleculeGraph.from_ic_primitives(
            #     chemper,
            #     distinguish_hydrogen=chemgraph_distinguish_hydrogen,
            #     depth_limit=chemgraph_depth_limit,
            #     min_depth=chemgraph_min_depth,
            #     explicit_hydrogen=chemgraph_explicit_hydrogen,
            # )
            chemper["graph"] = M
        return {target.payload: None, "return": {target.payload: {"data": chemper}}}

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


