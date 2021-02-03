import tempfile
from abc import ABC

import geometric
import geometric.internal
import geometric.molecule

import offsb.rdutil.mol
import offsb.treedi.tree
import offsb.ui.qcasb


class _DummyTree:
    __slots__ = ["source"]

class InteralCoordinateGeometricOperation(offsb.treedi.tree.TreeOperation, ABC):
    def __init__(self, source_tree, name, verbose=False):
        super().__init__(source_tree, name, verbose=verbose)

        self._select = "Entry"

        source = self.source
        self.source = _DummyTree
        self.source.source = source

    def op(self, node, partition):
        pass

    def _generate_apply_kwargs(self, i, target, kwargs=None):

        entry = self.source.source.db[target.payload]["data"]

        out_str = ""

        kwargs = {}

        smi = entry.attributes["canonical_isomeric_explicit_hydrogen_mapped_smiles"]
        if "initial_molecule" in entry.dict():
            qcid = entry.dict()["initial_molecule"]
        elif "initial_molecules" in entry.dict():
            qcid = entry.dict()["initial_molecules"]
        else:
            out_str += "{:d} initial mol was empty for this record: {:s}; target {:s}".format(
                i, str(qcid), target.payload
            )
            return {"error": out_str}

        if isinstance(qcid, set):
            qcid = list(qcid)
        if isinstance(qcid, list):
            qcid = str(qcid[0])

        qcmolid = "QCM-" + qcid

        if qcmolid not in self.source.source.db:
            out_str += "{:d} initial mol was not cached: {:s}; target {:s}".format(
                i, str(qcmolid), target.payload
            )
            return {"error": out_str}

        if "data" in self.source.source.db.get(qcmolid):
            qcmol = self.source.source.db.get(qcmolid).get("data")
        else:
            out_str += (
                "{:d} initial mol was cached correctly: {:s}; target {:s}".format(
                    i, str(qcmolid), target.payload
                )
            )
            return {"error": out_str}

        # kwargs["smi"] = smi
        kwargs["qcmol"] = qcmol

        # kwargs.update({"name": self.name, "entry": str(entry)})
        return kwargs

    def _apply_initialize(self, targets):
        pass

    def _apply_finalize(self, targets):
        pass

    def _unpack_result(self, ret):
        self.db[ret[0]] = {"data": ret[1]}

    @staticmethod
    def apply_single(i, target, **kwargs):

        out_str = ""

        if "error" in kwargs:
            return {
                target.payload: kwargs["error"],
                "return": [target.payload, {}],
            }

        qcmol = kwargs["qcmol"]

        with tempfile.NamedTemporaryFile(mode="wt") as f:
            offsb.qcarchive.qcmol_to_xyz(qcmol, fnm=f.name)
            gmol = geometric.molecule.Molecule(f.name, ftype="xyz")

        ic_prims = geometric.internal.PrimitiveInternalCoordinates(
            gmol,
            build=True,
            connect=True,
            addcart=False,
            constraints=None,
            cvals=None,
        )

        return {
            target.payload: out_str,
            "return": [target.payload, ic_prims],
        }

    def apply(self, targets=None):
        super().apply(self._select, targets=targets)
