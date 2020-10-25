#!/usr/bin/env python3

"""
"""

import abc
import contextlib
import io
import logging
import os

from pkg_resources import iter_entry_points

import offsb.rdutil.mol
import offsb.treedi
import offsb.treedi.tree
import openforcefield.typing.engines.smirnoff as OFF
from openforcefield.topology import Molecule
from offsb.search.smiles import SmilesSearchTree
from offsb.tools.util import flatten_list
from offsb.treedi.tree import DEFAULT_DB, PartitionTree


class _DummyTree:
    __slots__ = ["source"]


class OpenForceFieldTreeBase(offsb.treedi.tree.TreeOperation, abc.ABC):
    def __init__(self, source_tree, name, filename=None):
        super().__init__(source_tree, name)

        # This tree operates on entries
        self._select = "Entry"

        if filename is not None:

            logger = logging.getLogger("openforcefield")
            level = logger.getEffectiveLevel()
            logger.setLevel(level=logging.ERROR)
            ext = ".offxml"
            if not filename.endswith(ext):
                filename += ext
            self.filename = filename

            found = False
            for entry_point in ["."] + list(
                iter_entry_points(group="openforcefield.smirnoff_forcefield_directory")
            ):
                if type(entry_point) == str:
                    pth = entry_point
                else:
                    pth = entry_point.load()()[0]
                abspth = os.path.join(pth, filename)
                self.logger.info("Searching + {}".format(abspth))
                if os.path.exists(abspth):
                    self.abs_path = abspth
                    self.logger.info("Found {}".format(abspth))
                    found = True
                    break
            if not found:
                raise Exception("Forcefield could not be found")
            self.forcefield = OFF.ForceField(
                self.abs_path, disable_version_check=True
            )
            logger.setLevel(level=level)

    @classmethod
    def from_forcefield(cls, forcefield, source_tree, name):
        obj = cls(source_tree, name)
        obj.forcefield = forcefield
        return obj

    def to_pickle_str(self):

        tmp = self.forcefield
        self.forcefield = None
        obj = super().to_pickle_str()
        self.forcefield = tmp
        return obj

    def isolate(self):
        super().isolate()
        self.forcefield = None

    def associate(self, source):
        super().associate(source)
        if self.forcefield is None:
            self.forcefield = OFF.ForceField(
                self.abs_path, disable_version_check=True
            )

    def count_oFF_labels(self, node):
        """
        provide a summary of the oFF labels found
        """
        return

    def _generate_apply_kwargs(self, i, target, kwargs=None):

        # labels = self.source.db[target.payload]["data"]
        entry = self.source.source.db[target.payload]["data"]

        out_str = ""

        if kwargs is None:
            kwargs = {}

        qcmol = kwargs.get("qcmol")
        smi = kwargs.get("smi")

        if qcmol is None or smi is None:
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

            kwargs["smi"] = smi
            kwargs["qcmol"] = qcmol

        masks = kwargs.get("masks")
        if masks is None:
            obj = self.source.db[self.source[target.index].payload]
            masks = obj["data"]
            kwargs["masks"] = flatten_list([v for v in masks.values()], times=1)

        kwargs.update({"name": self.name, "entry": str(entry)})
        return kwargs

    def op(self, node, partition):
        pass

    def _apply_initialize(self, targets):

        all_labels = self.db.get(self.source.root().payload)

        if all_labels is None:
            self.all_labels = {"data": {}}
            self.db[self.source.root().payload] = {"data": {}}

    def _apply_finalize(self, targets):
        pass

    def _unpack_result(self, ret):

        self.db[ret[0]] = {"data": ret[1]}

        if self.source.root().payload not in self.db:
            self.db[self.source.root().payload] = {}

        self.db[self.source.root().payload]["data"].update(ret[2])

    def apply_single(self, i, target, **kwargs):

        out_str = ""
        all_labels = {}
        out_dict = {}
        smi = kwargs["smi"]
        # qcmol = kwargs['qcmol']
        # mol = kwargs.get('mol')
        mmol = kwargs.get("mmol")

        labels = kwargs.get("labels")
        if labels is None:
            # mol = kwargs['mol']
            # map_idx = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
            if mmol is None:
                with io.StringIO() as f:
                    with contextlib.redirect_stderr(f):
                        # mmol = oFF.topology.Molecule.from_rdkit(mol,
                        #         allow_undefined_stereo=True)
                        mmol = Molecule.from_smiles(
                            smi, allow_undefined_stereo=True
                        )
                    for line in f:
                        if "not error because allow_undefined_stereo" not in line:
                            print(line)

            # just skip molecules that oFF can't handle for whatever reason
            try:
                top = oFF.topology.Topology.from_molecules(mmol)
            except AssertionError as e:
                out_str += "FAILED TO BUILD OFF MOL:\n"
                out_str += str(e)
                # pdb.set_trace()
                return {
                    target.payload: out_str,
                    "return": [target.payload, out_dict, all_labels],
                    "error": out_str,
                }

            labels = self.forcefield.label_molecules(top)[0]

        # keys = [
        #     "Bonds",
        #     "Angles",
        #     "ProperTorsions",
        #     "vdW",
        #     "ImproperTorsions",
        #     "Electrostatics",
        #     "ToolkitAM1BCC",
        # ]
        # keys = ["Bonds", "Angles", "ProperTorsions", "ImproperTorsions", "vdW"]

        shared_members = ["smirks", "id"]

        uniq_members = self._unique_members
        key = self._key

        params = labels.get(key)
        out_dict[key] = {}

        if key not in all_labels:
            all_labels.update({key: {}})

        masks = kwargs["masks"]
        if masks is None:
            masks_sorted = []
        else:
            if isinstance(masks, dict):
                masks = masks[key]
            masks_sorted = masks  # [sorted(k) for k in masks]

        for mask in map(tuple, masks_sorted):

            # we are using the chemical indices; no map needed
            atoms = tuple(mask)
            # FF indices are unmapped (ordered by smiles appearance)
            # map them now since we work in the mapped CMILES in QCA
            val = params.get(atoms)


            # since we are only applying labels to the underlying partition
            # skip any labels that were produced that are extraneous
            # But, in the end, allow it to just work if no mask is present
            if val is None:
                # this means that the mask is not in the param list
                # if [*] used, then unexpected, but possible to only
                # want specific parameters
                # also allows easy identification of unparameterized
                # atoms, e.g. impropers

                out_dict[key][mask] = None
                continue

            out_dict[key][mask] = val.id

            ret = {}
            for name in shared_members + uniq_members:
                prop = getattr(val, name)
                ret[name] = prop

            if val.id not in all_labels[key]:
                all_labels[key][val.id] = ret

        return {
            target.payload: out_str,
            "return": [target.payload, out_dict, all_labels],
        }

    def apply(self, targets=None):
        super().apply(self._select, targets=targets)


class OpenForceFieldvdWTree(OpenForceFieldTreeBase):
    def __init__(self, source_tree, name, filename=None):

        self._key = "vdW"
        self._unique_members = ["rmin_half", "epsilon"]

        if not issubclass(type(source_tree), PartitionTree):
            partition = SmilesSearchTree("[*]", source_tree, self._key)
            partition.apply()
            source_tree = partition

        super().__init__(source_tree, name, filename)


class OpenForceFieldBondTree(OpenForceFieldTreeBase):
    def __init__(self, source_tree, name, filename=None):

        self._key = "Bonds"
        self._unique_members = ["k", "length"]

        if not issubclass(type(source_tree), PartitionTree):
            partition = SmilesSearchTree("[*]~[*]", source_tree, self._key)
            partition.apply()
            source_tree = partition

        super().__init__(source_tree, name, filename)


class OpenForceFieldAngleTree(OpenForceFieldTreeBase):
    def __init__(self, source_tree, name, filename=None):

        self._key = "Angles"
        self._unique_members = ["k", "angle"]

        if not issubclass(type(source_tree), PartitionTree):
            partition = SmilesSearchTree("[*]~[*]~[*]", source_tree, self._key)
            source_tree = partition
            partition.apply()

        super().__init__(source_tree, name, filename)


class OpenForceFieldTorsionTree(OpenForceFieldTreeBase):
    def __init__(self, source_tree, name, filename=None):

        self._key = "ProperTorsions"
        self._unique_members = ["k", "periodicity", "phase"]

        if not issubclass(type(source_tree), PartitionTree):
            partition = SmilesSearchTree("[*]~[*]~[*]~[*]", source_tree, self._key)
            partition.apply()
            source_tree = partition

        super().__init__(source_tree, name, filename)


class OpenForceFieldImproperTorsionTree(OpenForceFieldTreeBase):
    def __init__(self, source_tree, name, filename=None):

        self._key = "ImproperTorsions"
        self._unique_members = ["k", "periodicity", "phase"]

        if not issubclass(type(source_tree), PartitionTree):
            partition = SmilesSearchTree("[*]~[*](~[*])~[*]", source_tree, self._key)
            partition.apply()
            source_tree = partition

        super().__init__(source_tree, name, filename)

    def parse_labels(self):
        pass


class OpenForceFieldTree(OpenForceFieldTreeBase):
    """
    A helper class that combines the various operations into a single object
    The operations here are vdW, bond, angle, proper, and improper labeling
    Also caches ff creation and other goodies that are shared between the
    5 operations
    """

    def __init__(self, source_tree, name, filename):
        super().__init__(source_tree, name, filename=filename)

        self._fields = ["_vdw", "_bonds", "_angles", "_outofplane", "_dihedral"]
        types = [
            OpenForceFieldvdWTree,
            OpenForceFieldBondTree,
            OpenForceFieldAngleTree,
            OpenForceFieldImproperTorsionTree,
            OpenForceFieldTorsionTree,
        ]

        args = [self.forcefield, source_tree]
        for field, obj in zip(self._fields, types):
            setattr(self, field, obj.from_forcefield(*args, field))

        # workaround since this Operation does not interface through
        # a partition (maybe make a TreeFunction that is both an operation
        # and a partition?
        source = self.source
        self.source = _DummyTree
        self.source.source = source

        # multiple processes is buggy, and worse, slower
        self.processes = 1

    def op(self, node, partition):
        pass

    def _apply_initialize(self, targets):

        root = self.source.source.root()
        all_labels = self.db.get(root.payload)

        if all_labels is None:
            self.all_labels = DEFAULT_DB({"data": DEFAULT_DB()})
            self.db[root.payload] = DEFAULT_DB({"data": DEFAULT_DB()})

    def _apply_finalize(self, targets):
        pass

    def _unpack_result(self, ret):

        if ret[0] not in self.db:
            self.db[ret[0]] = DEFAULT_DB({"data": DEFAULT_DB()})

        for k, v in ret[1].items():
            self.db[ret[0]]["data"][k] = DEFAULT_DB(v)

        root = self.source.source.root()

        if self.source.source.root().payload not in self.db:
            self.db[root.payload] = DEFAULT_DB()

        for k, v in ret[2].items():
            if k not in self.db[root.payload]["data"]:
                self.db[root.payload]["data"][k] = DEFAULT_DB()
            self.db[root.payload]["data"][k].update(v)

    def _generate_apply_kwargs(self, i, target, kwargs=None):
        """
        Generate the labels, keep in kwargs

        """

        if kwargs is None:
            kwargs = {}

        entry = self.source.source.db[target.payload]["data"]
        self.logger.debug("Pulled entry {}".format(target))
        self.logger.debug("Pulled entry data {}".format(entry))

        # we skip using 3D since labeling does not depend on coordinates
        smi = kwargs.get("smi")

        CIEHMS = "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        # if qcmol is None or smi is None:
        if smi is None:
            smi = entry.attributes[CIEHMS]
            kwargs["smi"] = smi
        else:
            o = "Molecule already present. Skipping creation"
            self.logger.debug(o)

        masks = kwargs.get("masks")
        if masks is None:
            masks = {}
            for field in self._fields:
                term = getattr(self, field)
                obj = term.source.db[target.payload]

                masks[term._key] = obj

            kwargs["masks"] = masks

        return kwargs

    def apply_single(self, i, target, **kwargs):

        out_str = ""
        all_labels = DEFAULT_DB()
        out_dict = DEFAULT_DB()

        smi = kwargs["smi"]

        mmol = Molecule.from_smiles(smi, allow_undefined_stereo=True)

        # just skip molecules that oFF can't handle for whatever reason

        try:
            top = mmol.to_topology()
        except AssertionError as e:
            out_str = ""
            out_str += "FAILED TO BUILD OFF MOL:\n"
            out_str += str(e)
            # pdb.set_trace()
            out_dict["error"] = out_str
            return out_dict

        labels = self.forcefield.label_molecules(top)[0]

        # calculations inside of a calculation....

        # if we have an error, then skip the underlying handlers
        if "error" in kwargs:
            return {
                target.payload: out_str,
                "debug": "",
                "return": [target.payload, dict(), dict()],
                "error": kwargs["error"],
            }

        subkwargs = kwargs.copy()
        subkwargs["labels"] = labels
        subkwargs["mmol"] = mmol

        masks = kwargs.get("masks")
        if masks is not None:
            for key in masks:
                obj = masks[key]

                m = flatten_list([v for v in obj["data"].values()], times=1)
                masks[key] = m

        subkwargs["masks"] = masks

        for field in self._fields:
            term = getattr(self, field)

            out_dict[term._key] = DEFAULT_DB()

            ret = term.apply_single(i, target, **subkwargs)


            # append the output string
            out_str += ret[target.payload]

            out_dict[term._key] = ret["return"][1][term._key]

            all_labels.update(ret["return"][2])

        return {
            target.payload: out_str,
            "debug": "",
            "return": [target.payload, out_dict, all_labels],
        }

    def to_pickle(self, db=True, name=None):

        tmp_src = self.source
        self.source = self.source.source
        nosave = [getattr(self, field) for field in self._fields]
        [setattr(self, field, None) for field in self._fields]
        super().to_pickle(db=db, name=name)
        for field, item in zip(self._fields, nosave):
            setattr(self, field, item)
        self.source = tmp_src

    def apply(self, targets=None):

        offsb.treedi.tree.LOG = True
        super().apply(targets=targets)

