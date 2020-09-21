import os
import sys
import io
import pdb
import treedi
import treedi.tree
import simtk.unit
import openforcefield as oFF
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
import smirnoff99frosst as ff
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import FragmentMatcher
from ..tools import const
import offsb.rdutil.mol
from offsb.search.smiles import SmilesSearchTree
from treedi.tree import (TreeOperation, PartitionTree)
import abc
import logging
from pkg_resources import iter_entry_points
import contextlib
from offsb.tools.util import flatten_list

from treedi.tree import DEFAULT_DB

class OpenForceFieldTreeBase(treedi.tree.TreeOperation, abc.ABC):

    def __init__(self, source_tree, name, filename=None ):
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
            for entry_point in ['.'] + list(iter_entry_points(
                group="openforcefield.smirnoff_forcefield_directory"
            )):
                if type(entry_point) == str:
                    pth = entry_point
                else:
                    pth = entry_point.load()()[0]
                abspth = os.path.join(pth, filename)
                print("Searching", abspth)
                if os.path.exists(abspth):
                    self.abs_path = abspth
                    print("Found")
                    found = True
                    break
            if not found:
                raise Exception("Forcefield could not be found")
            self.forcefield = oFF.typing.engines.smirnoff.ForceField(
                self.abs_path, disable_version_check=True
            )
            logger.setLevel(level=level)

    @classmethod
    def from_forcefield(cls, forcefield, source_tree, name):
        obj = cls(source_tree, name)
        obj.forcefield = forcefield
        return obj

    def to_pickle_str(self):
        import pickle

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
            self.forcefield = oFF.typing.engines.smirnoff.ForceField(
                self.abs_path, disable_version_check=True
            )

    def count_oFF_labels(self, node):
        """
        provide a summary of the oFF labels found
        """
        return

    def _generate_apply_kwargs(self, i, target, kwargs={}):

        # labels = self.source.db[target.payload]["data"]
        entry = self.source.source.db[target.payload]["data"]

        out_str = ""

        mol = kwargs.get('mol')

        if mol is None:
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
            kwargs['mol'] = mol

        
        masks = kwargs.get('masks')
        if masks is None:
            obj = self.source.db[self.source[target.index].payload]
            masks = obj["data"]

            kwargs['masks'] = flatten_list([v for v in masks.values()], times=1)


        kwargs.update({"name": self.name, "entry": str(entry)})
        return kwargs

    def op(self, node, partition):
        pass

    def _apply_initialize(self, targets):

        all_labels = self.db.get(
            self.source.root().payload
        )

        if all_labels is None:
            self.all_labels = {"data": {}}
            self.db[self.source.root().payload] = {'data': {}}
    
    def _apply_finalize(self, targets):
        pass

    def _unpack_result(self, ret):

        self.db[ret[0]] = {"data" : ret[1] }

        if self.source.root().payload not in self.db:
            self.db[self.source.root().payload] = {}

        self.db[self.source.root().payload]['data'].update(ret[2])

    def apply_single(self, i, target, kwargs):

        out_str = ""
        all_labels = {}
        out_dict = {}

        labels = kwargs.get('labels')
        if labels is None:
            mol = kwargs['mol']
            # map_idx = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
            with io.StringIO() as f:
                with contextlib.redirect_stderr(f):
                    mmol = oFF.topology.Molecule.from_rdkit(mol,
                            allow_undefined_stereo=True)
                for line in f:
                    if 'not error because allow_undefined_stereo' not in line:
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
                    "error": out_str
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
        # for key in keys:
        uniq_members = self._unique_members
        key = self._key
            # print( key)

        params = labels.get(key)
        out_dict[key] = {}
        # print( "params", params)
        if key not in all_labels:
            all_labels.update({key: {}})

        mol = kwargs['mol']
        map_idx = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
        map_inv = {v-1:k for k,v in map_idx.items()}

        masks = kwargs['masks']
        if masks is None:
            masks_sorted = []
        else:
            if isinstance(masks, dict):
                masks = masks[key]
            masks_sorted = masks #[sorted(k) for k in masks]

        for mask in map(tuple, masks_sorted):
            
            # we are using the chemical indices; no map needed
            atoms = tuple(mask)
            # FF indices are unmapped (ordered by smiles appearance)
            # map them now since we work in the mapped CMILES in QCA
            val = params.get(atoms)

            # since params are Valence/ImproperDicts, sorting is correct
            # atoms = tuple([map_idx[i]-1 for i in params.get(mask)])

            # if any([x-1 < 0 for x in atoms]):
            #     print("Atoms are 0 based!")
            # if any([x-1 < 0 for y in masks_sorted for x in y]):
            #     print("masks are 0 based!")
            #     assert False

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
            # params[ mapped_atoms] = val.id

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
            OpenForceFieldTorsionTree
        ]

        args = [self.forcefield, source_tree]
        for field, obj in zip(self._fields, types):
            setattr(self, field, obj.from_forcefield(*args, field))
        
        class DummyTree:
            __slots__ = ["source"]


        # workaround since this Operation does not interface through
        # a partition (maybe make a TreeFunction that is both an operation
        # and a partition?
        source = self.source
        self.source = DummyTree
        self.source.source = source

    def op(self, node, partition):
        pass

    def _apply_initialize(self, targets):

        root = self.source.source.root() 
        all_labels = self.db.get(
            root.payload
        )

        if all_labels is None:
            self.all_labels = DEFAULT_DB({"data": DEFAULT_DB()})
            self.db[root.payload] = DEFAULT_DB({'data': DEFAULT_DB()})
    
    def _apply_finalize(self, targets):
        pass

    def _unpack_result(self, ret):

        if ret[0] not in self.db:
            self.db[ret[0]] = DEFAULT_DB({"data" : DEFAULT_DB() })

        for k,v in ret[1].items():
            self.db[ret[0]]['data'][k] = DEFAULT_DB(v)

        root = self.source.source.root()

        if self.source.source.root().payload not in self.db:
            self.db[root.payload] = DEFAULT_DB()

        for k,v in ret[2].items():
            if k not in self.db[root.payload]['data']:
                self.db[root.payload]['data'][k] = DEFAULT_DB()
            self.db[root.payload]['data'][k].update(v)

    def _generate_apply_kwargs(self, i, target, kwargs={}):
        """
        Generate the labels, keep in kwargs

        """

        entry = self.source.source.db[target.payload]["data"]
        self.logger.debug("Pulled entry {}".format(target) )
        self.logger.debug("Pulled entry data {}".format(entry) )

        out_str = ""

        mol = kwargs.get('mol')

        CIEHMS = "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        if mol is None:
            smi = entry.attributes[CIEHMS]
            self.logger.debug("Creating mol with CIEHMS " + smi)
            if "initial_molecule" in entry.dict():
                qcid = entry.dict()["initial_molecule"]
            elif "initial_molecules" in entry.dict():
                qcid = entry.dict()["initial_molecules"]
            else:
                qid = str(qcid)
                out_str += "{:d} initial mol was empty: {:s}".format(i, qid)
                return {"error": out_str}

            if isinstance(qcid, set):
                qcid = list(qcid)
            if isinstance(qcid, list):
                qcid = str(qcid[0])

            qcmolid = "QCM-" + qcid

            if qcmolid not in self.source.source.db:
                qid = str(qcmolid)
                out_str += "{:d} initial mol was empty: {:s}".format(i, qid)
                return {"error": out_str}

            if "data" in self.source.source.db.get(qcmolid):
                qcmol = self.source.source.db.get(qcmolid).get("data")
            else:
                qid = str(qcmolid)
                out_str += "{:d} initial mol was empty: {:s}".format(i, qid)
                return {"error": out_str}
            shape = str(qcmol['geometry'].shape)
            self.logger.debug("Molecule shape is {}".format(shape))
            mol = offsb.rdutil.mol.rdmol_from_smiles_and_qcmol(smi, qcmol)
            kwargs['mol'] = mol
        else:
            o = "Molecule already present. Skipping creation"
            self.logger.debug(o)

        
        masks = kwargs.get('masks')
        if masks is None:
            masks = {}
            for field in self._fields:
                term = getattr(self, field)
                obj = term.source.db[target.payload]

                m = flatten_list([v for v in obj["data"].values()], times=1)
                masks[term._key] = m

            kwargs['masks'] = masks

        # This calls Base, which collects the molecule
        # kwargs = super()._generate_apply_kwargs(i, target, kwargs)
        
        # The point of generating kwargs is to avoid pickling self
        # for parallel exe

        # Right now, it seems to make sense to generate the labels
        # since not doing so will cause a pickle of the FF which
        # is larger and more complex.

        mol = kwargs['mol']

        # FF and smiles searches are on chemical index, so no mapping needed
        # map_idx = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}


        lvl = logging.getLogger("openforcefield").getEffectiveLevel()
        logging.getLogger("openforcefield").setLevel(logging.ERROR)
        with io.StringIO() as f:
            with contextlib.redirect_stderr(sys.stdout):
                with contextlib.redirect_stdout(f):
                    mmol = oFF.topology.Molecule.from_rdkit(mol,
                            allow_undefined_stereo=True)
                for line in f:
                    if 'not error because allow_undefined_stereo' not in line:
                        print(line)
        logging.getLogger("openforcefield").setLevel(lvl)
        # mmol = oFF.topology.Molecule.from_rdkit(
        #     mol, allow_undefined_stereo=True)

        # just skip molecules that oFF can't handle for whatever reason
        try:
            top = oFF.topology.Topology.from_molecules(mmol)
        except AssertionError as e:
            out_str = ""
            out_str += "FAILED TO BUILD OFF MOL:\n"
            out_str += str(e)
            # pdb.set_trace()
            kwargs['error'] = out_str
            return kwargs

        labels = self.forcefield.label_molecules(top)[0]

        kwargs['labels'] = labels

        return kwargs

    def apply_single(self, i, target, kwargs):

        out_str = ""
        all_labels = DEFAULT_DB()
        out_dict = DEFAULT_DB()


        # calculations inside of a calculation....

        # if we have an error, then skip the underlying handlers
        if 'error' in kwargs:
            return {
                target.payload: out_str, "debug": "",
                "return": [target.payload, dict(), dict()],
                "error": kwargs['error']
            }

        for field in self._fields:
            term = getattr(self, field)
            self.logger.debug("Parameter {} begin".format(type(term)))
            out_dict[term._key] = DEFAULT_DB()
            ret = term.apply_single(i, target, kwargs)
            
            # maybe skipping this line will allow us to keep only one copy
            #term._unpack_result(ret['return'])

            # append the output string
            out_str += ret[target.payload]

            # Just keep all data generated. It will be stored in this obj
            # and the underlying handler obj
            out_dict[term._key] = ret['return'][1][term._key]

            all_labels.update(ret['return'][2])

        return {
            target.payload: out_str, "debug": "",
            "return": [target.payload, out_dict, all_labels],
        }


    def to_pickle(self, db=True, name=None):
        # tmp = self.forcefield
        tmp_src = self.source
        self.source = self.source.source
        nosave = [getattr(self, field) for field in self._fields]
        [setattr(self,field,None) for field in self._fields]
        super().to_pickle(db=db, name=name)
        for field,item in zip(self._fields, nosave):
            setattr(self, field, item)
        self.source = tmp_src
        # self.forcefield = tmp

    def apply(self, targets=None):

        treedi.tree.LOG = True
        # # workaround; populate targets so downstream doesn't try
        # # to access self.source.source
        # root = self.source.root()
        # targets = list(
        #     self.source.node_iter_depth_first(root, select=self._select)
        # )
        super().apply(targets=targets)


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

        # shared_members = ["smirks", "id"]
        # # for key in keys:
        
        # key = self._key
        #     # print( key)

        # params = labels.get(key)
        # out_dict[key] = {}
        # # print( "params", params)
        # if key not in all_labels:
        #     all_labels.update({key: {}})
        # for atoms in params:
        #     val = params.get(atoms)
        #     mapped_atoms = tuple([map_idx[i] for i in atoms])
        #     # params[ mapped_atoms] = val.id
        #     out_dict[key][mapped_atoms] = val.id

        #     ret = {}
        #     for name in shared_members + uniq_members[key]:
        #         prop = getattr(val, name)
        #         ret[name] = prop

        #     if val.id not in all_labels[key]:
        #         all_labels[key][val.id] = ret

        # return {
        #     target.payload: out_str,
        #     "return": [target.payload, out_dict, all_labels],
        # }

    # def apply(self, targets=None):
    #     super().apply(self._select, targets=targets)

        # for field in self._fields:
        #     term = getattr(self, field)
        #     term.labels
        #     term.apply(targets=targets)


        # if targets is None:
        #     targets = list(self.source.iter_entry())
        # elif not hasattr( targets, "__iter__"):
        #     targets = [targets]
# # expand if a generator # targets = list(targets)

        # n_targets = len(targets)
        # all_labels = self.db.get(
        #     self.source.root().payload
        # )
        # if all_labels is None:
        #     all_labels = {"data": {}}
        #     self.db[self.source.root().payload] = {'data': {}}

        # if self.processes > 1:
        #     import concurrent.futures
        #     exe = concurrent.futures.ProcessPoolExecutor(
        #         max_workers=self.processes)

        #     work = [ exe.submit(
        #         __class__.apply_single, self, n, target )
        #         for n, target in enumerate(targets, 1) ]
        #     for n,future in enumerate(concurrent.futures.as_completed(work), 1):
        #         if future.done:
        #             try:
        #                 val = future.result()
        #             except RuntimeError:
        #                 print("RUNTIME ERROR; race condition??")
        #         if val is None:
        #             print("data is None?!?")
        #             continue

        #         for tgt, ret in val.items():
        #             if tgt == "return":
        #                 self.db[ret[0]] = {"data" : ret[1] }

        #                 if self.source.root().payload not in self.db:
        #                     self.db[self.source.root().payload] = {}

        #                 self.db[self.source.root().payload]['data'].update(ret[2])
        #             else:
        #                 print( n,"/", n_targets, tgt)
        #                 for line in ret:
        #                     print(line, end="")

        #     exe.shutdown()

        # elif self.processes == 1:
        #     for n, target in enumerate( targets, 1):
        #         val = self.apply_single( n, target)
        #         for tgt, ret in val.items():
        #             if tgt == "return":
        #                 self.db[ret[0]] = {"data" : ret[1] }

        #                 if self.source.root().payload not in self.db:
        #                     self.db[self.source.root().payload] = {}

        #                 self.db[self.source.root().payload]['data'].update(ret[2])
        #             else:
        #                 print( n,"/", n_targets, tgt)
        #                 for line in ret:
        #                     print(line, end="")

        # for n, target in enumerate(targets, 1):
        #    print( n,"/",n_targets, target)
        #    #if n == 2:
        #    #    break
        #    entry = self.source.db.get(target.payload)['data'].dict()
        #    attrs = entry['attributes']

        #    if 'initial_molecule' in entry:
        #        qcid = entry['initial_molecule']
        #    else:
        #        qcid = entry['initial_molecules']

        #    if isinstance(qcid, set):
        #        qcid = list(qcid)
        #    if isinstance(qcid, list):
        #        qcid = str(qcid[0])

        #    qcmolid = 'QCM-' + qcid

        #    if qcmolid not in self.source.db:
        #        continue

        #    qcmol = self.source.db.get( qcmolid).get( "data")
        #    #print("INITIAL MOL:", qcmol.get( "geometry").shape )
        #    smiles_pattern = attrs.get( 'canonical_isomeric_explicit_hydrogen_mapped_smiles')
        #    #print( smiles_pattern)
        #    try:
        #        mol = self._generate_mol(smiles_pattern, qcmol)
        #        map_idx = { a.GetIdx() : a.GetAtomMapNum() for a in mol.GetAtoms()}
        #    except Exception as msg:
        #        print("Error:", msg)
        #        continue

        #    mmol = oFF.topology.Molecule.from_rdkit( mol, allow_undefined_stereo=True)

        #    # just skip molecules that oFF can't handle for whatever reason
        #    try:
        #        top = oFF.topology.Topology.from_molecules(mmol)
        #    except AssertionError as e:
        #        print("FAILED TO BUILD OFF MOL:")
        #        print(e)
        #        pdb.set_trace()
        #        continue
        #    labels = self.forcefield.label_molecules( top)[0]
        #    #labels = labels[0]
        #    #print( labels)
        #    #print( type(labels))
        #    mapped_labels = {}

        #    keys = ['Bonds', 'Angles', 'ProperTorsions', 'vdW', 'ImproperTorsions', 'Electrostatics', 'ToolkitAM1BCC']
        #    #keys = ["Bonds", "Angles", "ProperTorsions", "ImproperTorsions", "vdW"]
        #    simple = {}
        #    shared_members = [ "smirks", "id" ]
        #    uniq_members = { "vdW": ["rmin_half", "epsilon"],
        #            "Bonds": ["k", "length"],
        #            "Angles": ["k", "angle"],
        #            "ImproperTorsions": ["k", "periodicity", "phase"],
        #            "ProperTorsions": ["k", "periodicity", "phase"] }
        #    for key in keys:
        #        #print( key)
        #        params = labels.get( key)
        #        simple[ key] = {}
        #        #print( "params", params)
        #        if key not in all_labels.get( "data"):
        #            all_labels.get( "data").update( {key : {} })
        #        for atoms in params:
        #            val = params.get( atoms)
        #            mapped_atoms = tuple([map_idx[i] for i in atoms])
        #            #params[ mapped_atoms] = val.id
        #            simple[ key][ mapped_atoms] = val.id

        #            ret = {}
        #            for name in shared_members + uniq_members[ key] :
        #                prop = getattr( val, name)
        #                if isinstance( prop, list) and len( prop) > 0:
        #                    # upcast
        #                    pass
        #                    #prop = list( prop)
        #                    #if isinstance( i[0], simtk.unit.Quantity):
        #                    #    i = [(float(j / j.unit), str(j.unit)) for j in i]
        #                #elif isinstance( i, simtk.unit.Quantity):
        #                #    i = (float(i / i.unit), str(i.unit))
        #                ret[ name] = prop

        #            if val.id not in all_labels.get( "data")[ key]:
        #                all_labels.get( "data")[ key][ val.id ] = ret
        #            #print( val.__dict__)
        #    #print( "simple", simple)

        #    #for match in p.GetMatches(mol, uniquify=0): # since oFF is a redundant set
        #    #    # deg = mol.GetAtomWithIdx(match[0]).GetDegree()
        #    #    #if(not (mol_smiles in matches)):
        #    #    #    matches[mol_smiles] = []
        #    #    mapped_match = [map_idx[i] for i in match]
        #    #    matches.append( mapped_match)
        #    #    hits += 1
        #    self.db.__setitem__( target.payload, {"data": simple })
        #    #link_node.state = CLEAN
        # self.db.update( { self.node_index.get( self.source.root_index).payload : all_labels } )
        # pp = pprint.PrettyPrinter( indent=4)
        # pp.pprint( full_node.payload)
        # print("Found", mol_hits, "molecules with", hits, "hits")
