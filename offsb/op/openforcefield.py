
import os
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

class OpenForceFieldTree( treedi.tree.PartitionTree):
    """ Just a quick way to get the indices and apply them to the entries """

    def __init__( self, filename, source_tree, name):
        super().__init__( source_tree, name)
        import logging
        logger = logging.getLogger()
        level = logger.getEffectiveLevel()
        logger.setLevel(level=logging.ERROR)
        self.filename = filename
        from pkg_resources import iter_entry_points
        for entry_point in iter_entry_points(group='openforcefield.smirnoff_forcefield_directory'):
            pth = entry_point.load()()[0]
            abspth = os.path.join(pth, filename)
            print("Searching", abspth)
            if os.path.exists( abspth):
                self.abs_path = abspth
                print("Found")
                break
            raise Exception("Forcefield could not be found")
        self.forcefield= oFF.typing.engines.smirnoff.ForceField(
            self.abs_path, disable_version_check=True)
        logger.setLevel(level=level)
        print("My db is", self.db)

    def to_pickle_str(self):
        import pickle
        tmp = self.forcefield
        self.forcefield = None
        obj = super().to_pickle_str()
        self.forcefield = tmp
        return obj

    def to_pickle(self, db=True, name=None):
        #tmp = self.forcefield
        super().to_pickle( db=db, name=name)
        #self.forcefield = tmp

    def isolate(self):
        super().isolate()
        self.forcefield = None

    def associate(self, source):
        super().associate( source)
        if self.forcefield is None:
            self.forcefield = oFF.typing.engines.smirnoff.ForceField(
                self.abs_path, disable_version_check=True)

    def count_oFF_labels(self, node):
        """
        provide a summary of the oFF labels found
        """
        return
    
    
    def _generate_mol(self, smiles_pattern, qcmol):
        
        mol = Chem.MolFromSmiles(smiles_pattern, sanitize=False)
        #print([m.GetSymbol() for m in mol.GetAtoms()])
        flags = Chem.SanitizeFlags.SANITIZE_ALL \
                ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS \
                ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY

        Chem.SanitizeMol( mol, flags)
        Chem.SetAromaticity( mol, Chem.AromaticityModel.AROMATICITY_MDL)
        Chem.SanitizeMol( mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)

        id = offsb.rdutil.mol.embed_qcmol_3d(mol, qcmol)
        if id < 0:
            raise Exception()

        Chem.rdmolops.AssignStereochemistryFrom3D( mol, id, replaceExistingTags=True)
        Chem.rdmolops.AssignAtomChiralTagsFromStructure( mol, id, replaceExistingTags=True)

        return mol

    def apply_single(self, i, target):

        out_str = ""
        all_labels = {}
        out_dict = {}

        entry = self.source.db.get(target.payload)['data'].dict()
        attrs = entry['attributes']

        if 'initial_molecule' in entry:
            qcid = entry['initial_molecule']
        else:
            qcid = entry['initial_molecules']

        if isinstance(qcid, set):
            qcid = list(qcid)
        if isinstance(qcid, list):
            qcid = str(qcid[0])

        qcmolid = 'QCM-' + qcid
        
        if qcmolid not in self.source.db:
            return { target.payload: out_str, "return": [target.payload, out_dict, all_labels]}
            

        qcmol = self.source.db.get( qcmolid).get( "data")
        #print("INITIAL MOL:", qcmol.get( "geometry").shape )
        smiles_pattern = attrs.get( 'canonical_isomeric_explicit_hydrogen_mapped_smiles')
        #print( smiles_pattern)
        try:
            mol = self._generate_mol(smiles_pattern, qcmol)
            map_idx = { a.GetIdx() : a.GetAtomMapNum() for a in mol.GetAtoms()}
        except Exception as msg:
            out_str += "Error: {:s}\n".format(msg)
            return { target.payload: out_str, "return": [target.payload, out_dict, all_labels]}

        mmol = oFF.topology.Molecule.from_rdkit( mol, allow_undefined_stereo=True)

        # just skip molecules that oFF can't handle for whatever reason
        try:
            top = oFF.topology.Topology.from_molecules(mmol)
        except AssertionError as e:
            out_str += "FAILED TO BUILD OFF MOL:\n"
            out_str += str(e)
            # pdb.set_trace()
            return { target.payload: out_str, "return": [target.payload, out_dict, all_labels]}

        labels = self.forcefield.label_molecules( top)[0]

        mapped_labels = {}

        keys = ['Bonds', 'Angles', 'ProperTorsions', 'vdW', 'ImproperTorsions', 'Electrostatics', 'ToolkitAM1BCC']
        #keys = ["Bonds", "Angles", "ProperTorsions", "ImproperTorsions", "vdW"]

        shared_members = [ "smirks", "id" ]
        uniq_members = { "vdW": ["rmin_half", "epsilon"],
                "Bonds": ["k", "length"],
                "Angles": ["k", "angle"],
                "ImproperTorsions": ["k", "periodicity", "phase"],
                "ProperTorsions": ["k", "periodicity", "phase"] }

        for key in keys:
            #print( key)
            params = labels.get( key)
            out_dict[ key] = {}
            #print( "params", params)
            if key not in all_labels:
                all_labels.update( {key : {} })
            for atoms in params:
                val = params.get( atoms)
                mapped_atoms = tuple([map_idx[i] for i in atoms])
                #params[ mapped_atoms] = val.id
                out_dict[ key][ mapped_atoms] = val.id

                ret = {}
                for name in shared_members + uniq_members[ key] :
                    prop = getattr(val, name)
                    ret[name] = prop

                if val.id not in all_labels[ key]:
                    all_labels[ key][ val.id ] = ret
        
        return { target.payload: out_str, "return": [target.payload, out_dict, all_labels]}

    def apply(self, targets=None):
        if targets is None:
            targets = list(self.source.iter_entry())
        elif not hasattr( targets, "__iter__"):
            targets = [targets]

        # expand if a generator
        targets = list(targets)

        n_targets = len(targets)
        all_labels = self.db.get(
            self.source.root().payload
        )
        if all_labels is None:
            all_labels = {"data": {}}
            self.db[self.source.root().payload] = {'data': {}}

        if self.processes > 1:
            import concurrent.futures
            exe = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.processes)

            work = [ exe.submit( 
                __class__.apply_single, self, n, target )
                for n, target in enumerate(targets, 1) ]
            for n,future in enumerate(concurrent.futures.as_completed(work), 1):
                if future.done:
                    try:
                        val = future.result()
                    except RuntimeError:
                        print("RUNTIME ERROR; race condition??")
                if val is None:
                    print("data is None?!?")
                    continue

                for tgt, ret in val.items():
                    if tgt == "return":
                        self.db[ret[0]] = {"data" : ret[1] }

                        if self.source.root().payload not in self.db:
                            self.db[self.source.root().payload] = {}

                        self.db[self.source.root().payload]['data'].update(ret[2])
                    else:
                        print( n,"/", n_targets, tgt)
                        for line in ret:
                            print(line, end="")

            exe.shutdown()

        elif self.processes == 1:
            for n, target in enumerate( targets, 1):
                val = self.apply_single( n, target)
                for tgt, ret in val.items():
                    if tgt == "return":
                        self.db[ret[0]] = {"data" : ret[1] }

                        if self.source.root().payload not in self.db:
                            self.db[self.source.root().payload] = {}

                        self.db[self.source.root().payload]['data'].update(ret[2])
                    else:
                        print( n,"/", n_targets, tgt)
                        for line in ret:
                            print(line, end="")



            #for n, target in enumerate(targets, 1):
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
            #self.db.update( { self.node_index.get( self.source.root_index).payload : all_labels } )
            #pp = pprint.PrettyPrinter( indent=4)
            #pp.pprint( full_node.payload)
            #print("Found", mol_hits, "molecules with", hits, "hits") 

