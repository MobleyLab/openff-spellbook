
import treedi.tree as Tree
import treedi.node as Node
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import FragmentMatcher
import offsb.rdutil.mol

class SmilesSearchTree( Tree.PartitionTree):
    """ Just a quick way to get the indices and apply them to the entries """

    def __init__( self, smiles, source_tree, name):
        super().__init__( source_tree, name)
        self.smiles = smiles
        self.processes=1
    
    def apply( self, targets=None):
        if targets is None:
            targets = self.source.iter_entry()
        elif not hasattr(targets, "__iter__"):
            targets = [targets]
        # frag = self.smiles
        # p = FragmentMatcher.FragmentMatcher()
        # p = FragmentMatcher.FragmentMatcher()
        # p.Init(frag)
        matches = {}
        mol_hits = 0
        hits = 0
        qmol = Chem.MolFromSmarts(self.smiles)
        ind_map = {}
        for atom in qmol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num:
                ind_map[map_num - 1] = atom.GetIdx()

        map_list = [ind_map[x] for x in sorted(ind_map)]
        

        for target in targets:
            #print( "Entry", target)
            # if target.state == Node.CLEAN:
            #     continue
            obj = self.source.db[target.payload]['data'].dict()
            attrs = obj['attributes']
            smiles_pattern = attrs['canonical_isomeric_explicit_hydrogen_mapped_smiles']
            # mol = Chem.MolFromSmiles(smiles_pattern, sanitize=False)
            mol = offsb.rdutil.mol.build_from_smiles(smiles_pattern)
            #mol = Chem.AddHs(mol)
            #link_node = self.node_index.get( target.index)
            matches = []
            if True and mol.HasSubstructMatch(qmol):
                mol_hits += 1
                map_idx = {a.GetIdx() : a.GetAtomMapNum() for a in mol.GetAtoms()}
                for match in mol.GetSubstructMatches(qmol): # since oFF is a redundant set
                    # deg = mol.GetAtomWithIdx(match[0]).GetDegree()
                    #if(not (mol_smiles in matches)):
                    #    matches[mol_smiles] = []
                    mapped_match = [map_idx[i]-1 for i in match]
                    if len(map_list) > 0:
                        mapped_match = [mapped_match[x] for x in map_list]
                    matches.append( mapped_match)
                    hits += 1

            #elif(p.HasMatch(mol)):
            #    mol_hits += 1
            #    map_idx = {a.GetIdx() : a.GetAtomMapNum() for a in mol.GetAtoms()}
            #    for match in p.GetMatches(mol, uniquify=0): # since oFF is a redundant set
            #        # deg = mol.GetAtomWithIdx(match[0]).GetDegree()
            #        #if(not (mol_smiles in matches)):
            #        #    matches[mol_smiles] = []
            #        mapped_match = [map_idx[i]-1 for i in match]
            #        matches.append( mapped_match)
            #        hits += 1

            self.db.__setitem__( target.payload, { "data": matches })
            target.state = Node.CLEAN

        print("Found", mol_hits, " new molecules with", hits, "hits") 

