#!/usr/bin/env python3

"""
"""

import treedi.tree as Tree
import treedi.node as Node
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import FragmentMatcher
import offsb.rdutil.mol
import logging

DEFAULT_DB = Tree.DEFAULT_DB

class SmilesSearchTree(Tree.PartitionTree):
    """
    Just a quick way to get the indices and apply them to the entries
    """


    def __init__(self, smiles, source_tree, name):
        super().__init__(source_tree, name)
        self.smiles = smiles
        if hasattr(smiles, "__iter__") and isinstance(smiles[0], str):
            self.smiles = [smiles]
        self.processes = 1

    def apply(self, targets=None):
        if targets is None:
            targets = self.source.iter_entry()
        elif not hasattr(targets, "__iter__"):
            targets = [targets]
        # frag = self.smiles
        # p = FragmentMatcher.FragmentMatcher()
        # p = FragmentMatcher.FragmentMatcher()
        # p.Init(frag)
        mol_hits = {}
        hits = {}
        queries = {}
        for smi in self.smiles:
            qmol = Chem.MolFromSmarts(smi)
            ind_map = {}
            for atom in qmol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num:
                    ind_map[map_num - 1] = atom.GetIdx()

            map_list = [ind_map[x] for x in sorted(ind_map)]
            queries[smi] = (qmol, map_list)
            hits[smi] = 0
            mol_hits[smi] = 0

        CIEHMS = 'canonical_isomeric_explicit_hydrogen_mapped_smiles'
        for target in targets:
            # print("Entry", target, target.payload, self[target.parent])
            # print("Specs", len(list(
            #   self.source.node_iter_depth_first(
            #           target, select="Specification"))))
            # if target.state == Node.CLEAN:
            #     continue
            obj = self.source.db[target.payload]['data'].dict()
            attrs = obj['attributes']
            smiles_pattern = attrs[CIEHMS]
            # mol = Chem.MolFromSmiles(smiles_pattern, sanitize=False)
            mol = offsb.rdutil.mol.build_from_smiles(smiles_pattern)
            # mol = Chem.AddHs(mol)
            # link_node = self.node_index.get( target.index)
            results = DEFAULT_DB()
            # oid = id(results)
            # self.logger.debug("Creating SMI results db id {}".format(oid))
            # breakpoint()
            for smi, (qmol, map_list) in queries.items():
                matches = list()
                if mol.HasSubstructMatch(qmol):
                    mol_hits[smi] += 1
                    map_idx = {a.GetIdx(): a.GetAtomMapNum()
                               for a in mol.GetAtoms()}
                    map_inv = {v-1:k for k,v in map_idx.items()}
                    for match in mol.GetSubstructMatches(qmol):
                        # deg = mol.GetAtomWithIdx(match[0]).GetDegree()
                        # if(not (mol_smiles in matches)):
                        #    matches[mol_smiles] = []
                        mapped_match = [map_idx[i] - 1 for i in match]
                        if len(map_list) > 0:
                            mapped_match = [mapped_match[x] for x in map_list]

                        mapped_match = tuple(map_inv[i] for i in mapped_match)
                        matches.append(mapped_match)
                        hits[smi] += 1
                results[smi] = matches

            # elif(p.HasMatch(mol)):
            #     mol_hits += 1
            #     map_idx = {a.GetIdx() : a.GetAtomMapNum() for a in mol.GetAtoms()}
            #    for match in p.GetMatches(mol, uniquify=0): # since oFF is a
            #    redundant set
            #        # deg = mol.GetAtomWithIdx(match[0]).GetDegree()
            #        #if(not (mol_smiles in matches)):
            #        #    matches[mol_smiles] = []
            #        mapped_match = [map_idx[i]-1 for i in match]
            #        matches.append( mapped_match)
            #        hits += 1


            assert  target.payload not in self.db


            ret_db = DEFAULT_DB({"data": results})
            o, r = str(target),str(results)
            self.logger.debug("SMI SEARCH: {} {}\n".format(o, r)) 

            self.db.__setitem__(target.payload, ret_db)
            target.state = Node.CLEAN

        fmt_str = "Query {} : Found {:d} new molecules with {:d} hits\n"
        out_str = ""
        for smi in queries:
            out_str += fmt_str.format(smi, mol_hits[smi], hits[smi])
        self.logger.info(out_str) 

