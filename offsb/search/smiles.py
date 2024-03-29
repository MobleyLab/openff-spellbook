#!/usr/bin/env python3

"""
"""

import tqdm

import offsb.rdutil.mol
import offsb.treedi.node as Node
import offsb.treedi.tree as Tree
from rdkit import Chem
import logging

from offsb.api.tk import ValenceDict, ImproperDict

DEFAULT_DB = Tree.DEFAULT_DB


def smarts_in_smiles(sma, smi) -> bool:

    if type(sma) is str:
        sma = [sma]

    mol = offsb.rdutil.mol.build_from_smiles(smi)
    return any([mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in sma])


class SmilesSearchTree(Tree.PartitionTree):
    """
    Just a quick way to get the indices and apply them to the entries
    """

    def __init__(self, smiles, source_tree, name, verbose=False):
        super().__init__(source_tree, name, verbose=verbose)
        self.smiles = smiles
        if hasattr(smiles, "__iter__") and (isinstance(smiles[0], str) and len(smiles[0]) == 1):
            self.smiles = [smiles]
        self.processes = 1
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
        self.valence = False

    def apply(self, targets=None):
        if targets is None:
            targets = self.source.iter_entry()
        elif not hasattr(targets, "__iter__"):
            targets = [targets]

        targets = list(targets)
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

        CIEHMS = "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        for target in tqdm.tqdm(
            targets,
            total=len(targets),
            ncols=80,
            desc="SMARTS Search",
            disable=not self.verbose,
        ):
            # print("Entry", target, target.payload, self[target.parent])
            # print("Specs", len(list(
            #   self.source.node_iter_depth_first(
            #           target, select="Specification"))))
            # if target.state == Node.CLEAN:
            #     continue
            obj = self.source.db[target.payload]["data"].dict()
            attrs = obj["attributes"]
            try:
                smiles_pattern = attrs[CIEHMS]
            except KeyError:
                breakpoint()
            # mol = Chem.MolFromSmiles(smiles_pattern, sanitize=False)
            mol = offsb.rdutil.mol.build_from_smiles(smiles_pattern)
            # mol = Chem.AddHs(mol)
            # link_node = self.node_index.get( target.index)
            results = DEFAULT_DB()
            match_kwargs={}
            if self.valence:
                match_kwargs.update(dict(uniquify=False, useChirality=True))
            # oid = id(results)
            # self.logger.debug("Creating SMI results db id {}".format(oid))
            # breakpoint()
            for smi, (qmol, map_list) in queries.items():
                matches = list()
                if mol.HasSubstructMatch(qmol):
                    mol_hits[smi] += 1
                    map_idx = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
                    map_inv = {v - 1: k for k, v in map_idx.items()}
                    for match in mol.GetSubstructMatches(qmol, **match_kwargs):
                        # deg = mol.GetAtomWithIdx(match[0]).GetDegree()
                        # if(not (mol_smiles in matches)):
                        #    matches[mol_smiles] = []
                        mapped_match = [map_idx[i] - 1 for i in match]
                        if len(map_list) > 0:
                            mapped_match = [mapped_match[x] for x in map_list]

                        mapped_match = tuple(map_inv[i] for i in mapped_match)
                        if self.valence:
                            fn = ValenceDict.key_transform

                            if "(" in smi:
                                fn = ImproperDict.key_transform

                            fn(mapped_match)

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

            assert target.payload not in self.db

            ret_db = DEFAULT_DB({"data": results})
            o, r = str(target), str(results)
            self.logger.debug("SMI SEARCH: {} {}\n".format(o, r))

            self.db.__setitem__(target.payload, ret_db)
            target.state = Node.CLEAN

        fmt_str = "Query {} : Found {:d} new molecules with {:d} hits\n"
        out_str = ""
        for smi in queries:
            out_str += fmt_str.format(smi, mol_hits[smi], hits[smi])
        self.logger.info(out_str)
