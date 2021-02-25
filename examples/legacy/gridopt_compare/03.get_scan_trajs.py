#!/usr/bin/env python3
import pickle
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import numpy as np
import re

import offsb.tools.const as const
import offsb.qcarchive
import offsb.rdutil
import simtk.unit as unit

with open('QCA.p', 'rb') as fid:
    QCA = pickle.load(fid)
if QCA.db is None:
    with open('QCA.db.p', 'rb') as fid:
        QCA.db = pickle.load(fid).db


def sort_mol_by_scan( nodes):
    ret = 0
    order = np.arange( len( nodes))
    vals = []
    for node in nodes:
        vals.append( tuple([ c.payload[2] for c in \
            QCA.node_iter_to_root( node, 
                select="Constraint")]))
    vals = np.array( vals)
    if len(vals) == 0:
        ret = -1
        return ret, []
    order = np.lexsort( vals.T)
    nodes_in_order = [nodes[i] for i in order]
    return ret, nodes_in_order

key = 'canonical_isomeric_explicit_hydrogen_smiles'
key_mapped = 'canonical_isomeric_explicit_hydrogen_mapped_smiles'
comment_str = "ENE: {:11.8e} SMILES: {:s}"

entries = list( QCA.iter_entry())
entries_n = len( entries)

for i, entry_node in enumerate( entries, 1):
    
    print( i, "/", entries_n, entry_node.payload)

    nodes = list( QCA.node_iter_optimization_minimum( entry_node))
    ret, nodes = sort_mol_by_scan( nodes)
    if ret:
        continue
    
    entry = QCA.db.get( entry_node.payload).get( "entry")
    mol_name = entry.name
    smiles = entry.attributes[ key]
    smiles_mapped = entry.attributes[ key_mapped]

    rdmol = offsb.rdutil.mol.build_from_smiles( smiles_mapped)
    atom_map = offsb.rdutil.mol.atom_map( rdmol)
    mol_xyz_frames = []


    for mol_node in nodes:

        opt = next(QCA.node_iter_to_root( mol_node, select="Optimization"))
        
        if QCA.db.get( opt.payload).get( "data").get( "energies") is None:
            continue
        try:
            ene = QCA.db.get( opt.payload).get( "data").get( "energies")[ -1]
        except TypeError:
            continue

        comment = comment_str.format(ene, smiles)
        qcmol = QCA.db.get( mol_node.payload).get( "data")
        frame = offsb.qcarchive.qcmol_to_xyz( qcmol, 
                atom_map=atom_map, comment=comment)
        mol_xyz_frames.append( frame)

    with open( entry_node.payload + ".scan_trj.xyz", 'w') as fd:
        [ fd.write( line) for lines in mol_xyz_frames for line in lines]


