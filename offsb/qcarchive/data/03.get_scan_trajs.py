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


def sort_mol_by_scan(nodes):
    ret = 0
    order = np.arange(len(nodes))
    vals = []
    for node in nodes:
        v = tuple([c.payload[2] for c in \
            QCA.node_iter_to_root(node, 
                select="Constraint")])
        if len(v) > 0:
            vals.append(v)
    if len(vals) == 0:
        ret = -1
        return ret, []
    vals = np.array(vals)
    try:
        order = np.lexsort(vals.T)
    except Exception:
        ret = -1
        return ret, []
    nodes_in_order = [nodes[i] for i in order]
    return ret, nodes_in_order

key = 'canonical_isomeric_explicit_hydrogen_smiles'
key_mapped = 'canonical_isomeric_explicit_hydrogen_mapped_smiles'
comment_str = "ENE: {:11.8e} {:s} SMILES: {:s}"

entries = list(QCA.iter_entry())
entries_n = len(entries)
spec = "openff-1.0.0"

tds = []
for i, entry_node in enumerate( entries, 1):
    spec = [QCA[x] for x in entry_node.children if QCA[x].payload == "QCS-openff-1.0.0"][0] 
    tds.extend([QCA[x] for x in spec.children])

QCA.cache_torsiondriverecord_minimum_molecules(tds)

for i, entry_node in enumerate( entries, 1):

    node = [QCA[x] for x in entry_node.children if QCA[x].payload == "QCS-openff-1.0.0"][0] 

    td = list(QCA.node_iter_depth_first(node, select="TorsionDrive"))[0]
    nodes = list( QCA.node_iter_optimization_minimum([td]))
    if len(nodes) == 0:
        continue
    constr = str(next(QCA.node_iter_depth_first([td], select="Constraint")).payload[1])
    ret, n = sort_mol_by_scan( nodes)
    if ret == 0:
        nodes = n
    print( i, "/", entries_n, td.payload, entry_node.payload, "Frames:", len(nodes))
    
    entry = QCA.db[entry_node.payload]['data']
    mol_name = entry.name
    smiles = entry.attributes[key]
    smiles_mapped = entry.attributes[key_mapped]

    rdmol = offsb.rdutil.mol.build_from_smiles(smiles_mapped)
    atom_map = offsb.rdutil.mol.atom_map(rdmol)
    mol_xyz_frames = []


    for mol_node in nodes:

        opt = next(QCA.node_iter_to_root(mol_node, select="Optimization"))
        
        if QCA.db.get( opt.payload).get( "data").get( "energies") is None:
            continue
        try:
            ene = QCA.db[opt.payload]["data"]["energies"][-1]
        except TypeError:
            continue

        comment = comment_str.format(ene, constr, smiles)
        qcmol = QCA.db[mol_node.payload]["data"]
        if "symbols" not in qcmol:
            continue
        frame = offsb.qcarchive.qcmol_to_xyz(qcmol, comment=comment)
        mol_xyz_frames.append(frame)

    with open( "td." + td.payload + ".scan_trj.xyz", 'w') as fd:
        [ fd.write( line) for lines in mol_xyz_frames for line in lines]


