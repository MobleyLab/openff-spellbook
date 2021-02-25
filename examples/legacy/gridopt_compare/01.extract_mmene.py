#!/usr/bin/env python3
import pickle
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import numpy as np
import re
import os

import offsb.tools.const as const
import simtk.unit as unit

PREFIX="."

#with open('oFF-1.1.0.p', 'rb') as fid:
#    print("Loading oFF data")
#    oFF10 = pickle.load(fid)
with open(os.path.join(PREFIX,'QCA.p'), 'rb') as fid:
    QCA = pickle.load(fid)
if QCA.db is None:
    with open(os.path.join(PREFIX, 'QCA.db.p'), 'rb') as fid:
        QCA.db = pickle.load(fid).db

def tuple_to_hyphenated( x):
    if isinstance( x, tuple):
        return re.sub("[() ]", "", str(x).replace(",","-"))
    else:
        return x


converters = { \
#               "Bonds" : const.bohr2angstrom, \
#               "Angles": 1.0, \
#               "ImproperTorsions": 1.0, \
#               "ProperTorsions": 1.0, \
#               "vdW": 1.0 \
               "MMEnergyA": 1.0, \
               "MMEnergyB": 1.0 \
}
data_chart = { \
#               "Bonds" : "bonds.p", \
#               "Angles": "angles.p", \
#               "ImproperTorsions": "outofplane.p", \
#               "ProperTorsions": "torsions.p" \
               "MMEnergyA": "oMM.oFF-Parsley-uncons.p" \
#               "MMEnergyB": "oMM.oFF-Parsley-uncons"  \
}

#label_db = oFF10.db.get( "ROOT").get( "data")

param_types = list(data_chart.keys())

# get the last structure from opts -> min energy 
mol_idx = -1

# hardcode this to only handle 1D scans for now

for param in param_types:
        data = None

        num_angs = 1 #len(ang_pl)
        out_str = "{:12s} {:12s} {:16.10e} "+"{:12s} {:12s} {:8.4f}"*num_angs +" {:10.4f} {:64s}\n"
        filename = data_chart.get( param)
        with open( os.path.join( PREFIX, filename), 'rb') as fid:
    #        print("Loading", filename)
            data = pickle.load(fid)
        key = 'canonical_isomeric_explicit_hydrogen_mapped_smiles'
        i = 0
        #breakpoint()
        for entry_node in QCA.iter_entry():
            print( i, entry_node)
            i += 1
            #labels = oFF10.db.get( entry.payload).get( "data").get( param)
            entry = QCA.db.get( entry_node.payload).get( "entry")
            mol_name = entry.name
            smiles_indexed = entry.attributes[ key]
        #if entryA is None or entryB is None:
        #    return False

    #        for group in labels:
    #            label = labels.get( group)
    #            d = collections.defaultdict(list)
            nodes = list(QCA.node_iter_optimization_minimum( entry_node, select="Molecule"))
            order = np.arange( len( nodes))
            vals = []
            for node in nodes:
                vals.append( tuple([ c.payload[2] for c in \
                    QCA.node_iter_to_root( node, 
                        select="Constraint")]))
            vals = np.array( vals)
            if len(vals) == 0:
                continue
            order = np.lexsort( vals.T)
            nodes_in_order = [nodes[i] for i in order]
            fnm = entry_node.payload + "." + param + ".dat"
            fd = open( fnm, 'w')
            spec_written = False
            ds = list(QCA.node_iter_to_root( entry_node))[1]
            for mol_node in nodes_in_order:
    #            for opt in QCA.node_iter_depth_first( cons, select="Optimization"):
                opt = next(QCA.node_iter_to_root( mol_node, select="Optimization"))
                if not spec_written:
                    qc_spec = QCA.db.get( opt.payload).get( "data").get( "qc_spec")
                    method = str( qc_spec.method)
                    basis  = str( qc_spec.basis)
                    fd.write( ds.name + "\n" + method + "\n" + basis + "\n")

                    header = "{:12s} {:12s} {:16s} "+"{:12s} {:12s} {:8s}"*num_angs +" {:10s} {:64s}\n"
                    fd.write(header.format( "# QCAProc", " QCAMol", " QMEne", " ScanType", " Atoms", " ScanVal", \
                          param, \
                      " SmilesMapped" ))
                    spec_written = True
                if QCA.db.get( opt.payload).get( "data").get( "energies") is None:
                    fd.close()
                    continue
                try:
                    ene = QCA.db.get( opt.payload).get( "data").get( "energies")[ mol_idx]
                except TypeError:
                    fd.close()
                    continue

                mol_id = mol_node.payload
                # assume we are taking only minimum energy
                syms = QCA.db.get( mol_node.payload).get( "data").get( "symbols")

                try:
                    vals = data.db.get( mol_node.payload).get( "data").get("energy")
                except Exception:
                    fd.close()
                    continue
                vals *= converters.get( param)


                ang_pl = [c.payload for c in QCA.node_iter_to_root( mol_node, select="Constraint")][0]
                if num_angs > 1:
                    ang = [ tuple_to_hyphenated( x) for y in ang_pl for x in y]
                else:
                    ang = [ tuple_to_hyphenated( x) for x in ang_pl]

                fd.write(out_str.format( entry_node.payload, mol_id, ene, *ang, \
                      vals.value_in_unit( vals.unit), \
                      smiles_indexed ))
            fd.close()

#                    # sigh, assume vals are scalars
#                    d[ang].append( [ene, vals[0]] )
