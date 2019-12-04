import pickle
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import numpy as np

import offsb.tools.const as const
import simtk.unit as unit

#with open('oFF-1.1.0.p', 'rb') as fid:
#    print("Loading oFF data")
#    oFF10 = pickle.load(fid)
with open('QCA.p', 'rb') as fid:
    print("loading QCA data")
    QCA = pickle.load(fid)
if QCA.db is None:
    with open('QCA.db.p', 'rb') as fid:
        print("loading QCA db")
        QCA.db = pickle.load(fid).db


converters = { \
#               "Bonds" : const.bohr2angstrom, \
#               "Angles": 1.0, \
#               "ImproperTorsions": 1.0, \
#               "ProperTorsions": 1.0, \
#               "vdW": 1.0 \
               "MMEnergy": 1.0 \
}
data_chart = { \
#               "Bonds" : "bonds.p", \
#               "Angles": "angles.p", \
#               "ImproperTorsions": "outofplane.p", \
#               "ProperTorsions": "torsions.p" \
               "MMEnergy": "oMM.oFF-1.1.0.p" \
}

#label_db = oFF10.db.get( "ROOT").get( "data")

param_types = list(data_chart.keys())

# get the last structure from opts -> min energy 
mol_idx = -1

for param in param_types:
    data = None
    filename = data_chart.get( param)
    with open( filename, 'rb') as fid:
#        print("Loading", filename)
        data = pickle.load(fid)
    for entry in QCA.iter_entry():
        #labels = oFF10.db.get( entry.payload).get( "data").get( param)
        mol_name = QCA.db.get( entry.payload).get( "entry").name

#        for group in labels:
#            label = labels.get( group)
#            d = collections.defaultdict(list)
        for cons in QCA.node_iter_depth_first(entry, select="Constraint"):
            ang = eval(cons.payload)[0]
            for opt in QCA.node_iter_depth_first( cons, select="Optimization"):
                try:
                    ene = QCA.db.get( opt.payload).get( "data").get( "energies")[ mol_idx]
                except TypeError:
                    continue
                # assume we are taking only minimum energy
                grad = QCA.node_index.get( opt.children[ mol_idx])
                if grad == None:
                    #print("EMPTY GRAD")
                    continue
                mol = QCA.node_index.get( grad.children[0])
                if mol == None:
                    #print("EMPTY MOL")
                    continue
                syms = QCA.db.get( mol.payload).get( "data").get( "symbols")
                #vals = data.db.get( mol.payload).get( group) 
#                print( mol.payload)
#                print(data.db.get( mol.payload)) 
                if not (mol.payload in data.db):
                    #print("ABSENT. SKIPPING")
                    continue
                #print( QCA.db.get( entry.payload).get( "entry").td_keywords)
                vals = data.db.get( mol.payload).get( "data").get("energy")
                vals *= converters.get( param)
                out_str = "{:12s} {:16.10e} {:8.2f} {:4s} {:20s} {:10.4f} {:64s}"
                #print(out_str.format( entry.payload, ene, ang, label, \
                #      "-".join([str(syms[x-1])+str(x) for x in group]), vals[0], \
                #      mol_name ))
                out_str = "{:12s} {:16.10e} {:8.2f} {:10.4f} {:64s}"
                #print(vals.value_in_unit(vals.unit))
                print(out_str.format( entry.payload, ene, ang, \
                      vals.value_in_unit( vals.unit), \
                      mol_name ))

#                    # sigh, assume vals are scalars
#                    d[ang].append( [ene, vals[0]] )
