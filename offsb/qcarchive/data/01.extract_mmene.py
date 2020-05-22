#!/usr/bin/env python3

import pickle
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import numpy as np
import re
import os

from offsb.op.geometry import ( \
        BondOperation, 
        AngleOperation,
        ImproperTorsionOperation,
        TorsionOperation)

import offsb.tools.const as const
import simtk.unit as unit
import offsb.qcarchive as qca

PREFIX="."

#}}}
#{{{
#with open('oFF-1.1.0.p', 'rb') as fid:
#    print("Loading oFF data")
#    oFF10 = pickle.load(fid)
with open(os.path.join(PREFIX,'QCA.p'), 'rb') as fid:
    QCA = pickle.load(fid)
if QCA.db is None:
    with open(os.path.join(PREFIX, 'QCA.db.p'), 'rb') as fid:
        QCA.db = pickle.load(fid).db
#}}}
#{{{

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
#               "MMEnergyB": 1.0 \
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

#}}}
#{{{
# hardcode this to only handle 1D scans for now

if 0:
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

            for entry_node in QCA.iter_entry():
                print( i, entry_node)
                i += 1
                #labels = oFF10.db.get( entry.payload).get( "data").get( param)
                entry = QCA.db[entry_node.payload]["entry"]
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
                    val = tuple([ c.payload[2] for c in \
                        QCA.node_iter_to_root( node, 
                            select="Constraint")])
                    if len(val) > 0:
                        vals.append(val)
                if len(vals) == 0:
                    nodes_in_order = nodes
                else:
                    vals = np.array( vals)
                    order = np.lexsort( vals.T)
                    nodes_in_order = [nodes[i] for i in order]
                fnm = entry_node.payload + "." + param + ".dat"
                fd = open( fnm, 'w')
                spec_written = False
                ds = list(QCA.node_iter_to_root( entry_node))[-2]
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
#}}}
#{{{

for n in QCA.iter_entry():
    constraints = set([tuple(c.payload[1]) for c in  QCA.node_iter_depth_first( n, select="Constraint")])
    print(n)
    print("    ", constraints)
#}}}
#{{{


def extract_torsions_for_same_molecule(QCA, param_types, data_chart, converters):
    for param in param_types:
            data = None

            filename = data_chart[param]
            with open( os.path.join( PREFIX, filename), 'rb') as fid:
        #        print("Loading", filename)
                data = pickle.load(fid)

            key = 'canonical_isomeric_explicit_hydrogen_mapped_smiles'
            for folder in QCA.combine_by_entry():

                #check if there are torsions, if not, skip (for now)
                has_torsion = False
                for n in folder.children:
                    if QCA[n].name == "TorsionDrive":
                        has_torsion = True
                        break
                if not has_torsion:
                    continue
                print(folder)
                    

                for n in folder.children:
                    constraints = set([c.payload[:2] for c in
                        QCA.node_iter_depth_first(QCA[n], select="Constraint")])
                num_angs = len(constraints)
                out_str = "{:12s} {:12s} {:16.10e} "+"{:12s} {:12s} {:8.4f}"*num_angs +" {:10.4f} {:64s}\n"
                
                i = 0
                for entry_id in folder.children:
                    entry_node = QCA[entry_id]
                    print( "  ",i, entry_node)
                    i += 1

                    #labels = oFF10.db.get( entry.payload).get( "data").get( param)

                    entry = QCA.db[entry_node.payload]["entry"]
                    mol_name = entry.name
                    smiles_indexed = entry.attributes[ key]

                    # choose between all mins or just min along a TD
                    # Note that not sure how TD iter will work when entry 
                    # is an optimization
                    #nodes = list(QCA.node_iter_optimization_minimum(entry_node, select="Molecule"))
                    nodes = list(QCA.node_iter_torsiondriverecord_minimum(entry_node, select="Molecule"))

                    order = np.arange( len( nodes))
                    cons = []
                    for node in nodes:
                        # for kk in QCA.node_iter_to_root( QCA[node.index]):
                        #     print(kk)
                        #     print("    ", kk.parent, kk.children)
                        val = tuple([ c.payload[2] for c in \
                            QCA.node_iter_to_root( node, 
                                select="Constraint")])
                        if len(val) > 0:
                            cons.append(val)

                    if len(cons) == 0:
                        nodes_in_order = nodes
                        order = np.arange(len(nodes))
                    else:
                        cons = np.array( cons)
                        order = np.lexsort( cons.T)
                        nodes_in_order = [nodes[i] for i in order]
                    cons_set = cons
                    fnm = entry_node.payload + "." + param + ".dat"
                    fd = open( fnm, 'w')
                    spec_written = False

                    # Assumes -1 is root, so -2 are the datasets
                    ds = list(QCA.node_iter_to_root( entry_node))[-2]
                    mols = []
                    for m_idx, mol_node in enumerate(nodes_in_order):
            #            for opt in QCA.node_iter_depth_first( cons, select="Optimization"):
                        opt = next(QCA.node_iter_to_root( mol_node, select="Optimization"))
                        if not spec_written:
                            qc_spec = QCA.db.get( opt.payload).get( "data").get( "qc_spec")
                            method = str( qc_spec.method)
                            basis  = str( qc_spec.basis)
                            fd.write( ds.name + "\n" + method + "\n" + basis + "\n")

                            header = "{:12s} {:12s} {:16s} "+"{:12s} {:12s} {:8s}"*num_angs +" {:10s} {:64s}\n"
                            fd.write(header.format( "# QCAProc", " QCAMol", " QMEne", 
                                *[" ScanType", " Atoms", " ScanVal"] * num_angs, \
                                  param, " SmilesMapped" ))
                            spec_written = True
                        if QCA.db.get( opt.payload).get( "data").get( "energies") is None:
                            #fd.close()
                            print("No energies")
                            continue
                        try:
                            ene = QCA.db.get( opt.payload).get( "data").get( "energies")[ mol_idx]
                        except TypeError:
                            print("No energies for this mol", mol_idx)
                            #fd.close()
                            continue

                        mol_id = mol_node.payload
                        # assume we are taking only minimum energy
                        syms = QCA.db.get( mol_node.payload).get( "data").get( "symbols")
                        qc_mol = QCA.db.get( mol_node.payload).get( "data")

                        # this is the "data" e.g. openMM energy eval
                        try:
                            vals = data.db.get( mol_node.payload).get( "data").get("energy")
                            vals *= converters.get( param)
                        except Exception:
                            #fd.close()
                            print("No aux data from", data.name, mol_node.payload)
                            continue

                    # need to now measure each constraint
                    #ang_pl = [c.payload for c in QCA.node_iter_to_root( mol_node, select="Constraint")][0]
                    #ang = [tuple_to_hyphenated(x[1]) for x in constraints]
                        angle_str = []
                        for cons in constraints:
                            indices = tuple_to_hyphenated(cons[1])
                            angle = TorsionOperation.measure_praxeolitic_single(qc_mol, list(cons[1]))
                            if np.sign(angle) != np.sign(cons_set[order[m_idx]]):
                                angle *= -1.0
                            angle_str = [cons[0],indices,float(cons_set[order[m_idx]])]
                            angle_str = [cons[0],indices,angle]
                            #qc_angle = qcelemental.models.molecule.Molecule.from_data(qc_mol).measure(list(cons[1]))
                            #angle_str = [cons[0],indices,float(qc_angle)]
                            #angle_str = [cons[0],indices,angle]

                            if type(vals) is unit.Quantity:
                                vals = vals.value_in_unit( vals.unit)
                            print("    ", entry_node.payload, mol_id, ene, *angle_str,vals)
                            fd.write(out_str.format( entry_node.payload, mol_id, ene, *angle_str, \
                                  vals, \
                                  smiles_indexed ))
                    #for atoms in constraints:
                    #    ang 
                    #if num_angs > 1:
                    #    ang = [ tuple_to_hyphenated( x) for y in ang_pl for x in y]
                    #else:
                    #    ang = [ tuple_to_hyphenated( x) for x in ang_pl]

                fd.close()
#}}}
#{{{

extract_torsions_for_same_molecule(QCA, param_types, data_chart, converters)

#}}}
if __name__ == "__main__":
    extract_torsions_for_same_molecule(QCA, param_types, data_chart, converters)

#                    # sigh, assume vals are scalars
#                    d[ang].append( [ene, vals[0]] )
