
#
#{{{
import os
import qcfractal.interface as ptl
import offsb.qcarchive.qcatree as qca
import offsb
import treedi.node as Node
client = ptl.FractalClient()
import pandas as pd
#}}}

#
#{{{

#import pickle
#with open("QCA.p", 'rb') as fid:
#    QCA = pickle.load(fid)

with pd.option_context(
        'display.max_rows', 10000000, 
        'display.max_columns', 100,
        'display.max_colwidth', 10000):
    print(client.list_collections())

#}}}

#{{{
sets = [
    ( "TorsionDriveDataset", "OpenFF Substituted Phenyl Set 1" )
]
sets = [
    ( "TorsionDriveDataset", "OpenFF Fragmenter Validation 1.0" )
]
sets = [
    ( "OptimizationDataset", "OpenFF Full Optimization Benchmark 1" )
]
sets = [
    ( "TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 1 Roche'),
    ( "TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 1 Roche 2'),
    ( "TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 2 Coverage'),
    ( "TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 2 Coverage 2'),
    ( "TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 3 Pfizer Discrepancy'),
    ( "TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 3 Pfizer Discrepancy 2'),
    ( "TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 4 eMolecules Discrepancy'),
    ( "TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 4 eMolecules Discrepancy 2'),
    ( "TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 5 Bayer'),
    ( "TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 5 Bayer 2'),
    ( "TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 6 Supplemental'),
    ( "TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 6 Supplemental 2'),
]
def load(sets, QCA=None):

    if QCA is None:
        QCA = qca.QCATree("QCA", root_payload=client, node_index=dict(), db=dict())
    for s in sets:
        if not any([s[1] == QCA[i].name for i in QCA.root().children]):
            print("Dataset", s, "not in local db, fetching...")
            ds = client.get_collection( *s)
            drop = ["Hessian"] if s[0] == "TorsionDriveDataset" else []
            #QCA.build_index( ds, drop=["Hessian"])
            #drop=[]
            drop=["Intermediates", "Hessian"]
            QCA.build_index( ds, drop=drop, keep_specs=["default"],
                start=0, limit=0)
        else:
            print("Dataset", s, "already indexed")

    return QCA

def save( tree):
    name = os.path.join(".", tree.name + ".p")
    print("Saving: ", tree.ID, "as", name, end=" ... ")
    tree.to_pickle( db=True, name=name)
    print( "{:12.1f} MB".format(os.path.getsize( name)/1024**2))
import pickle
QCA = None
if os.path.exists('QCA.p'):
    with open("QCA.p", 'rb') as fid:
        QCA = pickle.load(fid)
QCA = load(sets, QCA=QCA)
#QCA.cache_optimization_minimum_molecules()
#save(QCA)
#ds_nodes = [ QCA[index] for index in QCA.root().children]
#entries = list(QCA.iter_entry( ds_nodes))
#}}}

fname = "report.log"
print("Generating report in", fname)
fid = open(fname, 'w')
for ds_id in QCA.root().children:
    ds_node = QCA[ds_id]
    default = [n for n in QCA.node_iter_depth_first(ds_node) if "QCS-default" in n.payload]
    mindepth = QCA.node_depth(default[0])
    tdi = 0
    #fid = open("biphenyl_ds_errors.log", 'w')
    fid.write("==== DATASET ==== {}\n".format(ds_node.name))
    for node in QCA.node_iter_dive(default):
        i = QCA.node_depth(node) - mindepth+1
        status=""
        statbar= "    "
        errmsg = ""
        hasstat = False
        try:
            hasstat = "status" in QCA.db[node.payload]["data"]
        except Exception:
            pass
        if hasstat:
            status = QCA.db[node.payload]["data"]["status"][:]
            if status != "COMPLETE" and node.name == "Optimization":
                try:
                    statbar = "----"
                    qcp = QCA.db[node.payload]['data']
                    client = qcp['client']
                    # print("error", qcp['error'], qcp['error'] is None)

                    errtype = ""
                    xyzerrmsg = ""
                    if not (qcp['error'] is None):
                        err = list(client.query_kvstore(int(qcp['error'])).values())[0]
                        errmsg += "####ERROR####\n"
                        errmsg += "Error type: " + err['error_type'] + "\n"
                        errmsg += "Error:\n" + err['error_message']
                        errmsg += "############\n"
                        if "RuntimeError: Not bracketed" in err['error_message']:
                            errtype="bracketed"

                        if "Cannot continue a constrained optimization; please implement constrained optimization in Cartesian coordinates" in err['error_message']:
                            errtype="cartconstr"

                        if "numpy.linalg.LinAlgError: Eigenvalues did not converge" in err['error_message']:
                            errtype="numpy-eigh"

                        if "geometric.errors.GeomOptNotConvergedError: Optimizer.optimizeGeometry() failed to converge." in err['error_message']:
                            errtype="optconverge"

                        if "RuntimeError: Unsuccessful run. Possibly -D variant not available in dftd3 version." in err['error_message']:
                            errtype="dftd3variant"

                        for line in err['error_message'].split('\n')[::-1]:
                            if len(line) > 0:
                                xyzerrmsg = line
                                break

                    elif status == "ERROR":
                        errmsg += "####FATAL####\n"
                        errmsg += "Job errored and no error on record. Input likely bogus; check your settings\n"
                        errmsg += "############\n"
                        errtype="bogus"
                        xyzerrmsg="Job errored and no error on record. Input likely bogus; check your settings"
                        
                    if len(errtype) > 0: 
                        traj = list(QCA.node_iter_depth_first(node, select="Molecule"))
                        metadata = ""
                        tdids = [n.payload for n in QCA.node_iter_to_root(node, select="TorsionDrive")]

                        if len(tdids) > 0:
                            metadata += 'TdIDs ' + '.'.join(tdids) + ' '

                        contrs = [str(n.payload) for n in QCA.node_iter_to_root(node, select="Constraint")]
                        if len(contrs) > 0:
                            metadata += 'Constraints ' + '.'.join(contrs) + ' '

                        tdname = '.'.join(tdids)
                        if len(tdname) > 0:
                            tdname += '.'
                        if len(traj) > 0:
                            fname ='geometric.'+errtype+'.traj.' + tdname + node.payload + '.' + traj[-1].payload + '.xyz'
                            print("Trajectory found... saving", fname)
                            mol = QCA.db[traj[-1].payload]['data']
                            with open(fname, 'w') as xyzfid:
                                _ = offsb.qcarchive.qcmol_to_xyz(mol, fd=xyzfid, comment=metadata+"OptID "+ node.payload + " MoleculeID " + traj[-1].payload + " Error= " + xyzerrmsg)
                        else:
                            mol_id = 'QCM-'+qcp['initial_molecule']
                            fname ='geometric.'+errtype+'.initial.'+ tdname + node.payload + '.' + mol_id + '.xyz'
                            print("Trajectory not found... saving input molecule", fname)
                            mol = QCA.db[mol_id]['data']
                            with open(fname, 'w') as xyzfid:
                                _ = offsb.qcarchive.qcmol_to_xyz(mol, fd=xyzfid, comment=metadata+"OptID "+ node.payload + " MoleculeID " + mol_id + " Error= " + xyzerrmsg)
                    if not (qcp['stdout'] is None):
                        msg = list(client.query_kvstore(int(qcp['stdout'])).values())[0]
                        errmsg += "----ISSUE----\n"
                        errmsg += "Status was not complete; stdout is:\n"
                        errmsg += msg
                        errmsg += "-------------\n"
                    if not (qcp['stderr'] is None):
                        msg = list(client.query_kvstore(int(qcp['stderr'])).values())[0]
                        errmsg += "####ISSUE####\n"
                        errmsg += "Status was not complete; stderr is:\n"
                        errmsg += msg
                        errmsg += "-------------\n"
                except Exception as e:
                    fid.write("Internal issue:\n" + str(e) + '\n')

            if status != "COMPLETE" and node.name == "TorsionDrive":
                statbar = "XXXX"
        if node.name == "TorsionDrive" and status != "COMPLETE":
            tdi += 1
        fid.write("{:2d} {} {} {}\n".format(tdi,statbar*i, " ".join([str(node.index),str(node.name),  str(node.payload)]), status))
        if errmsg != "":
            fid.write("\n{}\n".format(errmsg))


fid.close()

