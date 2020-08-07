
import offsb
import offsb.qcarchive.qcatree as qca
import qcportal as ptl
import qcfractal.interface as ptl
import os
import sys
import pickle

class QCArchiveSpellBook():

    openff_qcarchive_datasets_bad_name = [
        ('OptimizationDataset', 'FDA Optimization Dataset 1'),
        ('OptimizationDataset', 'Kinase Inhibitors: WBO Distributions'),
        ('OptimizationDataset', 'Pfizer Discrepancy Optimization Dataset 1')
    ]
    openff_qcarchive_datasets_default = [
        ('GridOptimizationDataset', 'OpenFF Trivalent Nitrogen Set 1'),
        ('GridOptimizationDataset', 'OpenFF Trivalent Nitrogen Set 2'),
        ('GridOptimizationDataset', 'OpenFF Trivalent Nitrogen Set 3'),
        ("TorsionDriveDataset", 'OpenFF Fragmenter Validation 1.0'),              
        ("TorsionDriveDataset", 'OpenFF Full TorsionDrive Benchmark 1'),               
        ("TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 1 Roche'),
        ("TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 1 Roche 2'),
        ("TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 2 Coverage'),
        ("TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 2 Coverage 2'),
        ("TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 3 Pfizer Discrepancy'),
        ("TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 3 Pfizer Discrepancy 2'),
        ("TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 4 eMolecules Discrepancy'),
        ("TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 4 eMolecules Discrepancy 2'),
        ("TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 5 Bayer'),
        ("TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 5 Bayer 2'),
        ("TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 6 Supplemental'),
        ("TorsionDriveDataset", 'OpenFF Gen 2 Torsion Set 6 Supplemental 2'),
        ("TorsionDriveDataset", 'OpenFF Group1 Torsions'),                            
        ("TorsionDriveDataset", 'OpenFF Group1 Torsions 2'),                           
        ("TorsionDriveDataset", 'OpenFF Group1 Torsions 3'),                         
        ("TorsionDriveDataset", 'OpenFF Primary Benchmark 1 Torsion Set'),           
        ("TorsionDriveDataset", 'OpenFF Primary Benchmark 2 Torsion Set'),            
        ("TorsionDriveDataset", 'OpenFF Primary TorsionDrive Benchmark 1'),           
        ("TorsionDriveDataset", 'OpenFF Substituted Phenyl Set 1'),                  
        ("TorsionDriveDataset", 'OpenFF Rowley Biaryl v1.0'),                  
        ('OptimizationDataset', 'OpenFF Gen 2 Opt Set 1 Roche'),
        ('OptimizationDataset', 'OpenFF Gen 2 Opt Set 2 Coverage'),
        ('OptimizationDataset', 'OpenFF Gen 2 Opt Set 3 Pfizer Discrepancy'),
        ('OptimizationDataset', 'OpenFF Gen 2 Opt Set 4 eMolecules Discrepancy'),
        ('OptimizationDataset', 'OpenFF Gen 2 Opt Set 5 Bayer'),
        ('OptimizationDataset', 'FDA Optimization Dataset 1'),
        ('OptimizationDataset', 'Kinase Inhibitors: WBO Distributions'),
        ('OptimizationDataset', 'OpenFF Discrepancy Benchmark 1'),
        ('OptimizationDataset', 'OpenFF Ehrman Informative Optimization v0.2'),
        ('OptimizationDataset', 'OpenFF Full Optimization Benchmark 1'),
        ('OptimizationDataset', 'OpenFF Gen 2 Opt Set 1 Roche'),
        ('OptimizationDataset', 'OpenFF Gen 2 Opt Set 2 Coverage'),
        ('OptimizationDataset', 'OpenFF Gen 2 Opt Set 3 Pfizer Discrepancy'),
        ('OptimizationDataset', 'OpenFF Gen 2 Opt Set 4 eMolecules Discrepancy'),
        ('OptimizationDataset', 'OpenFF Gen 2 Opt Set 5 Bayer'),
        ('OptimizationDataset', 'OpenFF NCI250K Boron 1'),
        ('OptimizationDataset', 'OpenFF Optimization Set 1'),
        ('OptimizationDataset', 'OpenFF Primary Optimization Benchmark 1'),
        ('OptimizationDataset', 'OpenFF VEHICLe Set 1'),
        ('OptimizationDataset', 'Pfizer Discrepancy Optimization Dataset 1'),
        ('OptimizationDataset', 'OpenFF Protein Fragments v1.0'),
    ]

    # openff_qcarchive_datasets_default = [
    #     ('OptimizationDataset', 'OpenFF Gen 2 Opt Set 2 Coverage'),
    #     ("TorsionDriveDataset", 'OpenFF Group1 Torsions'),                            
    #     ("TorsionDriveDataset", 'OpenFF Group1 Torsions 2'),                           
    #     ("TorsionDriveDataset", 'OpenFF Group1 Torsions 3'),                         
    #     ("TorsionDriveDataset", 'OpenFF Primary Benchmark 1 Torsion Set'),           
    #     ("TorsionDriveDataset", 'OpenFF Primary Benchmark 2 Torsion Set'),            
    #     ("TorsionDriveDataset", 'OpenFF Primary TorsionDrive Benchmark 1'),           
    # ]

    cache_dir="."

    drop_hessians = True
    drop_intermediates = True

    def save(self, tree):
        name = os.path.join(self.cache_dir, tree.name + ".p")
        print("Saving: ", tree.ID, "as", name, end=" ... ")
        tree.to_pickle( db=True, name=name)
        print( "{:12.1f} MB".format(os.path.getsize( name)/1024**2))

    def load(self, sets, load_all=False):

        client = ptl.FractalClient()
        if self.QCA is None:
            self.QCA = qca.QCATree("QCA", root_payload=client, node_index=dict(), db=dict())
        newdata = False

        # take any dataset with OpenFF in the dataset name
        if load_all:
            sets = sets.copy()
            for index, row in client.list_collections().iterrows():
                if "OpenFF" in index[1] and index[0] != "Dataset":
                    sets.append(index)

        for s in sets:
            if not any([s[1] == self.QCA[i].name for i in self.QCA.root().children]):
                print("Dataset", s, "not in local db, fetching...")
                newdata = True
                ds = client.get_collection( *s)
                drop = ["Hessian"] if (s[0] == "TorsionDriveDataset" or self.drop_hessians) else []
                #QCA.build_index( ds, drop=["Hessian"])
                #drop=[]
                if self.drop_intermediates:
                    drop.append("Intermediates")
                self.QCA.build_index(ds, drop=drop, keep_specs=["default"],
                    start=0, limit=0)
            else:
                print("Dataset", s, "already indexed")

        if newdata:
            self.save(self.QCA)


    def __init__(self, QCA=None):

        import pickle
        self.QCA = None
        if QCA is not None:
            if isinstance(QCA, str):
                fname = os.path.join(self.cache_dir, QCA)
                if os.path.exists(fname):
                    with open(fname, 'rb') as fid:
                        self.QCA = pickle.load(fid)
        else:
            fname = os.path.join(self.cache_dir, "QCA.p")
            if os.path.exists(fname):
                with open(fname, 'rb') as fid:
                    self.QCA = pickle.load(fid)

            
            aux_sets = self.openff_qcarchive_datasets_bad_name
            
            self.load(aux_sets, load_all=True)
        self.folder_cache = {}

    def energy_minimum_per_molecule(self):

        print("Reporting final optimization energies per molecule")
        if 'per_molecule' not in self.folder_cache:
            fname = os.path.join(self.cache_dir, 'per_molecule.p')
            if os.path.exists(fname):
                with open(fname,'rb') as fid:
                    self.folder_cache['folder_cache'] = pickle.load(fid)
            else:
                print("Collection molecules across datasets...")
                self.folder_cache['per_molecule'] = [x for x in self.QCA.combine_by_entry()]
                with open(os.path.join(self.cache_dir, 'per_molecule.p'), 'wb') as fid:
                    pickle.dump(self.folder_cache['per_molecule'], fid)
        dat = {}
        for folder in self.folder_cache['per_molecule']:
            entry = self.QCA.db[self.QCA[folder.children[0]].payload]['data']
            cmiles = entry.attributes['canonical_isomeric_explicit_hydrogen_smiles']
            enes = []
            for opt in self.QCA.node_iter_depth_first(folder, select="Optimization"):
                rec = self.QCA.db[opt.payload]['data']
                if rec['status'] == "COMPLETE":
                    try:
                        enes.append(rec['energies'][-1])
                    except TypeError:
                        print("This optimization is COMPLETE but has no energy!")

            dat[cmiles] = enes
        return dat

    def constraints_per_molecule(self):
        pass

    def error_report_per_molecule(self):
        pass

    def error_report_per_dataset(self, save_xyz=True, out_fnm=None, full_report=False):
        if out_fnm is None:
            fid = sys.stdout
        else:
            fname = out_fnm
            fid = open(fname, 'w')
        QCA = self.QCA
        for ds_id in QCA.root().children:
            ds_node = QCA[ds_id]
            default = [n for n in QCA.node_iter_depth_first(ds_node) if "QCS-default" in n.payload]
            mindepth = QCA.node_depth(default[0])
            tdi = 0
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
                                elif "Cannot continue a constrained optimization; please implement constrained optimization in Cartesian coordinates" in err['error_message']:
                                    errtype="cartconstr"
                                elif "numpy.linalg.LinAlgError: Eigenvalues did not converge" in err['error_message']:
                                    errtype="numpy-eigh"
                                elif "geometric.errors.GeomOptNotConvergedError: Optimizer.optimizeGeometry() failed to converge." in err['error_message']:
                                    errtype="optconverge"
                                elif "RuntimeError: Unsuccessful run. Possibly -D variant not available in dftd3 version." in err['error_message']:
                                    errtype="dftd3variant"
                                elif "Could not converge SCF iterations" in err['error_message']:
                                    errtype="scfconv"
                                elif "distributed.scheduler.KilledWorker" in err['error_message']:
                                    errtype="needsrestart-daskkilled"
                                elif "concurrent.futures.process.BrokenProcessPool" in err['error_message']:
                                    errtype="needsrestart-brokenpool"
                                elif len(err['error_message'].strip().strip('\n')) == 0:
                                    errtype="emptyerror"
                                else:
                                    errtype="nocategory"

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
                                    # print("Trajectory found... saving", fname)
                                    mol = QCA.db[traj[-1].payload]['data']
                                    with open(fname, 'w') as xyzfid:
                                        _ = offsb.qcarchive.qcmol_to_xyz(mol, fd=xyzfid, comment=metadata+"OptID "+ node.payload + " MoleculeID " + traj[-1].payload + " Error= " + xyzerrmsg)
                                else:
                                    mol_id = 'QCM-'+qcp['initial_molecule']
                                    fname ='geometric.'+errtype+'.initial.'+ tdname + node.payload + '.' + mol_id + '.xyz'
                                    # print("Trajectory not found... saving input molecule", fname)
                                    mol = QCA.db[mol_id]['data']
                                    if 'geometry' in mol and 'symbols' in mol:
                                        with open(fname, 'w') as xyzfid:
                                            _ = offsb.qcarchive.qcmol_to_xyz(mol, fd=xyzfid, comment=metadata+"OptID "+ node.payload + " MoleculeID " + mol_id + " Error= " + xyzerrmsg)
                                    else:
                                        # print("Initial molecule missing!")
                                        errmsg += "XXXXISSUEXXXX\n"
                                        errmsg += "Initial molecule was missing!\n"
                                        errmsg += "xxxxxxxxxxxxx\n"
                            if not (qcp['stdout'] is None):
                                msg = list(client.query_kvstore(int(qcp['stdout'])).values())[0]
                                errmsg += "xxxxISSUExxxx\n"
                                errmsg += "Status was not complete; stdout is:\n"
                                errmsg += msg
                                errmsg += "xxxxxxxxxxxxx\n"
                            if not (qcp['stderr'] is None):
                                msg = list(client.query_kvstore(int(qcp['stderr'])).values())[0]
                                errmsg += "####ISSUE####\n"
                                errmsg += "Status was not complete; stderr is:\n"
                                errmsg += msg
                                errmsg += "#############\n"
                        except Exception as e:
                            fid.write("Internal issue:\n" + str(e) + '\n')

                    if status != "COMPLETE" and node.name == "TorsionDrive":
                        statbar = "XXXX"
                tderror = False
                if node.name == "TorsionDrive" and status != "COMPLETE":
                    tdi += 1
                    tderror = True
                if full_report or (len(errmsg) > 0 or tderror):
                    out_str = "{:2d} {} {} {}\n".format(tdi,statbar*i, " ".join([str(node.index),str(node.name),  str(node.payload)]), status)
                    fid.write(out_str)
                    if errmsg != "":
                        err_str = "\n{}\n".format(errmsg)
                        fid.write(err_str)


        fid.write("____Done____\n")

        # close file if not std.stdout
        if out_fnm is not None:
            fid.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The OpenForceField Spellbook")
    parser.add_argument('--report-errors', action="store_true")
    parser.add_argument('--report-energy', action="store_true")
    args = parser.parse_args()

    qcasb = None
    if args.report_energy:
        if qcasb is None:
            qcasb = QCArchiveSpellBook()
        enes_per_mol= qcasb.energy_minimum_per_molecule()
        with open("enes.txt", 'w') as fid:
            for cmiles,enes in enes_per_mol.items():
                fid.write("# {:s}\n".format(cmiles))
                for ene in enes:
                    fid.write("    {:12.8f}\n".format(ene))
    if args.report_errors:
        if qcasb is None:
            qcasb = QCArchiveSpellBook()
        qcasb.error_report_per_dataset()

