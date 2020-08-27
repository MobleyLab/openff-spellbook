import offsb
import offsb.qcarchive.qcatree as qca
import qcportal as ptl
import qcfractal.interface as ptl
import os
import sys
import pickle

def load_dataset_input(fnm):
    datasets = [tuple([line.split()[0]," ".join(line.strip('\n').split()[1:])])
        for line in open(fnm).readlines()]
    return datasets

class QCArchiveSpellBook():

    openff_qcarchive_datasets_bad_name = [
        ('OptimizationDataset', 'FDA Optimization Dataset 1'),
        ('OptimizationDataset', 'Kinase Inhibitors: WBO Distributions'),
        ('OptimizationDataset', 'Pfizer Discrepancy Optimization Dataset 1')
    ]
    openff_qcarchive_datasets_skip = [
        ('OptimizationDataset', 'OpenFF Ehrman Informative Optimization v0.1')
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

        print("Aux sets to load:")
        print(sets)

        # take any dataset with OpenFF in the dataset name
        if load_all:
            sets = sets.copy()
            for index, row in client.list_collections().iterrows():
                for skip_set in self.openff_qcarchive_datasets_skip:
                    if skip_set[1] == index[1]:
                        print("Skipping", index)
                        continue
                if "OpenFF" in index[1] and index[0] != "Dataset":
                    sets.append(index)

        for s in sets:
            name = s[1].split("/")
            specs = ['default'] if len(name) == 1 else name[1].split()
            specs = [x.strip() for x in specs]
            print("Input has specs", specs)
            name = name[0].rstrip().lstrip()
            if not any([name == self.QCA[i].name for i in self.QCA.root().children]):
                print("Dataset", s, "not in local db, fetching...")
                newdata = True
                ds = client.get_collection(s[0], name)
                drop = ["Hessian"] if (s[0] == "TorsionDriveDataset" or self.drop_hessians) else []
                #QCA.build_index( ds, drop=["Hessian"])
                #drop=[]

                if self.drop_intermediates:
                    drop.append("Intermediates")

                self.QCA.build_index(ds, drop=drop, keep_specs=specs,
                    start=0, limit=0)
            else:
                print("Dataset", s, "already indexed")

        if newdata:
            self.save(self.QCA)


    def __init__(self, datasets=None, QCA=None):

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
            load_all=True
            if datasets is not None:
                aux_sets = datasets
                load_all = False

            self.load(aux_sets, load_all=load_all)
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

    def torsiondrive_groupby_openff_param(self, ffname, param):

        """
        Create torsiondrive summary plot of a parameter.
        Should be able to detect type of measurement from param type
            Downloads data
            Downloads min torsion molecules
            Creates the appropriate ff-label command
            Creates the appropriate measure command from ff indices
            Aggregates data
            Sorts in a reasonable format
            Save as file (plot using something else)
        """
        pass

    #def collate_1d_torsiondrive(self, entry_data=[], molecule_data=[],
    #    targets=None, out_fname=None):

    #    # take a dict, with keys as filename pickles
    #    # then either Entry, Molecule, etc
    #    # then Any other key
    #    # therefore, this is a projection, and arg should be a list of keys
        
    #    if targets is None:
    #        targets = self.QCA.iter_entry()
    #    f = open("out.dat", 'w')
    #    for i,entry_node in enumerate(targets):
    #        key = entry_node.payload
    #        params = labels.db[key]['data']
    #        if "Bonds" not in params:
    #            print(i, entry_node, "Missing labels")
    #            continue
    #        all_params = params['Bonds']
    #        for mol_node in self.QCA.node_iter_depth_first(entry_node, select="Molecule"):
    #            constr = [x.payload for x in self.QCA.node_iter_to_root(mol_node, select="Constraint")]
    #            mol_key = mol_node.payload
    #            mol = self.QCA.db[mol_key]
    #            bond_vals = bonds.db[mol_key]
    #            for atom_key, lbl in all_params.items():
    #                #if atom_key not in bond_vals:
    #                #    atom_key = tuple(atom_key[::-1])
    #                try:
    #                    # print(i, entry_node, constr, atom_key, bond_vals[atom_key], lbl)
    #                    f.write("{} {} {} {} {} {}\n".format(i, constr[0][2], atom_key[0], atom_key[1], bond_vals[atom_key][0], lbl))
    #                except Exception as e:
    #                    print(self.QCA.db[key])
    #                    print(entry_node,lbl) print(bond_vals)
    #                    print(all_params)
    #                    print(mol_node, mol_key)
    #                    print(e)
    #                    print(i)
    #                    raise
    #    f.close()

    
    def assign_labels_from_openff(self, openff_name, name):
        from offsb.op.openforcefield import OpenForceFieldTree as OP
        ext = ".offxml"
        if not openff_name.endswith(ext):
            openff_name += ext
        labeler = OP(self.QCA, name, openff_name)
        labeler.apply()
        labeler.to_pickle()

    def _measure(self, smi, op_cls, name):
        from offsb.search.smiles import SmilesSearchTree

        # assume we want all final geometries
        self.QCA.cache_optimization_minimum_molecules()

        query = SmilesSearchTree(smi, self.QCA, "query")
        query.apply()

        op = op_cls(query, name)
        op.apply()
        op.to_pickle()

    def measure_bonds(self, name):
        from offsb.op.geometry import BondOperation as OP
        self._measure("[*]~[*]", OP, name)

    def measure_angles(self, name):
        from offsb.op.geometry import AngleOperation as OP
        self._measure("[*]~[*]~[*]", OP, name)

    def measure_dihedrals(self, name):
        from offsb.op.geometry import TorsionOperation as OP
        self._measure("[*]~[*]~[*]~[*]", OP, name)

    def measure_outofplanes(self, name):
        from offsb.op.geometry import ImproperTorsionOperation as OP
        self._measure("[*]~[*](~[*])~[*]", OP, name)


    def error_report_per_dataset(self, save_xyz=True, out_fnm=None, full_report=False):
        if out_fnm is None:
            fid = sys.stdout
        else:
            fname = out_fnm
            fid = open(fname, 'w')
        QCA = self.QCA
        for ds_id in QCA.root().children:
            ds_node = QCA[ds_id]
            specs = [n for n in QCA.node_iter_depth_first(ds_node) if "QCS" in n.payload]
            for spec in specs:
                mindepth = QCA.node_depth(spec)
                tdi = 0
                fid.write("==== DATASET ==== {}\n".format(ds_node.name))
                for node in QCA.node_iter_dive(spec):
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
    parser.add_argument('--datasets', type=str)
    args = parser.parse_args()

    if args.datasets is not None:
        datasets = load_dataset_input(args.datasets)
    
    qcasb = None
    if args.report_energy:
        qcasb = QCArchiveSpellBook(datasets=args.datasets)
        enes_per_mol= qcasb.energy_minimum_per_molecule()
        with open("enes.txt", 'w') as fid:
            for cmiles,enes in enes_per_mol.items():
                fid.write("# {:s}\n".format(cmiles))
                for ene in enes:
                    fid.write("    {:12.8f}\n".format(ene))
    if args.report_errors:
        if qcasb is None:
            qcasb = QCArchiveSpellBook(datasets=args.datasets)
        qcasb.error_report_per_dataset()

