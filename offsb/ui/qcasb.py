import json
import os
import pickle
import sys

import numpy as np

import offsb
import offsb.rdutil
import offsb.qcarchive.qcatree as qca
import qcfractal.interface as ptl
import simtk.unit
from openmmtools.utils import quantity_from_string


def load_dataset_input(fnm):
    datasets = [
        tuple([line.split()[0], " ".join(line.strip("\n").split()[1:])])
        for line in open(fnm).readlines()
        if not line.lstrip().startswith("#")
    ]
    return datasets


class QCArchiveSpellBook:

    # OpenFF datasets (but not named as such)
    openff_qcarchive_datasets_bad_name = [
        ("OptimizationDataset", "FDA Optimization Dataset 1"),
        ("OptimizationDataset", "Kinase Inhibitors: WBO Distributions"),
        ("OptimizationDataset", "Pfizer Discrepancy Optimization Dataset 1"),
    ]

    # blacklists
    openff_qcarchive_datasets_skip = [
        ("OptimizationDataset", "OpenFF Ehrman Informative Optimization v0.1")
    ]
    openff_qcarchive_datasets_default = []

    cache_dir = "."

    drop_hessians = True
    drop_intermediates = True

    def save(self, tree):
        name = os.path.join(self.cache_dir, tree.name + ".p")
        print("Saving: ", tree.ID, "as", name, end=" ... ")
        tree.to_pickle(db=True, name=name)
        print("{:12.1f} MB".format(os.path.getsize(name) / 1024 ** 2))

    def load(self, sets, load_all=False):

        # client = ptl.FractalClient("localhost:7777", verify=False)
        client = ptl.FractalClient()
        if self.QCA is None:
            self.QCA = qca.QCATree(
                "QCA", root_payload=client, node_index=dict(), db=dict()
            )
        newdata = False

        # print("Aux sets to load:")
        # print(sets)

        # take any dataset with OpenFF in the dataset name
        if load_all:
            sets = sets.copy()
            for index, row in client.list_collections().iterrows():
                for skip_set in self.openff_qcarchive_datasets_skip:
                    if skip_set[1] == index[1]:
                        print("Skipping", index, "because it is in the blacklist")
                        continue
                if "OpenFF" in index[1] and index[0] != "Dataset":
                    sets.append(index)

        for s in sets:
            name = s[1].split("/")
            specs = ["default"] if len(name) == 1 else name[1].split()
            specs = [x.strip() for x in specs]
            # print("Input has specs", specs)
            name = name[0].rstrip().lstrip()
            if not any([name == self.QCA[i].name for i in self.QCA.root().children]):
                print("Dataset", s, "not in local db, fetching...")
                newdata = True
                ds = client.get_collection(s[0], name)
                drop = (
                    ["Hessian"]
                    if (s[0] == "TorsionDriveDataset" or self.drop_hessians)
                    else []
                )
                # QCA.build_index( ds, drop=["Hessian"])
                # drop=[]

                if self.drop_intermediates:
                    drop.append("Intermediates")

                self.QCA.build_index(ds, drop=drop, keep_specs=specs, start=0, limit=0)
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
                    with open(fname, "rb") as fid:
                        self.QCA = pickle.load(fid)
        else:
            fname = os.path.join(self.cache_dir, "QCA.p")
            if os.path.exists(fname):
                with open(fname, "rb") as fid:
                    self.QCA = pickle.load(fid)

            aux_sets = self.openff_qcarchive_datasets_bad_name
            load_all = True
            if datasets is not None:
                aux_sets = datasets
                load_all = False

            self.load(aux_sets, load_all=load_all)
        self.folder_cache = {}

        # If data is generated for a parameters, save the list later
        # in case we will e.g. plot them
        self._param_cache = []

    def _plot_kt_displacements(
        self, ax, val, delta, marker="o", color="black", label=None
    ):

        ax.axhline(
            y=val,
            ls="-",
            marker=marker,
            color="black",
            ms=12,
            mec="black",
            mfc=color,
            label=label,
        )
        ax.axhline(
            y=val + delta,
            ls="--",
            marker=marker,
            color="black",
            ms=6,
            mec="black",
            mfc=color,
        )
        ax.axhline(
            y=val - delta,
            ls="--",
            marker=marker,
            color="black",
            ms=6,
            mec="black",
            mfc=color,
        )

    def energy_minimum_per_molecule(self):

        print("Reporting final optimization energies per molecule")
        if "per_molecule" not in self.folder_cache:
            fname = os.path.join(self.cache_dir, "per_molecule.p")
            if os.path.exists(fname):
                with open(fname, "rb") as fid:
                    self.folder_cache["folder_cache"] = pickle.load(fid)
            else:
                print("Collection molecules across datasets...")
                self.folder_cache["per_molecule"] = [
                    x for x in self.QCA.combine_by_entry()
                ]
                with open(os.path.join(self.cache_dir, "per_molecule.p"), "wb") as fid:
                    pickle.dump(self.folder_cache["per_molecule"], fid)
        dat = {}
        for folder in self.folder_cache["per_molecule"]:
            entry = self.QCA.db[self.QCA[folder.children[0]].payload]["data"]
            cmiles = entry.attributes["canonical_isomeric_explicit_hydrogen_smiles"]
            enes = []
            for opt in self.QCA.node_iter_depth_first(folder, select="Optimization"):
                rec = self.QCA.db[opt.payload]["data"]
                if rec.status[:] == "COMPLETE":
                    try:
                        enes.append(rec.energies[-1])
                    except TypeError:
                        print("This optimization is COMPLETE but has no energy!")

            dat[cmiles] = enes
        return dat

    def constraints_per_molecule(self):
        pass

    def error_report_per_molecule(self):
        pass

    def _apply_kt_plots(self, meta, ax, ax2, kwargs):

        if meta["param"]["id"][0] in "ait":
            equil_key = "angle"
        elif meta["param"]["id"][0] in "b":
            equil_key = "length"
        else:
            raise Exception("Could not parse param metadata")

        kT = simtk.unit.Quantity(0.001987 * 298.15, simtk.unit.kilocalorie_per_mole)

        k = quantity_from_string(meta["param"]["k"])
        if equil_key == "angle" and "radian" in str(k.unit):
            k = k.in_units_of(k.unit * simtk.unit.radian ** 2 / simtk.unit.degree ** 2)
        equil = quantity_from_string(meta["param"][equil_key])
        delta = (2 * (kT) / k) ** 0.5

        val = equil / equil.unit
        delta = delta / delta.unit
        self._plot_kt_displacements(ax, val, delta, **kwargs)
        self._plot_kt_displacements(ax2, val, delta, **kwargs)

        label_unit_map = {"angstrom": r"$\mathrm{\AA}$", "degree": r"deg"}

        label_unit = label_unit_map[str(equil.unit)]

        return label_unit

    def plot_torsiondrive_groupby_openff_param(self, infile, oldparamfile=None):

        import matplotlib.pyplot as plt
        import matplotlib as mpl

        # TODO: make the fancier multirow plots
        rows = 1

        meta = {}

        mpl.rc("font", **{"size": 13})

        files = []
        old_files = []
        if os.path.exists(infile):
            files = [infile]
            if oldparamfile is None:
                oldfiles = [None]
            else:
                old_files = [oldparamfile]
        elif len(self._param_cache) > 0:
            for param in self._param_cache:
                filename = ".".join([infile, param, "csv"])
                if os.path.exists(filename):
                    files.append(filename)
                else:
                    o = "Param {} is cached, but datafile {} not found"
                    print(o.format(param, filename))
                    continue
                if oldparamfile is None:
                    old_files.append(None)
                else:
                    filename = ".".join([oldparamfile, param, "csv"])
                    if os.path.exists(filename):
                        old_files.append(filename)
                    else:
                        o = "Param {} is cached, but previous datafile {} not found"
                        print(o.format(param, filename))

        for i, (file, oldfile) in enumerate(zip(files, old_files)):

            with open(file) as f:
                valid = False
                for line in f:
                    tokens = line.split()
                    if tokens[0] != "#JSON":
                        break
                    name = tokens[1]
                    js = "".join(tokens[2:])
                    meta[name] = json.loads(js)
                    valid = True

            if not valid:
                continue

            dat = np.loadtxt(file)

            if len(dat) == 0:
                continue

            oldmeta = None

            if oldfile is not None:
                oldmeta = {}
                with open(oldfile) as f:
                    for line in f:
                        tokens = line.split()
                        if tokens[0] != "#JSON":
                            break
                        name = tokens[1]
                        js = "".join(tokens[2:])
                        oldmeta[name] = json.loads(js)

            fig = plt.figure(figsize=(6, 4), dpi=300)
            fig.subplots_adjust(
                wspace=0.0, hspace=0.25, right=0.95, left=0.15, bottom=0.15
            )

            ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
            ax2 = plt.subplot2grid((1, 3), (0, 2), sharey=ax)

            color = "black"
            entries = list(set(dat[:, 0]))
            for entry in entries:
                mol = dat[dat[:, 0] == entry]
                params = set(list(mol[:, 2]))
                for param in params:
                    data = mol[mol[:, 2] == param]
                    x = data[:, 1]
                    y = data[:, 3]

                    # The points
                    ax.plot(x, y, lw=0.0, marker=".", color=color, ms=2, alpha=0.8)

                    # The lines
                    ax.plot(x, y, lw=0.1, ls="-", ms=0.0, color=color, alpha=0.5)

            # some hardcoding for b7 and doing a comparison...
            # TODO make config options for this
            # ax.set_ylim(1.42,1.56)

            ax2.hist(
                dat[:, 3],
                orientation="horizontal",
                color=color,
                histtype="stepfilled",
                alpha=0.3,
                bins=30,
            )
            ax2.hist(
                dat[:, 3],
                orientation="horizontal",
                color=color,
                histtype="step",
                lw=2,
                bins=30,
            )
            ax2.tick_params(labelleft=False, direction="inout")

            if oldmeta is not None:
                kwargs = dict(color="red", marker="s", label="Initial")
                self._apply_kt_plots(oldmeta, ax, ax2, kwargs)

            color = "black" if oldmeta is None else "green"
            kwargs = dict(color=color, label="Final")
            label_unit = self._apply_kt_plots(meta, ax, ax2, kwargs)

            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Measurement (" + label_unit + ")")
            ax2.set_xlabel("Count")
            t = "Parameter " + meta["param"]["id"] + " " + meta["param"]["smirks"]
            fig.suptitle(t, fontsize=11)

            if oldmeta is not None:
                legend = ax2.legend(
                    frameon=False, loc="upper right", markerscale=0.7, fontsize=8
                )
                # for mi in [0,1]:
                #     m = legend.legendHandles[mi].get_markersize()
                #     legend.legendHandles[mi].set_markersize(m)
                # ax2.legend(handles=legend.legendHandles, frameon=False, loc='upper right')
                # fig.canvas.draw()
                # fig.canvas.flush_events()

            prefix = infile.split(".")
            if len(prefix[-1]) == 3 or len(prefix[-1]) == 4:
                prefix = prefix[:-1]
            if oldmeta is not None:
                compare_from = os.path.basename(oldparamfile).split(".")
                # weird hack to make sure that the extension is actually
                # extension. Sometimes we give extensionless prefix names
                # so stripping the extension is not wanted
                if len(compare_from[-1]) == 3 or len(compare_from[-1]) == 4:
                    compare_from = compare_from[:-1]
                prefix += ["from_" + ".".join(compare_from)]
            figname = ".".join(prefix + ["td_groupby", meta["param"]["id"], "png"])
            fig.savefig(figname)
            print(i, "/", len(files), "Saved image", figname)
            plt.close(fig)

    def torsiondrive_select(
        self,
        smarts="[*]~[*]~[*]~[*]",
        exact=True,
        wbo=False,
        mbo=False,
        ffname=None,
        bond_param=None,
        torsion_param=None,
        energy="None",
        mm_minimize=False,
        mm_constrain=True,
        mm_geometric=True,
        out_fname=None,
        collapse_atom_tuples=True,
    ):

        """
        Create torsiondrive summary plot of a parameter.
        Should be able to detect type of measurement from param type
            Downloads data
            Downloads min torsion molecules/grads if wbo
            Creates the appropriate ff-label command
            Creates the appropriate measure command from ff indices
            Aggregates data
            Sorts in a reasonable format
            Save as file (plot using something else)
        """

        import offsb.search.smiles
        import offsb.op.openmm
        import offsb.op.geometry

        valence_types = {
            "n": "vdW",
            "b": "Bonds",
            "a": "Angles",
            "t": "ProperTorsions",
            "i": "ImproperTorsions",
        }

        valence_measures = {
            "b": self.measure_bonds,
            "a": self.measure_angles,
            "t": self.measure_dihedrals,
            "i": self.measure_outofplanes,
        }

        convert_map = {
            "b": offsb.tools.const.bohr2angstrom,
            "a": 1.0,
            "t": 1.0,
            "i": 1.0,
        }

        warned = False
        warned_once = False

        entries = list(
            self.QCA.node_iter_torsiondrive_molecule_by_smarts(
                smarts=smarts, exact=exact
            )
        )
        print("The search on smarts", smarts, "produced", len(entries), "hits")

        labeler = None
        if ffname is not None:
            labeler = self.assign_labels_from_openff(ffname, ffname, targets=entries)

        # param_records = {}

        # param_list = list()

        # if param[1] != "*":
        #     if param not in param_list:
        #         param_list = [param]

        newdata = 0
        newdata += self.QCA.cache_torsiondriverecord_minimum_molecules(entries)

        bond_op = None
        torsion_op = None
        if labeler is not None:
            param_code = "b"

            convert = convert_map[param_code]

            valence_type = valence_types[param_code]

            if bond_param and bond_param not in labeler.db["ROOT"]["data"][valence_type]:
                raise Exception(
                    "Parameter {} not found in this set".format(bond_param)
                )

            measure_fn = valence_measures[param_code]

            bond_op = measure_fn(valence_type)
            bond_op.apply(targets=entries)

            param_code = "t"

            convert = convert_map[param_code]

            valence_type = valence_types[param_code]

            if torsion_param and torsion_param not in labeler.db["ROOT"]["data"][valence_type]:
                raise Exception(
                    "Parameter {} not found in this set".format(torsion_param)
                )

            measure_fn = valence_measures[param_code]

            torsion_op = measure_fn(valence_type)
            torsion_op.apply(targets=entries)

        omm_op = None
        if ffname is not None and energy != "None":
            omm_op = offsb.op.openmm.OpenMMEnergy(ffname, self.QCA, "oMM-" + ffname)
            omm_op.minimize = mm_minimize
            # Assume we have more than the number of cpus, so we can do each one
            # on a single processor
            omm_op.processes = None
            os.environ["OMP_NUM_THREADS"] = "1"
            omm_op.constrain = mm_constrain
            omm_op.geometric = mm_geometric
            print("Running OpenMM minimization")
            omm_op.apply(targets=entries)
            # self.save(omm_op)

        # bond_op, torsion_op, omm_op, labeler

        if wbo or mbo:
            newdata += self.QCA.cache_torsiondriverecord_minimum_gradients(entries)

        if newdata:
            self.save(self.QCA)

        ##### TIME TO GROUP #####
        
        fid = open("out.dat", "w")
        # smarts angle qmene torsion_label bond_label bond_measure (wbo) (mbo) (mmangle) (mmene) (bond_mmeasure)
        folders = self.QCA.combine_by_entry(targets=entries)
        for mol_nr, molecule in enumerate(folders, 1):
            for entry_nr, entry in enumerate(self.QCA.node_iter_entry(molecule), 1):

                if labeler is not None:
                    rdmol = offsb.rdutil.mol.build_from_smiles(
                        self.QCA.db[entry.payload]["data"].attributes[
                            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                        ]
                    )
                    map_inv = {v - 1: k for k, v in offsb.rdutil.mol.atom_map(rdmol).items()}



                for td in self.QCA.node_iter_depth_first(entry, select="TorsionDrive"):
                    qmenes = {}
                    mmenes = {}

                    min_id = None
                    qmmin = None
                    mmmin = None


                    # First loop exists only to find the minimum energy
                    for nmol in self.QCA.node_iter_torsiondriverecord_minimum(
                        td, select="Molecule"
                    ):
                        opt = self.QCA.db[
                            next(
                                self.QCA.node_iter_to_root(nmol, select="Optimization")
                            ).payload
                        ]

                        if omm_op and nmol.payload not in omm_op.db:
                            print("case 1")
                            continue
                        mol = None
                        if omm_op is not None:
                            mol = omm_op.db[nmol.payload]["data"]
                            if mol == []:
                                print("case 2")
                                continue
                        ene = opt["data"].energies[-1]
                        if qmmin is None or ene < qmmin:
                            qmmin = ene
                            min_id = nmol.payload
                        qmenes[nmol.payload] = ene

                        if (
                            omm_op is not None
                            and omm_op.db[nmol.payload]["data"]["energy"] is not None
                            and len(opt["data"].energies) > 0
                        ):

                            mmenes[nmol.payload] = omm_op.db[nmol.payload]["data"][
                                "energy"
                            ]

                        # print(qmenes, len(qmenes))
                        # print(mmenes, len(mmenes))
                        if omm_op is not None:
                            mmmin = omm_op.db[min_id]["data"]["energy"]

                    header = True
                    traj_fd = open("{:s}.{:s}.xyz".format(td.index, td.payload), 'w')
                    for nmol in self.QCA.node_iter_torsiondriverecord_minimum(
                        td, select="Molecule"
                    ):

                        if omm_op is not None and nmol.payload not in omm_op.db:
                            continue
                        mol = []
                        if omm_op is not None:
                            mol = omm_op.db[nmol.payload]["data"]

                            if mol == []:
                                continue
                        if (
                            omm_op is not None
                            and omm_op.db[nmol.payload]["data"]["energy"] is None
                        ):
                            continue
                        opt = self.QCA.db[
                            next(
                                self.QCA.node_iter_to_root(nmol, select="Optimization")
                            ).payload
                        ]
                        qene = opt["data"].energies[-1] - qmmin

                        constr = []
                        constraints = list(
                            self.QCA.node_iter_to_root(nmol, select="Constraint")
                        )
                        if len(constraints) > 0:
                            constr = list(constraints[0].payload[1])

                        if labeler is not None:
                            torsion_indices = tuple([map_inv[j] for j in constr])
                            torsion_label = labeler.db[entry.payload]['data']['ProperTorsions'][torsion_indices]

                            bond_indices = (torsion_indices[1], torsion_indices[2])
                            bond_label = labeler.db[entry.payload]['data']['Bonds'][bond_indices]

                        if issubclass(type(qene), simtk.unit.quantity.Quantity):
                            qene /= simtk.unit.kilocalories_per_mole
                        else:
                            qene *= offsb.tools.const.hartree2kcalmol

                        if omm_op is not None:
                            mene = omm_op.db[nmol.payload]["data"]["energy"] - mmmin

                            if issubclass(type(mene), simtk.unit.quantity.Quantity):
                                mene /= simtk.unit.kilocalories_per_mole

                        # mol1 = oMM1.db[nmol.payload]['data']
                        # if mol1 == []:
                        #     continue
                        qcmol = self.QCA.db[nmol.payload]["data"].dict()
                        if header:
                            print("#", entry, nmol, end="\n")
                            fid.write(
                                "# {:s} {:s}\n".format(entry.__repr__(), nmol.__repr__())
                            )
                            header = False

                        for i in [constr]:
                            if len(i) != 4:
                                continue
                            offsb.qcarchive.qcmol_to_xyz(
                                qcmol,
                                fd=traj_fd,
                                comment=json.dumps(i),
                            )
                            if wbo or mbo:
                                grad = self.QCA.db[self.QCA[nmol.parent].payload][
                                    "data"
                                ]
                            if wbo:
                                bo_name = "WIBERG_LOWDIN_INDICES"
                                try:
                                    wbo_vec = grad.extras["qcvars"][bo_name]
                                    wbo_val = wbo_vec[i[1] * int(len(wbo_vec) ** 0.5) + i[2]]
                                except Exception:
                                    wbo_val = np.nan
            
                            if mbo:
                                bo_name = "MAYER_INDICES"
                                try:
                                    mbo_vec = grad["extras"]["qcvars"][bo_name]
                                    mbo_val = mbo_vec[i[1] * int(len(mbo_vec) ** 0.5) + i[2]]
                                except Exception:
                                    mbo_val = np.nan


                            angle = offsb.op.geometry.TorsionOperation.measure_praxeolitic_single(
                                qcmol["geometry"], i
                            )

                            if omm_op is not None:
                                if "geometry" in mol:
                                    anglemin = offsb.op.geometry.TorsionOperation.measure_praxeolitic_single(
                                        mol["geometry"], i
                                    )
                                else:
                                    anglemin = angle

                                out_str = "{:d} {:d} {:3d}-{:3d}-{:3d}-{:3d} Q= {:10.4f} MM= {:10.4f} QE= {:10.4f} ME= {:10.4f}".format(
                                    mol_nr, entry_nr, *i, angle, anglemin, qene, mene
                                )
                                if wbo:
                                    out_str += " WBO= {:6.4f}".format(wbo_val)
                                if mbo:
                                    out_str += " MBO= {:6.4f}".format(mbo_val)
                                if labeler is not None:
                                    bond_val = bond_op.db[nmol.payload][bond_indices][0] * offsb.tools.const.bohr2angstrom
                                    out_str += " BOND= {:4s} {:8.4f}".format(bond_label, bond_val)
                                    torsion_val = torsion_op.db[nmol.payload][torsion_indices]
                                    out_str += " DIHE= {:4s} {:8.4f}".format(torsion_label, torsion_val)

                                out_str += "\n"

                                print(out_str, end="")
                                fid.write(out_str)
                            else:
                                out_str = "{:d} {:d} {:3d}-{:3d}-{:3d}-{:3d} Q= {:10.4f} QE= {:10.4f}".format(
                                    mol_nr, entry_nr, *i, angle, qene
                                )
                                if wbo:
                                    out_str += " WBO= {:6.4f}".format(wbo_val)
                                if mbo:
                                    out_str += " MBO= {:6.4f}".format(mbo_val)

                                if labeler is not None:
                                    bond_val = bond_op.db[nmol.payload][bond_indices][0] * offsb.tools.const.bohr2angstrom
                                    out_str += " BOND= {:4s} {:8.4f}".format(bond_label, bond_val)
                                    torsion_val = torsion_op.db[nmol.payload][torsion_indices]
                                    out_str += " DIHE= {:4s} {:8.4f}".format(torsion_label, torsion_val)
                                out_str += "\n"
                                print(out_str, end="")
                                fid.write(out_str)
                    traj_fd.close()

        fid.close()

        # for pi, param in enumerate(param_list):

        #     filename = ".".join([out_fname, param, "csv"])
        #     if out_fname is not None:
        #         f = open(filename, "w")
        #     else:
        #         f = sys.stdout

        #     records = []
        #     param_metadata = labeler.db["ROOT"]["data"][valence_type][param]

        #     param_keys = list(param_metadata.keys())
        #     for pkey in param_keys:
        #         value = param_metadata[pkey]
        #         if type(value) == simtk.unit.Quantity:
        #             value = str(value / value.unit) + " * " + str(value.unit)
        #         param_metadata[pkey] = value
        #     f.write("#JSON param {}\n".format(json.dumps(param_metadata)))

        #     f.write(
        #         '#JSON header {"col_names":["entry_id", "angle", "param_id", "value_in_AKMA"]}\n'
        #     )

        #     for entry_node in entries:
        #         td_list = self.QCA.node_iter_depth_first(
        #             entry_node, select="TorsionDrive"
        #         )

        #         for i, td_node in enumerate(td_list):

        #             key = entry_node.payload
        #             params = labeler.db[key]["data"]

        #             if valence_type not in params:
        #                 if not warned:
        #                     print(i, entry_node, "Missing labels")
        #                 warned_once = True
        #                 continue
        #             if param not in self._param_cache:
        #                 self._param_cache.append(param)

        #             all_params = params[valence_type]

        #             mol_nds = self.QCA.node_iter_torsiondriverecord_minimum(
        #                 entry_node, select="Molecule"
        #             )

        #             all_vals = {}
        #             for mol_node in mol_nds:

        #                 constr = [
        #                     x.payload
        #                     for x in self.QCA.node_iter_to_root(
        #                         mol_node, select="Constraint"
        #                     )
        #                 ]

        #                 if len(constr) > 1:
        #                     raise Exception("Only 1d torsionscans supported")

        #                 angle = constr[0][2]
        #                 mol_key = mol_node.payload

        #                 mol = self.QCA.db[mol_key]
        #                 op_vals = op.db[mol_key]

        #                 map_idx = {}
        #                 used_indices = {}
        #                 for atom_key, lbl in all_params.items():
        #                     if lbl != param:
        #                         continue
        #                     if collapse_atom_tuples:
        #                         if lbl not in used_indices:
        #                             used_indices[lbl] = {}
        #                             map_idx[lbl] = 0
        #                         if atom_key not in used_indices[lbl]:
        #                             used_indices[lbl][atom_key] = map_idx[lbl]
        #                             map_idx[lbl] += 1
        #                         atom_key_id = used_indices[lbl][atom_key]
        #                     else:
        #                         atom_key_id = atom_key

        #                     val = op_vals[atom_key][0] * convert
        #                     all_vals.setdefault(atom_key_id, []).append(val)
        #                     rec = [
        #                         int(td_node.payload.split("-")[1]),
        #                         angle,
        #                         atom_key_id,
        #                         val,
        #                     ]
        #                     # print(rec)
        #                     f.write("{:d} {:7.2f} {} {:12.8f}\n".format(*rec))
        #             if len(all_vals) > 0:
        #                 for key, vals in all_vals.items():
        #                     vals = np.array(vals)
        #                     rec = [
        #                         int(td_node.payload.split("-")[1]),
        #                         key,
        #                         vals.min(),
        #                         vals.mean(),
        #                         vals.max(),
        #                     ]
        #                     f.write(
        #                         "#STATS {:d} {} _±¯ {:12.8} {:12.8f} {:12.8f}\n".format(
        #                             *rec
        #                         )
        #                     )

        #         param_records[param] = records
        #         if f != sys.stdout:
        #             print(pi, len(param_list), "Wrote datafile for", param, filename)
        #             f.close()

        #         if warned_once:
        #             warned = True

        # return param_records

    def torsiondrive_groupby_openff_param(
        self, ffname, param, energy="None", out_fname=None, collapse_atom_tuples=True
    ):

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

        valence_types = {
            "n": "vdW",
            "b": "Bonds",
            "a": "Angles",
            "t": "ProperTorsions",
            "i": "ImproperTorsions",
        }

        valence_measures = {
            "b": self.measure_bonds,
            "a": self.measure_angles,
            "t": self.measure_dihedrals,
            "i": self.measure_outofplanes,
        }

        convert_map = {
            "b": offsb.tools.const.bohr2angstrom,
            "a": 1.0,
            "t": 1.0,
            "i": 1.0,
        }

        param_code = param[0]

        convert = convert_map[param_code]

        valence_type = valence_types[param_code]
        measure_fn = valence_measures[param_code]

        op = measure_fn(valence_type)
        labeler = self.assign_labels_from_openff(ffname, ffname)

        newmols = self.QCA.cache_torsiondriverecord_minimum_molecules()
        if newmols > 0:
            self.QCA.to_pickle()

        param_records = {}

        param_list = list(labeler.db["ROOT"]["data"][valence_type].keys())

        if param[1] != "*":
            if param not in param_list:
                raise Exception("Parameter not found")
            param_list = [param]

        warned = False
        warned_once = False

        if out_fname is None:
            out_fname = ffname

        for pi, param in enumerate(param_list):

            filename = ".".join([out_fname, param, "csv"])
            if out_fname is not None:
                f = open(filename, "w")
            else:
                f = sys.stdout

            records = []
            param_metadata = labeler.db["ROOT"]["data"][valence_type][param]

            param_keys = list(param_metadata.keys())
            for pkey in param_keys:
                value = param_metadata[pkey]
                if type(value) == simtk.unit.Quantity:
                    value = str(value / value.unit) + " * " + str(value.unit)
                if hasattr(value, "__iter__") and type(value) is not str:
                    str_value = []
                    for v in value:
                        if type(v) == simtk.unit.Quantity:
                            v = str(v / v.unit) + " * " + str(v.unit)
                        str_value.append(v)
                    value = str_value
                param_metadata[pkey] = value
            f.write("#JSON param {}\n".format(json.dumps(param_metadata)))

            f.write(
                '#JSON header {"col_names":["entry_id", "angle", "param_id", "value_in_AKMA"]}\n'
            )

            td_list = self.QCA.node_iter_depth_first(
                self.QCA.root(), select="TorsionDrive"
            )

            for i, td_node in enumerate(td_list):

                entry_node = next(self.QCA.node_iter_to_root(td_node, select="Entry"))

                key = entry_node.payload
                params = labeler.db[key]["data"]

                if valence_type not in params:
                    if not warned:
                        print(i, entry_node, "Missing labels")
                    warned_once = True
                    continue
                if param not in self._param_cache:
                    self._param_cache.append(param)

                all_params = params[valence_type]

                mol_nds = self.QCA.node_iter_torsiondriverecord_minimum(
                    entry_node, select="Molecule"
                )

                all_vals = {}
                for mol_node in mol_nds:

                    constr = [
                        x.payload
                        for x in self.QCA.node_iter_to_root(
                            mol_node, select="Constraint"
                        )
                    ]

                    if len(constr) > 1:
                        raise Exception("Only 1d torsionscans supported")

                    angle = constr[0][2]
                    mol_key = mol_node.payload

                    mol = self.QCA.db[mol_key]
                    op_vals = op.db[mol_key]

                    map_idx = {}
                    used_indices = {}
                    for atom_key, lbl in all_params.items():
                        if lbl != param:
                            continue
                        if collapse_atom_tuples:
                            if lbl not in used_indices:
                                used_indices[lbl] = {}
                                map_idx[lbl] = 0
                            if atom_key not in used_indices[lbl]:
                                used_indices[lbl][atom_key] = map_idx[lbl]
                                map_idx[lbl] += 1
                            atom_key_id = used_indices[lbl][atom_key]
                        else:
                            atom_key_id = atom_key

                        val = op_vals[atom_key] 
                        try:
                            val = val[0]
                        except IndexError:
                            pass
                        val = val * convert

                        all_vals.setdefault(atom_key_id, []).append(val)
                        rec = [
                            int(td_node.payload.split("-")[1]),
                            angle,
                            atom_key_id,
                            val,
                        ]
                        # print(rec)
                        f.write("{:d} {:7.2f} {} {:12.8f}\n".format(*rec))
                if len(all_vals) > 0:
                    for key, vals in all_vals.items():
                        vals = np.array(vals)
                        rec = [
                            int(td_node.payload.split("-")[1]),
                            key,
                            vals.min(),
                            vals.mean(),
                            vals.max(),
                        ]
                        f.write(
                            "#STATS {:d} {} _±¯ {:12.8} {:12.8f} {:12.8f}\n".format(
                                *rec
                            )
                        )

            param_records[param] = records
            if f != sys.stdout:
                print(pi, len(param_list), "Wrote datafile for", param, filename)
                f.close()

            if warned_once:
                warned = True

        return param_records

    # def collate_1d_torsiondrive(self, entry_data=[], molecule_data=[],
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

    def assign_labels_from_openff(self, openff_name, name, targets=None):

        if os.path.exists(name + ".p"):
            return pickle.load(open(name + ".p", "rb"))

        from offsb.op.openforcefield import OpenForceFieldTree as OP

        ext = ".offxml"
        if not openff_name.endswith(ext):
            openff_name += ext
        labeler = OP(self.QCA, name, openff_name)
        labeler.apply(targets=targets)
        labeler.to_pickle()
        return labeler

    def _measure(self, smi, op_cls, name, targets=None):
        from offsb.search.smiles import SmilesSearchTree

        if os.path.exists(name + ".p"):
            return pickle.load(open(name + ".p", "rb"))

        # assume we want all final geometries
        if len(list(self.QCA.node_iter_depth_first(self.QCA.root(), select="Molecule"))) == 0:
            self.QCA.cache_optimization_minimum_molecules(targets)

        query = SmilesSearchTree(smi, self.QCA, "query")
        query.apply(targets=targets)

        op = op_cls(query, name)
        op.apply(targets=targets)
        op.to_pickle()
        return op

    def measure_bonds(self, name, targets=None):
        from offsb.op.geometry import BondOperation as OP

        return self._measure("[*]~[*]", OP, name, targets=targets)

    def measure_angles(self, name, targets=None):
        from offsb.op.geometry import AngleOperation as OP

        return self._measure("[*]~[*]~[*]", OP, name, targets=targets)

    def measure_dihedrals(self, name, targets=None):
        from offsb.op.geometry import TorsionOperation as OP

        return self._measure("[*]~[*]~[*]~[*]", OP, name, targets=targets)

    def measure_outofplanes(self, name, targets=None):
        from offsb.op.geometry import ImproperTorsionOperation as OP

        return self._measure("[*]~[*](~[*])~[*]", OP, name, targets=targets)

    def error_report_per_dataset(self, save_xyz=True, out_fnm=None, full_report=False):
        if out_fnm is None:
            fid = sys.stdout
        else:
            fname = out_fnm
            fid = open(fname, "w")
        QCA = self.QCA
        for ds_id in QCA.root().children:
            ds_node = QCA[ds_id]
            client = QCA.db[ds_node.payload]['data'].client
            fid.write("==== DATASET ==== {}\n".format(ds_node.name))
            for entry in QCA.iter_entry():
                specs = [
                    n for n in QCA.node_iter_depth_first(entry) if "QCS" in n.payload
                ]
                for spec in specs:
                    mindepth = QCA.node_depth(spec)
                    tdi = 0
                    for node in QCA.node_iter_dive(spec):
                        i = QCA.node_depth(node) - mindepth + 1
                        status = ""
                        statbar = "    "
                        errmsg = ""
                        hasstat = False

                        try:
                            # hasstat = "status" in QCA.db[node.payload]["data"]
                            hasstat = hasattr(QCA.db[node.payload]["data"], "status")
                        except Exception:
                            pass
                        if hasstat:
                            qcp = QCA.db[node.payload]["data"].dict()
                            status = qcp["status"][:]
                            if status == "ERROR" and node.name == "Optimization":
                                try:
                                    statbar = "----"
                                    # print("error", qcp['error'], qcp['error'] is None)

                                    errtype = ""
                                    xyzerrmsg = ""
                                    if not (qcp["error"] is None):
                                        err = json.loads(
                                            list(
                                                client.query_kvstore(
                                                    int(qcp["error"])
                                                ).values()
                                            )[0]
                                            .dict()["data"]
                                            .decode()
                                        )
                                        errmsg += "####ERROR####\n"
                                        errmsg += "Error type: " + err["error_type"] + "\n"
                                        errmsg += "Error:\n" + err["error_message"]
                                        errmsg += "############\n"
                                        if (
                                            "RuntimeError: Not bracketed"
                                            in err["error_message"]
                                        ):
                                            errtype = "bracketed"
                                        elif (
                                            "Cannot continue a constrained optimization; please implement constrained optimization in Cartesian coordinates"
                                            in err["error_message"]
                                        ):
                                            errtype = "cartconstr"
                                        elif (
                                            "numpy.linalg.LinAlgError: Eigenvalues did not converge"
                                            in err["error_message"]
                                        ):
                                            errtype = "numpy-eigh"
                                        elif (
                                            "geometric.errors.GeomOptNotConvergedError: Optimizer.optimizeGeometry() failed to converge."
                                            in err["error_message"]
                                        ):
                                            errtype = "optconverge"
                                        elif (
                                            "RuntimeError: Unsuccessful run. Possibly -D variant not available in dftd3 version."
                                            in err["error_message"]
                                        ):
                                            errtype = "dftd3variant"
                                        elif (
                                            "Could not converge SCF iterations"
                                            in err["error_message"]
                                        ):
                                            errtype = "scfconv"
                                        elif (
                                            "distributed.scheduler.KilledWorker"
                                            in err["error_message"]
                                        ):
                                            errtype = "needsrestart-daskkilled"
                                        elif (
                                            "struct.error: unpack requires a buffer of"
                                            in err["error_message"]
                                        ):
                                            errtype = "needsrestart-anistructunpack"
                                        elif (
                                            "concurrent.futures.process.BrokenProcessPool"
                                            in err["error_message"]
                                        ):
                                            errtype = "needsrestart-brokenpool"
                                        elif (
                                            len(err["error_message"].strip().strip("\n"))
                                            == 0
                                        ):
                                            errtype = "emptyerror"
                                        else:
                                            errtype = "nocategory"

                                        for line in err["error_message"].split("\n")[::-1]:
                                            if len(line) > 0:
                                                xyzerrmsg = line
                                                break

                                    elif status == "ERROR":
                                        errmsg += "####FATAL####\n"
                                        errmsg += "Job errored and no error on record. Input likely bogus; check your settings\n"
                                        errmsg += "############\n"
                                        errtype = "bogus"
                                        xyzerrmsg = "Job errored and no error on record. Input likely bogus; check your settings"

                                    if len(errtype) > 0:
                                        traj = list(
                                            QCA.node_iter_depth_first(
                                                node, select="Molecule"
                                            )
                                        )
                                        metadata = ""
                                        tdids = [
                                            n.payload
                                            for n in QCA.node_iter_to_root(
                                                node, select="TorsionDrive"
                                            )
                                        ]

                                        if len(tdids) > 0:
                                            metadata += "TdIDs " + ".".join(tdids) + " "

                                        contrs = [
                                            str(n.payload)
                                            for n in QCA.node_iter_to_root(
                                                node, select="Constraint"
                                            )
                                        ]
                                        if len(contrs) > 0:
                                            metadata += (
                                                "Constraints " + ".".join(contrs) + " "
                                            )

                                        tdname = ".".join(tdids)
                                        if len(tdname) > 0:
                                            tdname += "."
                                        if len(traj) > 0 and save_xyz:
                                            fname = (
                                                "geometric."
                                                + errtype
                                                + ".traj."
                                                + tdname
                                                + node.payload
                                                + "."
                                                + traj[-1].payload
                                                + ".xyz"
                                            )
                                            # print("Trajectory found... saving", fname)
                                            mol = QCA.db[traj[-1].payload]["data"].dict()
                                            with open(fname, "w") as xyzfid:
                                                _ = offsb.qcarchive.qcmol_to_xyz(
                                                    mol,
                                                    fd=xyzfid,
                                                    comment=metadata
                                                    + "OptID "
                                                    + node.payload
                                                    + " MoleculeID "
                                                    + traj[-1].payload
                                                    + " Error= "
                                                    + xyzerrmsg,
                                                )
                                        else:
                                            mol_id = "QCM-" + qcp["initial_molecule"]
                                            fname = (
                                                "geometric."
                                                + errtype
                                                + ".initial."
                                                + tdname
                                                + node.payload
                                                + "."
                                                + mol_id
                                                + ".xyz"
                                            )
                                            # print("Trajectory not found... saving input molecule", fname)
                                            mol = QCA.db[mol_id]["data"].dict()
                                            if "geometry" in mol and "symbols" in mol and save_xyz:
                                                with open(fname, "w") as xyzfid:
                                                    _ = offsb.qcarchive.qcmol_to_xyz(
                                                        mol,
                                                        fd=xyzfid,
                                                        comment=metadata
                                                        + "OptID "
                                                        + node.payload
                                                        + " MoleculeID "
                                                        + mol_id
                                                        + " Error= "
                                                        + xyzerrmsg,
                                                    )
                                            else:
                                                # print("Initial molecule missing!")
                                                errmsg += "XXXXISSUEXXXX\n"
                                                errmsg += "Initial molecule was missing!\n"
                                                errmsg += "xxxxxxxxxxxxx\n"
                                    if not (qcp["stdout"] is None):
                                        msg = list(
                                            client.query_kvstore(
                                                int(qcp["stdout"])
                                            ).values()
                                        )[0]
                                        errmsg += "xxxxISSUExxxx\n"
                                        errmsg += "Status was not complete; stdout is:\n"
                                        errmsg += msg
                                        errmsg += "xxxxxxxxxxxxx\n"
                                    if not (qcp["stderr"] is None):
                                        msg = list(
                                            client.query_kvstore(
                                                int(qcp["stderr"])
                                            ).values()
                                        )[0]
                                        errmsg += "####ISSUE####\n"
                                        errmsg += "Status was not complete; stderr is:\n"
                                        errmsg += msg
                                        errmsg += "#############\n"
                                except Exception as e:
                                    fid.write(
                                        "Internal issue:\n"
                                        + str(type(e))
                                        + "\n"
                                        + str(e)
                                        + "\n"
                                    )

                            if status != "COMPLETE" and node.name == "TorsionDrive":
                                statbar = "XXXX"
                        tderror = False
                        if node.name == "TorsionDrive" and status != "COMPLETE":
                            tdi += 1
                            tderror = True
                        if full_report or (len(errmsg) > 0 or tderror):
                            out_str = "{:2d} {} {} {}\n".format(
                                tdi,
                                statbar * i,
                                " ".join(
                                    [str(node.index), str(node.name), str(node.payload)]
                                ),
                                status,
                            )
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
    parser.add_argument("--report-errors", action="store_true")
    parser.add_argument("--report-energy", action="store_true")
    parser.add_argument("--datasets", type=str)
    args = parser.parse_args()

    if args.datasets is not None:
        datasets = load_dataset_input(args.datasets)

    qcasb = None
    if args.report_energy:
        qcasb = QCArchiveSpellBook(datasets=args.datasets)
        enes_per_mol = qcasb.energy_minimum_per_molecule()
        with open("enes.txt", "w") as fid:
            for cmiles, enes in enes_per_mol.items():
                fid.write("# {:s}\n".format(cmiles))
                for ene in enes:
                    fid.write("    {:12.8f}\n".format(ene))
    if args.report_errors:
        if qcasb is None:
            qcasb = QCArchiveSpellBook(datasets=args.datasets)
        qcasb.error_report_per_dataset()
