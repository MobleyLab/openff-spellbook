#!/usr/bin/env python3

import gzip
import json
import logging
import os
import shutil
from collections import OrderedDict
from io import BytesIO

import numpy as np

import forcebalance.forcefield
import forcebalance.target
import offsb.dev.hessian
import offsb.op.chemper
import offsb.op.openforcefield
import offsb.rdutil
import offsb.search.smiles
import offsb.tools.util
import offsb.treedi.tree
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer
from forcebalance.parser import parse_inputs
from openbabel import openbabel
from openforcefield.typing.engines.smirnoff.forcefield import ForceField
from openforcefield.typing.engines.smirnoff.parameters import (ImproperDict,
                                                               ValenceDict)
from rdkit import Chem

np.set_printoptions(linewidth=9999, formatter={"float_kind": "{:12.6e}".format})


class DummyTree:
    source = None
    ID = None


class ForceBalanceObjectiveOptGeo(offsb.treedi.tree.TreeOperation):
    def __init__(
        self, fbinput_fname, source_tree, name, ff_fname, init=None, verbose=True
    ):
        super().__init__(source_tree, name)
        import logging

        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.ERROR)

        self._select = "Molecule"

        if init is None:
            self._setup = ForceBalanceObjectiveOptGeoSetup(
                fbinput_fname,
                source_tree,
                "fb_setup." + name,
                ff_fname,
                "optimize",
                verbose=verbose,
            )

        else:
            self._setup = init
        self._init = False

        self.fbinput_fname = fbinput_fname
        self.ff_fname = ff_fname

        self.source = DummyTree

        self.cleanup = True
        self._options = None
        self._tgt_opts = None

        self.options_override = {}

        self.fitting_targets = ["geometry", "energy", "vibration"]
        DummyTree.source = source_tree

        self.cwd = os.path.abspath(os.path.curdir)

        self.X = None
        self.G = None
        self.H = None

    def _unpack_result(self, ret):
        self.db.update(ret)

    def _generate_apply_kwargs(self, i, target, kwargs=None):

        if kwargs is None:
            kwargs = {}
        arg = np.zeros(self._forcefield.np)
        kwargs["arg"] = arg
        found = False
        for tgt in self._objective.Targets:
            for key in tgt.internal_coordinates:
                if target.payload in key:
                    kwargs["tgt"] = tgt
                    found = True
                    break
            if found:
                break
        # kwargs["tgt"] = [
        #     tgt
        #     for tgt in self._objective.Targets
        #     if opt.payload+'.'+target.payload in tgt.internal_coordinates
        # ][0]
        return kwargs

    def op(self, mask):
        pass

    def to_pickle(self):
        ff, self._forcefield = self._forcefield, None
        obj, self._objective = self._objective, None

        super().to_pickle()

        self._forcefield = ff
        self._objective = obj

    # def optimize(self, targets=None, parameterize_handlers=None, jobtype="OPTIMIZE"):

    #     if parameterize_handlers is None:
    #         parameterize_handlers = self.parameterize_handlers

    #     if self._init == False:
    #         self._setup.apply(
    #             targets=targets,
    #             parameterize_handlers=parameterize_handlers,
    #             fitting_targets=["geometry"],
    #         )
    #         self._init = True
    #     else:
    #         self.remove_tmp(clean_input=False)

    #     if self._options is None or self._tgt_opts is None:
    #         self._options, self._tgt_opts = parse_inputs("optimize.in")

    #     self._options["jobtype"] = jobtype

    #     self._forcefield = forcebalance.forcefield.FF(self._options)
    #     self.db.clear()
    #     self.db["ROOT"] = {"data": self._forcefield.plist}

    #     objective = Objective(self._options, self._tgt_opts, self._forcefield)
    #     optimizer = Optimizer(self._options, objective, self._forcefield)

    #     fb_logger = logging.getLogger("forcebalance")
    #     fb_logger.setLevel(logging.INFO)
    #     ans = optimizer.Run()

    #     new_ff = self._setup.prefix + ".offxml"
    #     self.new_ff = None
    #     new_ff_path = os.path.join("result", self._setup.prefix, new_ff)
    #     if os.path.exists(new_ff_path):
    #         self.new_ff = ForceField(new_ff_path)

    def load_options(self, options_override=None):
        # The general options and target options that come from parsing the input file
        if os.path.exists("optimize.in"):
            self._options, self._tgt_opts = parse_inputs("optimize.in")
            # print("Current options overridden")
            # for k,v in self.options_override.items():
            #     print(k,v)
            self._options.update(self.options_override)
            if options_override is not None:
                # print("Overriding new options:")
                # for k,v in options_override.items():
                #     print(k,v)
                self._options.update(options_override)
            # self.options_override = None
        elif options_override is not None:
            self.options_override.update(options_override)

    def apply(
        self,
        targets=None,
        parameterize_handlers=None,
        fitting_targets=None,
        jobtype="GRADIENT",
    ):

        os.chdir(self.cwd)

        if parameterize_handlers is None:
            parameterize_handlers = self.parameterize_handlers

        if fitting_targets is None:
            fitting_targets = self.fitting_targets

        if self._init is False:
            self._setup.apply(
                targets=targets,
                parameterize_handlers=parameterize_handlers,
                fitting_targets=fitting_targets,
            )
            self._init = True
        else:
            self.remove_tmp(clean_input=False)

        lvl = self.logger.getEffectiveLevel()
        fb_logger = logging.getLogger("forcebalance")
        fb_logger.setLevel(lvl)

        if self._options is None or self._tgt_opts is None:
            self.load_options()

        # if self.options_override is not None:
        #     self._options.update(self.options_override)
        #     self.options_override = None

        self._options["jobtype"] = jobtype

        self._forcefield = forcebalance.forcefield.FF(self._options)

        # Because ForceBalance Targets contain unpicklable objects, we must
        # use single process
        # Use the work_queue implementation instead
        self.processes = 1
        opts = self._options.copy()

        self.db.clear()

        self.db["ROOT"] = {"data": self._forcefield.plist}

        # remote = opts.get("asynchronous", False)
        self._objective = Objective(opts, self._tgt_opts, self._forcefield)

        optimizer = Optimizer(self._options, self._objective, self._forcefield)

        fb_logger = logging.getLogger("forcebalance")
        # turning to info here, or else FB goes silent
        fb_logger.setLevel(logging.INFO)
        ans = optimizer.Run()


        best = optimizer.BestChk


        if best is None:
            print("best was None! breaking")
            breakpoint()

        self._options["trust0"] = best["trust"]
        self._options["finite_difference_h"] = best["finite_difference_h"]
        self._options["eig_lowerbound"] = best["eig_lowerbound"]

        print("FB: set options to")
        print("trust0:", self._options["trust0"])
        print("finite_difference_h:", self._options["finite_difference_h"])
        print("eig_lowerbound:", self._options["eig_lowerbound"])

        xk = best.get('xk')
        if xk is not None:
            optimizer.FF.make(xk, printdir=optimizer.resdir)
        
        self.X = 0
        for tgt, dat in self._objective.ObjDict.items():
            rec = tgt.split(".")[-1]

            # lets skip this since we use the unpenalized for assessing new
            # splits, which always have a 0 penalty (until I wrote code to
            # incorporate the old penalty, which will be weird since usually
            # a new parameter exists and it is unclear how to treat it
            # if tgt in "Regularization":
            #     self.X += dat['w'] * dat["x"]

            # skip known keys that we must skip
            if tgt in ["Total", "Regularization"]:
                continue

            self.X += dat["w"] * dat["x"]

            IC = dat.get("IC", None)
            if IC is not None:

                IC = dat["IC"][rec]
                # node = [ x for x in QCA.node_iter_depth_first(QCA.root()) if x.payload == rec ][0]

                ret = self._generate_ic_objective_pairs(IC, dat)
                dat = dat.copy()
                dat.update(ret)

            if rec not in self.db:
                self.db[rec] = {"data": dat}
            else:
                self.db[rec]["data"].update(dat)

        # might be safer?
        self.X = best["X"]

        dv = None
        for mol in self.db.values():
            mol = mol["data"]
            for k in mol:
                if type(k) is tuple and "dV" in mol[k]:
                    if dv is None:
                        dv = mol["w"] * mol[k]["dV"]
                    else:
                        dv += mol["w"] * mol[k]["dV"]

        # also safer since above only works for optgeo at the moment
        self.G = np.linalg.norm(best["G"])

        # if not remote:
        #     super().apply(self._select, targets=targets)
        # else:
        #     logging.getLogger("forcebalance").setLevel(logging.INFO)

        #     # QCA = self.source.source
        #     port = opts["wq_port"]
        #     print("Starting WorkQueue... please start a worker and set to port", port)
        #     arg = np.zeros(len(self._forcefield.plist))
        #     self._objective.Full(arg, Order=1, verbose=1)
        #     for tgt, dat in self._objective.ObjDict.items():
        #         rec = tgt.split(".")[-1]

        #         # skip known keys that we must skip
        #         if tgt in ["Total", "Regularization"]:
        #             continue

        #         IC = dat["IC"][rec]
        #         # node = [ x for x in QCA.node_iter_depth_first(QCA.root()) if x.payload == rec ][0]

        #         ret = self._generate_ic_objective_pairs(IC, dat)
        #         dat = dat.copy()
        #         dat.update(ret)

        #         if rec not in self.db:
        #             self.db[rec] = {"data": dat}
        #         else:
        #             self.db[rec]["data"].update(dat)
        logging.getLogger("forcebalance").setLevel(logging.WARN)

        new_ff = self._setup.prefix + ".offxml"
        self.new_ff = None
        new_ff_path = os.path.join("result", self._setup.prefix, new_ff)
        if os.path.exists(new_ff_path):
            # print("Loading new FF from optimizer:", new_ff_path)
            self.new_ff = ForceField(new_ff_path, allow_cosmetic_attributes=True)
        else:
            # print("Loading new FF from optimizer:", self.ff_fname)
            self.new_ff = ForceField(self.ff_fname, allow_cosmetic_attributes=True)

        if self.cleanup:
            self.remove_tmp()

    def remove_tmp(self, clean_input=False):
        for folder in ["optimize.tmp", "optimize.bak"]:
            try:
                shutil.rmtree(folder)
            except FileNotFoundError:
                pass

        fnames = ["smirnoff_parameter_assignments.json"]
        if clean_input:
            fnames.extend([self._setup.prefix + ".offxml", self._setup.prefix + ".in"])
        for fname in fnames:
            if os.path.exists(fname):
                os.remove(fname)

    def _generate_ic_objective_pairs(self, IC_dict, ans, i=None):

        ret = {}

        if i is None:
            i = 0

        IC = [
            ic
            for ic_type in [
                "ic_bonds",
                "ic_angles",
                "ic_dihedrals",
                "ic_impropers",
            ]
            for ic in IC_dict[ic_type]
        ]
        for ic, v in zip(IC, ans["V"][i:]):

            indices = "-".join(
                map(
                    str,
                    [
                        getattr(ic, letter)
                        for letter in ["a", "b", "c", "d"]
                        if hasattr(ic, letter)
                    ],
                )
            )
            integer_indices = tuple(map(int, indices.split("-")))
            ic_name = str(ic).split()[0]

            if ic_name == "OutOfPlane":
                integer_indices = ImproperDict.key_transform(integer_indices)
            else:
                integer_indices = ValenceDict.key_transform(integer_indices)

            vals = 2 * v * ans["dV"][:, i]
            ret[integer_indices] = {"V": v ** 2, "dV": vals}
            i += 1
        return ret

    def apply_single(self, i, target, **kwargs):

        tgt = kwargs.get("tgt", None)
        if tgt is None:
            return {target.payload: "", "return": None}
        arg = kwargs["arg"]

        ans = tgt.get(arg, 1, 1)
        ret_obj = {}
        i = 0

        for sys_name, sys in tgt.internal_coordinates.items():

            dat = self._generate_ic_objective_pairs(sys, ans, i)
            ret_obj[sys_name] = {"data": dat}
            i += len(dat)

        return {target.payload: "", "return": ret_obj}


class ForceBalanceObjectiveTorsionDriveSetup(offsb.treedi.tree.TreeOperation):

    """
    This class will iterate over the data and only
    1. generate the appropriate target options for each during apply (should be automatic)
    """

    def __init__(self, fbinput_fname, source_tree, name, ff_fname, prefix="optimize"):
        super().__init__(source_tree, name)
        import logging

        self._select = "TorsionDrive"
        self.prefix = prefix

        self.global_opts, _ = parse_inputs(fbinput_fname)

        self.fbinput_fname = fbinput_fname
        self.ff_fname = ff_fname

        self.td_denom = 1.0
        self.td_upper = 5.0
        self.restrain_k = 0.0

        self.parameterize_handlers = None

        self.processes = 1

        self.source = DummyTree
        DummyTree.source = source_tree
        print("My db is", self.db)

        # The FB source also hardcodes a "qdata.txt" file, which
        # reads the QM reference energy. Specifically it appears to grep the
        # file such that it sees
        # ENERGY 1.34
        # ENERGY 1.24
        # ENERGY 1.55
        # etc etc and then pulls the energy. In the reference (the official oFF
        # fits), the xyz and gradients are therefore not used
        self.fb_main_opts = (
            "\n"
            + "\n$target"
            + "\nname {dir:s}"
            + "\ntype TorsionProfile_SMIRNOFF"
            + "\npdb mol.pdb"
            + "\nmol2 mol.sdf"
            + "\ncoords scan.xyz"
            + "\nwritelevel 0"
            + "\nattenuate"
            + "\nenergy_denom {denom:f}"
            + "\nenergy_upper {upper:f}"
            + "\nrestrain_k {restrain_k:f}"
            + "\nremote 1"
            + "\n$end"
        )

        self.fb_tgt_opts = ""

    def _apply_initialize(self, targets):
        pass

    def _apply_finalize(self, targets):
        pass

    def op(self, node, partition):
        pass

    def _unpack_result(self, ret):

        for k, v in ret.items():
            v = v["data"]
            dir = os.path.join("targets", v["dir"])
            try:
                os.mkdir(dir)
            except FileExistsError:
                pass
            path = os.path.join(dir, "mol")
            for ext in ["pdb", "sdf"]:
                out_str = v[ext]
                mode = "wb" if ext.endswith("gz") else "wt"
                with open(path + "." + ext, mode) as f:
                    f.write(out_str)

        if len(ret) > 0:
            self.db.update(ret)

    def generate_targets(self):
        """
        for each of the types, generate a target
        """

        pass

    def parse_output(self):
        """
        take the results from a FB optimization, and gather the data for each
        """
        pass

    def _generate_apply_kwargs(self, i, target, kwargs=None):

        """
        get the entry
        really really want to try doing optgeo for everything
        Lets go for it :) this means it will generate a target for every molecule in the opt
        So we need make a target with a number id for each target xyz
        but maybe we make a target folder for each so we can apply a specific weight (likely a energy weighted)

        """

        QCA = self.source.source

        # # labels = self.source.db[target.payload]["data"]
        tdr = QCA.db[target.payload]["data"]
        entry = next(QCA.node_iter_to_root(target, select="Entry", dereference=True))

        # out_str = ""

        if kwargs is None:
            kwargs = {}

        """
        need to go into an entry, grab all molecules
        save coordinates and energy
        """

        smi = entry.attributes["canonical_isomeric_explicit_hydrogen_mapped_smiles"]

        ref_mol = QCA.db["QCM-" + list(tdr.initial_molecule)[0]]["data"]
        mol = offsb.rdutil.mol.rdmol_from_smiles_and_qcmol(smi, ref_mol)

        map_idx = offsb.rdutil.mol.atom_map(mol)
        map_inv = offsb.rdutil.mol.atom_map_invert(map_idx)

        dir = "targets/" + target.payload

        try:
            os.mkdir(dir)
        except FileExistsError:
            pass

        # This is a weird one: write the scan and the energy now, to avoid
        # the unnecessary send to multiprocess
        # This means half of the target is written now, half later (in unpack)
        scan_fid = open(os.path.join(dir, "scan.xyz"), "w")
        ene_fid = open(os.path.join(dir, "qdata.txt"), "w")

        grid = []

        for i, opt_node in enumerate(
            QCA.node_iter_torsiondriverecord_minimum(target, select="Optimization")
        ):

            opt = QCA.db[opt_node.payload]["data"]
            mol_node = list(QCA.node_iter_depth_first(opt_node, select="Molecule"))[0]

            constraints = list(QCA.node_iter_to_root(opt_node, select="Constraint"))[
                ::-1
            ]
            constraints = [x.payload[2] for x in constraints]
            grid.append(constraints)

            energies = opt.energies
            energy = energies[-1]

            qcmol = self.source.source.db.get(mol_node.payload).get("data")
            offsb.qcarchive.qcmol_to_xyz(
                qcmol, fd=scan_fid, atom_map={k: v - 1 for k, v in map_idx.items()}
            )
            ene_fid.write("ENERGY {:16.12f}\n".format(energy))

        # Since the mol2 and pdb writers use canonical ordering, we need to unmap
        # the QCA ordering

        dihedrals = entry.td_keywords.dihedrals
        for i, _ in enumerate(dihedrals):
            dihedrals[i] = [map_inv[j] for j in dihedrals[i]]

        metadata = {"dihedrals": dihedrals, "torsion_grid_ids": grid}

        with open(os.path.join(dir, "metadata.json"), "w") as metadata_fid:
            json.dump(metadata, metadata_fid)

        scan_fid.close()
        ene_fid.close()

        if len(grid) == 0:
            kwargs["error"] = "This torsiondrive empty! ID {:s}".format(target.payload)
            shutil.rmtree(dir)

        kwargs["smi"] = smi
        kwargs["cwd"] = target.payload
        kwargs["global"] = self.fb_main_opts
        kwargs["mol"] = ref_mol
        kwargs["metadata"] = metadata
        kwargs["map_inv"] = map_inv

        return kwargs

    def apply_single_target_objective():
        pass

    def apply_single(self, i, target, **kwargs):

        if "error" in kwargs:
            return {target.payload: kwargs["error"], "return": {}}

        smi = kwargs["smi"]
        qcmol = kwargs["mol"]
        dir = kwargs["cwd"]
        fb_main_opts = kwargs["global"]

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "sdf")
        obmol = openbabel.OBMol()

        mol = offsb.rdutil.mol.rdmol_from_smiles_and_qcmol(smi, qcmol)

        # doesn't write connectivity... sigh
        # sdf_str = StringIO()
        # writer = Chem.SDWriter(sdf_str)
        # writer.write(mol)
        # sdf_str.seek(0)
        # sdf_str = sdf_str.getvalue()

        pdb_str = Chem.MolToPDBBlock(mol)
        obConversion.ReadString(obmol, pdb_str)
        sdf_str = obConversion.WriteString(obmol)

        # ForceBalance does not detect compressed PDB (does an extension check for format)
        # with BytesIO() as bio:
        #     with gzip.GzipFile(fileobj=bio, mode='w') as fid:
        #         fid.write(pdb_str.encode())
        #     pdb_str = bio.getvalue()

        # with BytesIO() as bio:
        #     with gzip.GzipFile(fileobj=bio, mode="w") as fid:
        #         fid.write(sdf_str.encode())
        #     sdf_str = bio.getvalue()

        ret_obj = {
            dir: {
                "data": {
                    "dir": dir,
                    "pdb": pdb_str,
                    "sdf": sdf_str,
                    "global": fb_main_opts.format(
                        dir=dir,
                        denom=self.td_denom,
                        upper=self.td_upper,
                        restrain_k=self.restrain_k,
                    ),
                }
            }
        }
        return {target.payload: "", "return": ret_obj}

    def apply(
        self,
        targets=None,
        parameterize_handlers=None,
        fitting_targets=["geometry"],
        parameterize_terms=None,
        parameterize_spatial=True,
        parameterize_force=True,
    ):
        if parameterize_handlers is None:
            parameterize_handlers = self.parameterize_handlers

        lvl = self.logger.getEffectiveLevel()
        fb_logger = logging.getLogger("forcebalance")
        fb_logger.setLevel(lvl)
        off_ph_logger = logging.getLogger("openforcefield")
        off_ph_logger.setLevel(lvl)
        rdkit_logger = logging.getLogger("rdkit")
        rdkit_logger.setLevel(lvl)

        for folder in ["optimize.tmp", "optimize.bak", "result"]:
            try:
                shutil.rmtree(folder)
            except FileNotFoundError:
                pass

        if "geometry" in fitting_targets:
            self._optgeo = True
        if "energy" in fitting_targets:
            self._abinitio = True
        if "vibration" in fitting_targets:
            self._vibration = True
        try:
            os.mkdir("targets")
        except Exception:
            pass

        labeler = None

        if parameterize_handlers is None:
            parameterize_handlers = [
                "vdW",
                "Bonds",
                "Angles",
                "ProperTorsions",
                "ImproperTorsions",
            ]

        ff_kwargs = dict(allow_cosmetic_attributes=True)
        if len(parameterize_handlers) > 1:
            labeler = offsb.op.openforcefield.OpenForceFieldTree(
                self.source.source, "ff", self.ff_fname, ff_kwargs=ff_kwargs
            )
        elif len(parameterize_handlers) == 1:
            if "vdW" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldvdWTree(
                    self.source.source, "ff", self.ff_fname, ff_kwargs=ff_kwargs
                )
            elif "Bonds" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldBondTree(
                    self.source.source, "ff", self.ff_fname, ff_kwargs=ff_kwargs
                )
            elif "Angles" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldAngleTree(
                    self.source.source, "ff", self.ff_fname, ff_kwargs=ff_kwargs
                )
            elif "ProperTorsions" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldTorsionTree(
                    self.source.source, "ff", self.ff_fname, ff_kwargs=ff_kwargs
                )
            elif "ImproperTorsions" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldImproperTorsionTree(
                    self.source.source, "ff", self.ff_fname, ff_kwargs=ff_kwargs
                )
            else:
                raise Exception(
                    "Parameter handler" + parameterize_handlers[0] + "not supported"
                )

        labeler.processes = 1
        labeler.apply()

        # export the FF
        args = (self.prefix + ".offxml",)

        # default case if we don't want any parameters fit, i.e. no "cosmetic attributes"
        kwargs = dict(
            parameterize_handlers=parameterize_handlers,
            parameterize_spatial=parameterize_spatial,
            parameterize_force=parameterize_force,
            parameterize_terms=parameterize_terms,
        )

        # generate parameter fits to multiple handlers
        if len(parameterize_handlers) >= 1:
            kwargs.update(
                dict(
                    parameterize_spatial=parameterize_spatial,
                    parameterize_force=parameterize_force,
                )
            )
        if len(parameterize_handlers) == 1:
            # we are operating on only one handler, so it doesn't take a list of handlers
            kwargs.pop("parameterize_handlers")

        labeler.export_ff(*args, **kwargs)
        self.labeler = labeler
        #

        super().apply(self._select, targets=targets)

        # write the main output config file
        with open(self.fbinput_fname) as fin:
            header = fin.readlines()
            write_header = True
            if os.path.exists(self.prefix + ".in"):
                write_header = False
            with open(self.prefix + ".in", "w" if write_header else "a") as fout:
                if write_header:
                    _ = [fout.write(line) for line in header]
                for tgt in self.db:
                    opts = self.db[tgt]["data"].get("global")
                    if opts is not None:
                        fout.write(opts)


class ForceBalanceObjectiveOptGeoSetup(offsb.treedi.tree.TreeOperation):

    """
    This class will iterate over the data and only
    1. generate the appropriate target options for each during apply (should be automatic)
    """

    def __init__(
        self,
        fbinput_fname,
        source_tree,
        name,
        ff_fname,
        prefix="optimize",
        verbose=True,
    ):
        super().__init__(source_tree, name)
        import logging

        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.ERROR)

        self._select = "Entry"
        self.prefix = prefix

        self.global_opts, _ = parse_inputs(fbinput_fname)

        self.fbinput_fname = fbinput_fname
        self.ff_fname = ff_fname

        self.parameterize_handlers = None

        self.processes = 1

        self._abinitio = False
        self._optgeo = False
        self._vibration = False
        self._torsiondrive = False

        self.td_denom = 1.0
        self.td_upper = 5.0
        self.restrain_k = 0.0

        self._bond_denom = 0.05
        self._angle_denom = 8
        self._dihedral_denom = 0
        self._improper_denom = 20

        self.source = DummyTree
        DummyTree.source = source_tree
        self.fb_main_opts = (
            "\n"
            + "\n$target"
            + "\nname {:s}"
            + "\ntype OptGeoTarget_SMIRNOFF"
            + "\nweight {:16.13f}"
            + "\nwritelevel 2"
            + "\nremote 1"
            + "\n$end"
        )

        self.fb_tgt_opts = (
            "\n"
            + "\n$global"
            + "\nbond_denom {0:8.4f}"
            + "\nangle_denom {1:8.4f}"
            + "\ndihedral_denom {2:8.4f}"
            + "\nimproper_denom {3:8.4f}"
            + "\n$end"
            + "\n\n$system"
            + "\nname {4:s}"
            + "\ngeometry mol.xyz"
            + "\ntopology mol.pdb"
            + "\nmol2 mol.sdf"
            + "\n$end"
        )
        self.fb_main_opts_ai = (
            "\n"
            + "\n$target"
            + "\nname {:s}"
            + "\ntype AbInitio_SMIRNOFF"
            + "\nweight {:16.13f}"
            + "\nwritelevel 2"
            + "\nremote 1"
            + "\nforce true"
            + "\nenergy true"
            + "\nenergy_rms_override 1.0"
            + "\nenergy_denom 1.0"
            + "\ncoords mol.xyz"
            + "\npdb mol.pdb"
            + "\nmol2 mol.sdf"
            + "\n$end"
        )
        self.fb_main_opts_vf = (
            "\n"
            + "\n$target"
            + "\nname {:s}"
            + "\ntype VIBRATION_SMIRNOFF"
            + "\nwritelevel 2"
            + "\ncoords mol.pdb"
            + "\npdb mol.pdb"
            + "\nmol2 mol.sdf"
            + "\nweight 0.1"
            + "\nwavenumber_tol 200.0"
            + "\nremote 1"
            + "\n$end"
        )
        self.fb_main_opts_td = (
            "\n"
            + "\n$target"
            + "\nname {dir:s}"
            + "\ntype TorsionProfile_SMIRNOFF"
            + "\npdb mol.pdb"
            + "\nmol2 mol.sdf"
            + "\ncoords scan.xyz"
            + "\nwritelevel 2"
            + "\nattenuate"
            + "\nenergy_denom {denom:f}"
            + "\nenergy_upper {upper:f}"
            + "\nrestrain_k {restrain_k:f}"
            + "\nremote 1"
            + "\n$end"
        )

    def _apply_initialize(self, targets):
        pass

    def _apply_finalize(self, targets):
        pass

    def op(self, node, partition):
        pass

    def _unpack_result(self, ret):

        for k, tgt in ret.items():
            for target in [
                x
                for x, switch in zip(
                    ["geometry", "energy", "vibration", "torsiondrive"],
                    [self._optgeo, self._abinitio, self._vibration, self._torsiondrive],
                )
                if switch
            ]:
                v = tgt["data"].get(target, None)
                if v is None:
                    continue
                dir = os.path.join("targets", v["dir"])
                try:
                    os.mkdir(dir)
                except FileExistsError:
                    pass
                path = os.path.join(dir, "mol")
                for ext in ["pdb", "mol2", "xyz", "sdf"]:
                    if ext not in v:
                        continue
                    out_str = v[ext]

                    # Workaround so that xyz is considered a trajectory,
                    # but others like pdb is a topology with only one snap
                    if ext == "xyz":
                        mode = "a"
                    else:
                        mode = "w"
                    with open(path + "." + ext, mode) as f:
                        f.write(out_str)
                if v.get("local", None) is not None:
                    path = os.path.join(dir, v["local_fnm"])

                    # This writes the optgeo options in the individual target folders
                    with open(path, "w") as f:
                        for opts in ["local"]:
                            f.write(v[opts])

        self.db.update(ret)

    def generate_targets(self):
        """
        for each of the types, generate a target
        """

        pass

    def parse_output(self):
        """
        take the results from a FB optimization, and gather the data for each
        """
        pass

    def _generate_apply_kwargs(self, i, target, kwargs=None):

        """
        get the entry
        really really want to try doing optgeo for everything
        Lets go for it :) this means it will generate a target for every molecule in the opt
        So we need make a target with a number id for each target xyz
        but maybe we make a target folder for each so we can apply a specific weight (likely a energy weighted)

        """

        # # labels = self.source.db[target.payload]["data"]
        entry = self.source.source.db[target.payload]["data"]

        # out_str = ""

        if kwargs is None:
            kwargs = {}

        """
        need to go into an entry, grab all molecules
        save coordinates and energy
        """

        # mol = kwargs.get("mol")
        QCA = self.source.source
        cwd = []

        smi = entry.attributes["canonical_isomeric_explicit_hydrogen_mapped_smiles"]

        mols = []
        mol_ids = []
        enes = []
        has_hess = []
        cwd = []
        i = 0

        mass_table = offsb.dev.hessian.mass_table

        map_inv = None

        try:
            td = next(QCA.node_iter_depth_first(target, select="TorsionDrive"))
            is_td = True
        except StopIteration:
            is_td = False

        # # keeps track of energy of multiple opts
        # ensemble_energy = 0.0

        for opt_node in QCA.node_iter_depth_first(target, select="Optimization"):
            job = 0
            for mol_node in QCA.node_iter_depth_first(opt_node, select="Molecule"):
                # grad_node = QCA[mol_node.parent]

                opt = QCA.db[opt_node.payload]["data"]
                traj = opt.trajectory
                qcid = mol_node.payload.split("-")[1]
                grad_id = QCA[mol_node.parent].payload.split("-")[1]

                if map_inv is None:
                    ref_mol = QCA.db[mol_node.payload]["data"]
                    mol = offsb.rdutil.mol.rdmol_from_smiles_and_qcmol(smi, ref_mol)

                    map_idx = offsb.rdutil.mol.atom_map(mol)
                    map_inv = offsb.rdutil.mol.atom_map_invert(map_idx)

                if qcid == opt.initial_molecule:
                    continue
                i = traj.index(grad_id)
                energies = opt.energies
                energy = energies[i]
                qcmol = self.source.source.db.get(mol_node.payload).get("data")
                mols.append(qcmol)
                enes.append(energy)
                has_hess.append(False)
                mol_ids.append(mol_node.payload)
                cwd.append(opt_node.payload + "." + mol_node.payload)
                grad_node = QCA[mol_node.parent]
                if self._abinitio:
                    # combine all snaps into a single target.. needed for energy matching
                    dir = "targets/AI." + opt_node.payload

                    try:
                        os.mkdir(dir)
                    except FileExistsError:
                        pass
                    fid = open(os.path.join(dir, "qdata.txt"), "a")
                    fid.write("JOB {:d}\n".format(job))
                    job += 1
                    xyz = [qcmol.geometry[i] for i in map_inv]
                    xyz = offsb.tools.util.flatten_list(xyz)
                    fid.write("COORDS")
                    for x in xyz:
                        fid.write(
                            " {:16.12f}".format(x * offsb.tools.const.bohr2angstrom)
                        )
                    fid.write("\n")
                    fid.write("ENERGY {:16.12f}\n".format(energy))
                    if not "Stub" in grad_node.name:
                        gradient = QCA.db[grad_node.payload]["data"].return_result
                        fid.write("GRADIENT")
                        gradient = [gradient[i] for i in map_inv]
                        gradient = offsb.tools.util.flatten_list(gradient)
                        for g in gradient:

                            # the documentation says it should be in au, but these differ by a factor of 1000 compared to MM? So I assume they are in kcal or kj.. kcal matches closer but fb says everything is kJ.. both work out
                            gval = g * offsb.tools.const.hartree2kjmol

                            # gval = g
                            fid.write(" {:16.12f}".format(gval))
                        fid.write("\n")
                    fid.close()
                if self._vibration:
                    for hess_node in QCA.node_iter_depth_first(
                        mol_node, select="Hessian"
                    ):
                        has_hess[-1] = True
                        dir = "targets/VF." + cwd[-1]

                        try:
                            os.mkdir(dir)
                        except FileExistsError:
                            pass

                        fid = open(os.path.join(dir, "vdata.txt"), "w")

                        # need to write the xyz and the hessian
                        offsb.qcarchive.qcmol_to_xyz(qcmol, fd=fid)

                        hess = QCA.db[hess_node.payload]["data"].return_result
                        # hess /= offsb.tools.const.bohr2angstrom**2 * offsb.tools.const.kcalmol2hartree

                        mass = np.array(
                            [[mass_table[atom]] * 3 for atom in qcmol.symbols]
                        )

                        # assumes au with mass in dalton
                        modes, freq = offsb.dev.hessian.hessian_modes(
                            hess, qcmol.geometry, mass, 0, remove=6
                        )

                        for w, q in zip(freq[6:], modes[6:]):
                            fid.write("\n")
                            fid.write("{:18.14f}\n".format(w))
                            for row in q.reshape(-1, 3):
                                fid.write(
                                    " ".join([" {:20.6f}"] * 3).format(*row) + "\n"
                                )
                        fid.close()

        td_dir = None
        if is_td:

            # This is a weird one: write the scan and the energy now, to avoid
            # the unnecessary send to multiprocess
            # This means half of the target is written now, half later (in unpack)
            td_dir = "TD." + td.payload
            td_dir_path = os.path.join("targets", td_dir)
            if not os.path.exists(td_dir_path):
                os.mkdir(td_dir_path)
            scan_fid = open(os.path.join(td_dir_path, "scan.xyz"), "w")
            ene_fid = open(os.path.join(td_dir_path, "qdata.txt"), "w")

            grid = []

            for i, opt_node in enumerate(
                QCA.node_iter_torsiondriverecord_minimum(target, select="Optimization")
            ):

                opt = QCA.db[opt_node.payload]["data"]
                mol_node = list(QCA.node_iter_depth_first(opt_node, select="Molecule"))[
                    0
                ]

                constraints = list(
                    QCA.node_iter_to_root(opt_node, select="Constraint")
                )[::-1]
                constraints = [x.payload[2] for x in constraints]
                grid.append(constraints)

                energies = opt.energies
                energy = energies[-1]

                qcmol = self.source.source.db.get(mol_node.payload).get("data")
                offsb.qcarchive.qcmol_to_xyz(
                    qcmol, fd=scan_fid, atom_map={k: v - 1 for k, v in map_idx.items()}
                )
                ene_fid.write("ENERGY {:16.12f}\n".format(energy))

            # Since the mol2 and pdb writers use canonical ordering, we need to unmap
            # the QCA ordering

            dihedrals = entry.td_keywords.dihedrals
            for i, _ in enumerate(dihedrals):
                dihedrals[i] = [map_inv[j] for j in dihedrals[i]]

            metadata = {"dihedrals": dihedrals, "torsion_grid_ids": grid}

            with open(os.path.join(td_dir_path, "metadata.json"), "w") as metadata_fid:
                json.dump(metadata, metadata_fid)

            scan_fid.close()
            ene_fid.close()

            if len(grid) == 0:
                kwargs["error"] = "This torsiondrive empty! ID {:s}".format(
                    target.payload
                )
                shutil.rmtree(td_dir_path)

        # kwargs["td_metadata"] = metadata
        kwargs["map_inv"] = map_inv
        kwargs["geometry"] = self._optgeo
        kwargs["energy"] = self._abinitio
        kwargs["vibration"] = self._vibration
        kwargs["torsiondrive"] = self._torsiondrive
        kwargs["has_hess"] = has_hess
        kwargs["smi"] = smi
        kwargs["mol"] = mols
        kwargs["mol_ids"] = mol_ids
        kwargs["ene"] = enes
        kwargs["grads"] = enes
        kwargs["cwd"] = cwd
        kwargs["td_dir"] = td_dir
        kwargs["global"] = {}
        kwargs["local"] = {}
        kwargs["is_td"] = is_td
        if self._optgeo:
            kwargs["global"]["geometry"] = self.fb_main_opts
            kwargs["local"]["geometry"] = self.fb_tgt_opts
        if self._abinitio:
            kwargs["global"]["energy"] = self.fb_main_opts_ai
            kwargs["local"]["energy"] = None
        if self._vibration:
            kwargs["global"]["vibration"] = self.fb_main_opts_vf
            kwargs["local"]["vibration"] = None
        if self._torsiondrive:
            kwargs["global"]["torsiondrive"] = self.fb_main_opts_td
            kwargs["local"]["torsiondrive"] = None
        return kwargs

    def apply_single_target_objective():
        pass

    def apply_single(self, i, target, **kwargs):

        smi = kwargs["smi"]
        enes = np.array(kwargs["ene"])
        has_hess = kwargs["has_hess"]
        mols = kwargs["mol"]
        mol_ids = kwargs["mol_ids"]
        cwd = kwargs["cwd"]
        fb_main_opts = kwargs["global"]
        fb_tgt_opts = kwargs["local"]
        is_td = kwargs.get("is_td", False)

        # set T= 298 K
        kT = offsb.tools.const.kT2kjmol / 298.0

        if is_td:

            # TODO: make this configurable
            denom = 1.0
            upper = 5.0

            # input assumes kJ/mol, then converts to kcal/mol in FB
            enes = offsb.tools.const.hartree2kjmol * (enes - enes.min())
            for i, e in enumerate(enes):
                if e > upper:
                    enes[i] = 0.0
                elif e < denom:
                    enes[i] = 1.0 / denom
                else:
                    enes[i] = 1.0 / np.sqrt(denom ** 2 + (e - denom) ** 2)
            # enes = (
            #     np.exp(-(offsb.tools.const.hartree2kjmol * (enes - enes.min())) / kT)
            #     / total_ene
            # )
            enes /= sum(enes)
        else:
            total_ene = np.exp(
                -(offsb.tools.const.hartree2kjmol * (enes - enes.min())) / kT
            ).sum()
            enes = (
                np.exp(-(offsb.tools.const.hartree2kjmol * (enes - enes.min())) / kT)
                / total_ene
            )
        # print((np.exp(-(offsb.tools.const.hartree2kjmol*(enes - enes.min()))/kT)/total_ene).sum())

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "sdf")
        obmol = openbabel.OBMol()

        ret_obj = {}

        for i, (qcmol, ene, has_hess_i, dir) in enumerate(
            zip(mols, enes, has_hess, cwd)
        ):

            mol_id = mol_ids[i]
            mol = offsb.rdutil.mol.rdmol_from_smiles_and_qcmol(smi, qcmol)

            og_weight = ene

            xyz_str = Chem.MolToXYZBlock(mol)
            pdb_str = Chem.MolToPDBBlock(mol)
            obConversion.ReadString(obmol, pdb_str)
            mol2_str = obConversion.WriteString(obmol)

            ret_obj[mol_id] = {"data": {}}
            if kwargs["geometry"]:
                ret_obj[mol_id]["data"]["geometry"] = {
                    "dir": "OG." + dir,
                    "pdb": pdb_str,
                    "sdf": mol2_str,
                    "xyz": xyz_str,
                    "global": fb_main_opts["geometry"].format("OG." + dir, og_weight),
                    "local": fb_tgt_opts["geometry"].format(
                        self._bond_denom,
                        self._angle_denom,
                        self._dihedral_denom,
                        self._improper_denom,
                        mol_id,
                    ),
                    "local_fnm": "optgeo_options.txt",
                }
            if kwargs["energy"]:
                ret_obj[mol_id]["data"]["energy"] = {
                    "dir": "AI." + dir.split(".")[0],
                    "pdb": pdb_str,
                    "sdf": mol2_str,
                    "xyz": xyz_str,
                    "weight": 1.0,
                    "global": fb_main_opts["energy"].format(
                        "AI." + dir.split(".")[0], 0.1
                    ),
                }
            if kwargs["vibration"] and has_hess_i:
                ret_obj[mol_id]["data"]["vibration"] = {
                    "dir": "VF." + dir,
                    "pdb": pdb_str,
                    "sdf": mol2_str,
                    "weight": 1.0,
                    "global": fb_main_opts["vibration"].format("VF." + dir),
                }
            if kwargs["torsiondrive"]:
                td_dir = kwargs["td_dir"]
                ret_obj[mol_id]["data"]["torsiondrive"] = {
                    "dir": td_dir,
                    "pdb": pdb_str,
                    "sdf": mol2_str,
                    "weight": 1.0,
                    "global": fb_main_opts["torsiondrive"].format(
                        dir=td_dir,
                        denom=self.td_denom,
                        upper=self.td_upper,
                        restrain_k=self.restrain_k,
                    ),
                }

        return {target.payload: "", "return": ret_obj}

    def apply(
        self,
        targets=None,
        parameterize_handlers=None,
        fitting_targets=["geometry"],
        parameterize_terms=None,
        parameterize_spatial=True,
        parameterize_force=True,
    ):
        if parameterize_handlers is None:
            parameterize_handlers = self.parameterize_handlers

        lvl = self.logger.getEffectiveLevel()
        fb_logger = logging.getLogger("forcebalance")
        fb_logger.setLevel(lvl)
        off_ph_logger = logging.getLogger("openforcefield")
        off_ph_logger.setLevel(lvl)
        rdkit_logger = logging.getLogger("rdkit")
        rdkit_logger.setLevel(lvl)

        for folder in ["optimize.tmp", "optimize.bak", "result"]:
            try:
                shutil.rmtree(folder)
            except FileNotFoundError:
                pass

        if "geometry" in fitting_targets:
            self._optgeo = True
        if "energy" in fitting_targets:
            self._abinitio = True
        if "vibration" in fitting_targets:
            self._vibration = True
        if "torsiondrive" in fitting_targets:
            self._torsiondrive = True
        try:
            os.mkdir("targets")
        except Exception:
            pass

        labeler = None

        if parameterize_handlers is None:
            parameterize_handlers = [
                "vdW",
                "Bonds",
                "Angles",
                "ProperTorsions",
                "ImproperTorsions",
            ]

        print("Loading FF for labeling", self.ff_fname)
        ff_kwargs = dict(allow_cosmetic_attributes=True)
        if len(parameterize_handlers) > 1:
            labeler = offsb.op.openforcefield.OpenForceFieldTree(
                self.source.source, "ff", self.ff_fname, ff_kwargs=ff_kwargs
            )
        elif len(parameterize_handlers) == 1:
            if "vdW" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldvdWTree(
                    self.source.source, "ff", self.ff_fname, ff_kwargs=ff_kwargs
                )
            elif "Bonds" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldBondTree(
                    self.source.source, "ff", self.ff_fname, ff_kwargs=ff_kwargs
                )
            elif "Angles" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldAngleTree(
                    self.source.source, "ff", self.ff_fname, ff_kwargs=ff_kwargs
                )
            elif "ProperTorsions" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldTorsionTree(
                    self.source.source, "ff", self.ff_fname, ff_kwargs=ff_kwargs
                )
            elif "ImproperTorsions" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldImproperTorsionTree(
                    self.source.source, "ff", self.ff_fname, ff_kwargs=ff_kwargs
                )
            else:
                raise Exception(
                    "Parameter handler" + parameterize_handlers[0] + "not supported"
                )

        labeler.processes = 1
        labeler.apply()

        # export the FF
        args = (self.prefix + ".offxml",)

        # default case if we don't want any parameters fit, e.g. no "cosmetic attributes"
        kwargs = dict(
            parameterize_handlers=parameterize_handlers,
            parameterize_spatial=parameterize_spatial,
            parameterize_force=parameterize_force,
        )

        # generate parameter fits to multiple handlers
        if len(parameterize_handlers) >= 1:
            kwargs.update(
                dict(
                    parameterize_spatial=parameterize_spatial,
                    parameterize_force=parameterize_force,
                )
            )
        if len(parameterize_handlers) == 1:
            # we are operating on only one handler, so it doesn't take a list of handlers
            kwargs.pop("parameterize_handlers")

        # print("Exporting FF to", args)
        labeler.export_ff(*args, **kwargs)
        self.labeler = labeler
        #

        super().apply(self._select, targets=targets)

        targets = []
        # write the main output config file
        with open(self.fbinput_fname) as fin:
            header = fin.readlines()
            with open(self.prefix + ".in", "w") as fout:
                _ = [fout.write(line) for line in header]
                for tgt in self.db:
                    for config in self.db[tgt]["data"].values():
                        opts = config.get("global")
                        if opts is not None:
                            if opts not in targets:
                                fout.write(opts)
                                targets.append(opts)


class ForceBalanceObjectiveEvaluatorSetup(offsb.treedi.tree.TreeOperation):
    def __init__(
        self,
        fbinput_fname,
        source_tree,
        name,
        ff_fname,
        prefix="optimize",
        verbose=True,
    ):
        super().__init__(source_tree, name)
        import logging

        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.ERROR)

        self._select = "Entry"
        self.prefix = prefix

        self.global_opts, _ = parse_inputs(fbinput_fname)

        self.fbinput_fname = fbinput_fname
        self.ff_fname = ff_fname

        self.parameterize_handlers = None

        self.processes = 1

        self._abinitio = False
        self._optgeo = False
        self._vibration = False

        self.td_denom = 1.0
        self.td_upper = 5.0

        self._bond_denom = 0.05
        self._angle_denom = 8
        self._dihedral_denom = 0
        self._improper_denom = 20

        self._fb_tgt_opts = {
            "connection_options": {"server_address": "localhost", "server_port": 8000},
            "data_set_path": "training-set.json",
            "denominators": {
                "Density": {
                    "@type": "openff.evaluator.unit.Quantity",
                    "unit": "g / ml",
                    "value": 0.05,
                },
                "EnthalpyOfMixing": {
                    "@type": "openff.evaluator.unit.Quantity",
                    "unit": "kJ / mol",
                    "value": 1.6,
                },
            },
            "estimation_options": {
                "batch_mode": {
                    "@type": "openff.evaluator.client.client.BatchMode",
                    "value": "SharedComponents",
                },
                "calculation_layers": ["SimulationLayer"],
            },
            "polling_interval": 600,
            "weights": {"Density": 1.0, "EnthalpyOfMixing": 1.0},
        }

        self.source = DummyTree
        DummyTree.source = source_tree
        self.fb_main_opts = (
            "\n"
            + "\n$target"
            + "\nname {:s}"
            + "\ntype Evaluator_SMIRNOFF"
            + "\nweight {:16.13f}"
            + "\nopenff.evaluator_input options.json"
            + "\n$end"
        )


# Secret notes

global_opts = {
    "adaptive_damping": 0.5,
    "adaptive_factor": 0.25,
    "amberhome": None,
    "amoeba_eps": None,
    "amoeba_pol": None,
    "asynchronous": 0,
    "backup": 1,
    "constrain_charge": False,
    "constrain_h": 0,
    "continue": 0,
    "converge_lowq": 0,
    "convergence_gradient": 0.1,
    "convergence_objective": 0.1,
    "convergence_step": 0.01,
    "criteria": 2,
    "duplicate_pnames": 0,
    "eig_lowerbound": 0.01,
    "error_tolerance": 0.0,
    "ffdir": "forcefield",
    "finite_difference_factor": 0.1,
    "finite_difference_h": 0.01,
    "forcefield": ["param_valence.offxml"],
    "gmxpath": "",
    "gmxsuffix": "",
    "have_vsite": 0,
    "input_file": "optimize.in",
    "jobtype": "SINGLE",
    "lm_guess": 1.0,
    "logarithmic_map": 0,
    "maxstep": 100,
    "mintrust": 0.0,
    "normalize_weights": False,
    "objective_history": 2,
    "penalty_additive": 1.0,
    "penalty_alpha": 0.001,
    "penalty_hyperbolic_b": 1e-06,
    "penalty_multiplicative": 0.0,
    "penalty_power": 2.0,
    "penalty_type": "L2",
    "print_gradient": 1,
    "print_hessian": 0,
    "print_parameters": 1,
    "priors": OrderedDict(
        [
            ("Bonds/Bond/k", 100.0),
            ("Bonds/Bond/length", 0.1),
            ("Angles/Angle/k", 50.0),
            ("Angles/Angle/angle", 10.0),
            ("ProperTorsions/Proper/k", 0.2),
        ]
    ),
    "read_mvals": None,
    "read_pvals": None,
    "readchk": None,
    "reevaluate": None,
    "rigid_water": 0,
    "root": "redacted",
    "rpmd_beads": 0,
    "scan_vals": None,
    "scanindex_name": [],
    "scanindex_num": [],
    "search_tolerance": 0.0001,
    "step_lowerbound": 1e-06,
    "tinkerpath": "",
    "trust0": -1.0,
    "use_pvals": 0,
    "verbose_options": 0,
    "vsite_bonds": 0,
    "wq_port": 0,
    "writechk": None,
    "writechk_step": 1,
    "zerograd": -1,
}

tgt_opts = [
    {
        "adapt_errors": 0,
        "all_at_once": 1,
        "amber_leapcmd": None,
        "anisotropic_box": 0,
        "attenuate": 0,
        "coords": None,
        "dipole_denom": 1.0,
        "do_cosmo": 0,
        "energy": 1,
        "energy_asymmetry": 1.0,
        "energy_denom": 1.0,
        "energy_mode": "average",
        "energy_rms_override": 0.0,
        "energy_upper": 30.0,
        "engine": None,
        "epsgrad": 0.0,
        "eq_steps": 20000,
        "evaluator_input": "evaluator_input.json",
        "expdata_txt": "expset.txt",
        "fd_ptypes": [],
        "fdgrad": 0,
        "fdhess": 0,
        "fdhessdiag": 0,
        "fitatoms": "0",
        "force": 1,
        "force_average": 0,
        "force_cuda": 0,
        "force_map": "residue",
        "force_rms_override": 0.0,
        "fragment1": "",
        "fragment2": "",
        "gas_coords": None,
        "gas_eq_steps": 10000,
        "gas_interval": 0.1,
        "gas_md_steps": 100000,
        "gas_timestep": 1.0,
        "gmx_eq_barostat": "berendsen",
        "gmx_mdp": None,
        "gmx_ndx": None,
        "gmx_top": None,
        "hfe_pressure": 1.0,
        "hfe_temperature": 298.15,
        "hfedata_txt": "hfedata.txt",
        "hfemode": "single",
        "hvap_subaverage": 0,
        "inter_txt": "interactions.txt",
        "lipid_coords": None,
        "lipid_eq_steps": 1000,
        "lipid_interval": 0.1,
        "lipid_md_steps": 10000,
        "lipid_timestep": 1.0,
        "liquid_coords": None,
        "liquid_eq_steps": 1000,
        "liquid_fdiff_h": 0.01,
        "liquid_interval": 0.1,
        "liquid_md_steps": 10000,
        "liquid_timestep": 1.0,
        "manual": 0,
        "md_steps": 50000,
        "md_threads": 1,
        "minimize_energy": 1,
        "mol2": [],
        "mts_integrator": 0,
        "n_mcbarostat": 25,
        "n_molecules": -1,
        "n_sim_chain": 1,
        "name": "optgeo_OpenFF_Gen_2_Opt_Set_1_Roche-0",
        "nonbonded_cutoff": None,
        "normalize": 0,
        "nvt_coords": None,
        "nvt_eq_steps": 10000,
        "nvt_interval": 0.1,
        "nvt_md_steps": 100000,
        "nvt_timestep": 1.0,
        "openmm_platform": None,
        "openmm_precision": None,
        "optgeo_options_txt": "optgeo_options.txt",
        "optimize_geometry": 1,
        "pdb": None,
        "polarizability_denom": 1.0,
        "pure_num_grad": 0,
        "qdata_txt": None,
        "quadrupole_denom": 1.0,
        "quantities": [],
        "read": None,
        "reassign_modes": None,
        "recharge_esp_store": "esp-store.sqlite",
        "recharge_property": "esp",
        "remote": True,
        "remote_backup": 0,
        "remote_prefix": "",
        "resp": 0,
        "resp_a": 0.001,
        "resp_b": 0.1,
        "restrain_k": 1.0,
        "rmsd_denom": 0.1,
        "run_internal": 1,
        "save_traj": 0,
        "self_pol_alpha": 0.0,
        "self_pol_mu0": 0.0,
        "shots": -1,
        "sleepy": 0,
        "subset": None,
        "tinker_key": None,
        "type": "OPTGEOTARGET_SMIRNOFF",
        "vdw_cutoff": None,
        "w_al": 1.0,
        "w_alpha": 1.0,
        "w_cp": 1.0,
        "w_energy": 1.0,
        "w_eps0": 1.0,
        "w_force": 1.0,
        "w_hvap": 1.0,
        "w_kappa": 1.0,
        "w_netforce": 0.0,
        "w_normalize": 0,
        "w_resp": 0.0,
        "w_rho": 1.0,
        "w_scd": 1.0,
        "w_surf_ten": 0.0,
        "w_torque": 0.0,
        "wavenumber_tol": 10.0,
        "weight": 0.1,
        "writelevel": 1,
    }
]

# Notes
"""
specify a file with the global opts as FB input

then, generate the target folders and append targets to global
then run fb as normal

this means the target generator should know exactly the node -> dir -> output ref

name each target folder as the targets/(DB-ID)/QCP-ID
    optgeo_options.txt
    mol.xyz
    mol.pdb
    mol.mol2

Now, modify the FF such that we parameterize all covered params, then save as ff

at this point, the main input, the targets, and the ff is made

run FB

TD has these opts
$target
name td_OpenFF_Group1_Torsions_715_C6H7N3O
type TorsionProfile_SMIRNOFF
mol2 input.mol2
pdb conf.pdb
coords scan.xyz
writelevel 2
attenuate
energy_denom 1.0
energy_upper 5.0
remote 1
$end

everything seems to be in angstrom; qdata.txt energy likely a.u.

"""
