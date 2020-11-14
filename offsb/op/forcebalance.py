#!/usr/bin/env python3

import logging
import os
import shutil
from collections import OrderedDict

import numpy as np
import tqdm

import forcebalance.target
import offsb.op.chemper
import offsb.op.openforcefield
import offsb.rdutil
import offsb.search.smiles
import offsb.treedi.tree
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer
from forcebalance.parser import parse_inputs
from openbabel import openbabel
from openforcefield.typing.engines.smirnoff.forcefield import ForceField
from openforcefield.typing.engines.smirnoff.parameters import (ImproperDict,
                                                               ValenceDict)
from rdkit import Chem

np.set_printoptions(linewidth=9999, formatter={"float_kind": "{:8.1e}".format})


class DummyTree:
    source = None
    ID = None


class ForceBalanceObjectiveOptGeo(offsb.treedi.tree.TreeOperation):
    def __init__(self, fbinput_fname, source_tree, name, ff_fname, init=None):
        super().__init__(source_tree, name)
        import logging

        self._select = "Molecule"

        if init is None:
            self._setup = ForceBalanceObjectiveOptGeoSetup(
                fbinput_fname, source_tree, "fb_setup." + name, ff_fname, "optimize"
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

        DummyTree.source = source_tree
        print("My db is", self.db)

    def _unpack_result(self, ret):
        self.db.update(ret)

    def _generate_apply_kwargs(self, i, target, kwargs=None):

        if kwargs is None:
            kwargs = {}
        arg = np.zeros(self._forcefield.np)
        QCA = self.source.source
        node = next(QCA.node_iter_depth_first(QCA.root(), select=self._select))
        opt = next(QCA.node_iter_to_root(node, select="Optimization"))
        kwargs["arg"] = arg
        found=False
        for tgt in self._objective.Targets:
            for key in tgt.internal_coordinates:
                if target.payload in key:
                    kwargs["tgt"] = tgt
                    found=True
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

    def optimize(self, targets=None, parameterize_handlers=None, jobtype="OPTIMIZE"):

        if self._init == False:
            self._setup.apply(
                targets=targets,
                parameterize_handlers=parameterize_handlers,
                fitting_targets=["geometry"],
            )
            self._init = True
        else:
            self.remove_tmp(clean_input=False)

        if self._options is None or self._tgt_opts is None:
            self._options, self._tgt_opts = parse_inputs("optimize.in")

        self._options["jobtype"] = "OPTIMIZE"

        self._forcefield = FF(self._options)
        self.db.clear()
        self.db["ROOT"] = {"data": self._forcefield.plist}

        objective = Objective(self._options, self._tgt_opts, self._forcefield)
        optimizer = Optimizer(self._options, objective, self._forcefield)

        fb_logger = logging.getLogger("forcebalance")
        fb_logger.setLevel(logging.INFO)
        ans = optimizer.Run()

        new_ff = self._setup.prefix + ".offxml"
        self.new_ff = None
        if os.path.exists("results/" + new_ff):
            self.new_ff = ForceField("results/" + new_ff)

    def apply(self, targets=None, parameterize_handlers=None):

        if self._init == False:
            self._setup.apply(
                targets=targets,
                parameterize_handlers=parameterize_handlers,
                fitting_targets=["geometry"],
            )
            self._init = True
        else:
            self.remove_tmp(clean_input=False)

        fb_logger = logging.getLogger("forcebalance")
        fb_logger.setLevel(logging.WARN)

        # The general options and target options that come from parsing the input file
        if self._options is None or self._tgt_opts is None:
            self._options, self._tgt_opts = parse_inputs("optimize.in")

        self._forcefield = FF(self._options)

        # Because ForceBalance Targets contain unpicklable objects, we must
        # use single process
        # TODO: use the FB work queue interface
        self.processes = 1
        opts = self._options.copy()

        self.db.clear()
        self.db["ROOT"] = {"data": self._forcefield.plist}

        remote = opts.get("asynchronous", False)
        self._objective = Objective(opts, self._tgt_opts, self._forcefield)

        if not remote:
            super().apply(self._select, targets=targets)
        else:
            logging.getLogger("forcebalance").setLevel(logging.INFO)

            # QCA = self.source.source
            port = opts["wq_port"]
            print("Starting WorkQueue... please start a worker and set to port", port)
            arg = np.zeros(len(self._forcefield.plist))
            self._objective.Full(arg, Order=1, verbose=1)
            for tgt, dat in self._objective.ObjDict.items():
                rec = tgt.split(".")[-1]

                # skip known keys that we must skip
                if tgt in ['Total', 'Regularization']:
                    continue

                IC = dat['IC'][tgt]
                # node = [ x for x in QCA.node_iter_depth_first(QCA.root()) if x.payload == rec ][0]
                
                ret = self._generate_ic_objective_pairs(IC, dat)
                dat = dat.copy()
                dat.update(ret)

                if rec not in self.db:
                    self.db[rec] = {"data": dat}
                else:
                    self.db[rec]["data"].update(dat)
            logging.getLogger("forcebalance").setLevel(logging.WARN)

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

    def _generate_ic_objective_pairs(self, IC_dict, ans, i=0):

        ret = {}

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

            vals = v * ans["dV"][:, i]
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


class ForceBalanceObjectiveOptGeoSetup(offsb.treedi.tree.TreeOperation):

    """
    This class will iterate over the data and only
    1. generate the appropriate target options for each during apply (should be automatic)
    """

    def __init__(self, fbinput_fname, source_tree, name, ff_fname, prefix="optimize"):
        super().__init__(source_tree, name)
        import logging

        self._select = "Entry"
        self.prefix = prefix

        self.global_opts, _ = parse_inputs(fbinput_fname)

        self.fbinput_fname = fbinput_fname
        self.ff_fname = ff_fname

        self.processes = 1

        self.source = DummyTree
        DummyTree.source = source_tree
        print("My db is", self.db)
        self.fb_main_opts = (
            "\n"
            + "\n$target"
            + "\nname {:s}"
            + "\ntype OptGeoTarget_SMIRNOFF"
            + "\nweight {:8.6f}"
            + "\nwritelevel 0"
            + "\nremote 1"
            + "\n$end"
        )

        self.fb_tgt_opts = (
            "\n"
            + "\n$global"
            + "\nbond_denom 0.05"
            + "\nangle_denom 8"
            + "\ndihedral_denom 20"
            + "\nimproper_denom 20"
            + "\n$end"
            + "\n\n$system"
            + "\nname {:s}"
            + "\ngeometry mol.xyz"
            + "\ntopology mol.pdb"
            + "\nmol2 mol.mol2"
            + "\n$end"
        )

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
            for ext in ["pdb", "mol2", "xyz"]:
                out_str = v[ext]
                with open(path + "." + ext, "w") as f:
                    f.write(out_str)
            path = os.path.join(dir, "optgeo_options.txt")

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
        cwd = []
        i = 0
        for opt_node in QCA.node_iter_depth_first(target, select="Optimization"):
            for mol_node in QCA.node_iter_depth_first(opt_node, select="Molecule"):
                # grad_node = QCA[mol_node.parent]
                opt = QCA.db[opt_node.payload]["data"]
                traj = opt.trajectory
                qcid = mol_node.payload.split("-")[1]
                grad_id = QCA[mol_node.parent].payload.split("-")[1]
                if qcid == opt.initial_molecule:
                    continue
                i = traj.index(grad_id)
                energies = opt.energies
                energy = energies[i]
                qcmol = self.source.source.db.get(mol_node.payload).get("data")
                mols.append(qcmol)
                enes.append(energy)
                mol_ids.append(mol_node.payload)
                cwd.append(opt_node.payload + "." + mol_node.payload)

        kwargs["smi"] = smi
        kwargs["mol"] = mols
        kwargs["mol_ids"] = mol_ids
        kwargs["ene"] = enes
        kwargs["cwd"] = cwd
        kwargs["global"] = self.fb_main_opts
        kwargs["local"] = self.fb_tgt_opts
        return kwargs

    def apply_single_target_objective():
        pass

    def apply_single(self, i, target, **kwargs):

        smi = kwargs["smi"]
        enes = kwargs["ene"]
        mols = kwargs["mol"]
        mol_ids = kwargs["mol_ids"]
        cwd = kwargs["cwd"]
        fb_main_opts = kwargs["global"]
        fb_tgt_opts = kwargs["local"]

        total_ene = sum(enes)

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "mol2")
        obmol = openbabel.OBMol()

        ret_obj = {}

        for i, (qcmol, ene, dir) in enumerate(zip(mols, enes, cwd)):

            mol_id = mol_ids[i]
            mol = offsb.rdutil.mol.rdmol_from_smiles_and_qcmol(smi, qcmol)

            weight = ene / total_ene

            xyz_str = Chem.MolToXYZBlock(mol)
            pdb_str = Chem.MolToPDBBlock(mol)
            obConversion.ReadString(obmol, pdb_str)
            mol2_str = obConversion.WriteString(obmol)

            ret_obj[mol_id] = {
                "data": {
                    "dir": dir,
                    "pdb": pdb_str,
                    "mol2": mol2_str,
                    "xyz": xyz_str,
                    "weight": weight,
                    "global": fb_main_opts.format(dir, weight),
                    "local": fb_tgt_opts.format(mol_id),
                }
            }

        return {target.payload: "", "return": ret_obj}

    def apply(
        self, targets=None, parameterize_handlers=None, fitting_targets=["geometry"]
    ):

        fb_logger = logging.getLogger("forcebalance")
        fb_logger.setLevel(logging.WARNING)
        off_ph_logger = logging.getLogger("openforcefield")
        off_ph_logger.setLevel(logging.WARNING)
        rdkit_logger = logging.getLogger("rdkit")
        rdkit_logger.setLevel(logging.ERROR)

        for folder in ["optimize.tmp", "optimize.bak", "result", "targets"]:
            try:
                shutil.rmtree(folder)
            except FileNotFoundError:
                pass

        if "geometry" in fitting_targets:
            self._optgeo = True
        if "energy" in fitting_targets:
            self._abinitio = True
        os.mkdir("targets")

        labeler = None
        if parameterize_handlers is None:
            parameterize_handlers = [
                "vdW",
                "Bonds",
                "Angles",
                "ProperTorsions",
                "ImproperTorsions",
            ]

        if len(parameterize_handlers) > 1:
            labeler = offsb.op.openforcefield.OpenForceFieldTree(
                self.source.source, "ff", self.ff_fname
            )
        elif len(parameterize_handlers) == 1:
            if "vdW" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldvdWTree(
                    self.source.source, "ff", self.ff_fname
                )
            elif "Bonds" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldBondTree(
                    self.source.source, "ff", self.ff_fname
                )
            elif "Angles" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldAngleTree(
                    self.source.source, "ff", self.ff_fname
                )
            elif "ProperTorsions" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldProperTorsionTree(
                    self.source.source, "ff", self.ff_fname
                )
            elif "ImproperTorsions" in parameterize_handlers:
                labeler = offsb.op.openforcefield.OpenForceFieldImproperTorsionTree(
                    self.source.source, "ff", self.ff_fname
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
            parameterize_spatial=False,
            parameterize_force=False,
        )

        # generate parameter fits to multiple handlers
        if len(parameterize_handlers) > 1:
            kwargs.update(
                dict(
                    parameterize_spatial=True,
                    parameterize_force=True,
                )
            )
        elif len(parameterize_handlers) == 1:
            # we are operating on only one handler, so it doesn't take a list of handlers
            kwargs.pop("parameterize_handlers")

        labeler.export_ff(*args, **kwargs)
        #

        super().apply(self._select, targets=targets)

        # write the main output config file
        with open(self.fbinput_fname) as fin:
            header = fin.readlines()
            with open(self.prefix + ".in", "w") as fout:
                _ = [fout.write(line) for line in header]
                for tgt in self.db:
                    opts = self.db[tgt]["data"].get("global")
                    if opts is not None:
                        fout.write(opts)


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
