#!/usr/bin/env python

import concurrent.futures
import copy
import functools
import itertools
import json
import logging
import multiprocessing
import os
import pickle
import re
import sys
import time

import numpy as np
import offsb.chem.graph
import offsb.chem.types
import offsb.op.chemper
import offsb.op.forcebalance
import offsb.op.internal_coordinates
import offsb.tools.const
import offsb.tools.sorted_collection
import offsb.treedi.node
import offsb.treedi.tree
import offsb.ui.qcasb
import scipy.stats
import simtk.unit
import tqdm
from offsb.api.tk import (AngleHandler, BondHandler, ForceField, ImproperDict,
                          ImproperTorsionHandler, ParameterList,
                          ProperTorsionHandler, ValenceDict, vdWHandler)
from offsb.treedi.tree import DEFAULT_DB

# sort by largest difference, then by fewest atoms, most bits, so two splits with
# the same difference will cause the most general split to win

VDW_DENOM = 10.0

# some clusters may reduce this default value; we need it since our trees
# can extend 1000+ nodes
sys.setrecursionlimit(10000)

prim_to_graph = {
    "n": offsb.chem.types.AtomType,
    "b": offsb.chem.types.BondGraph,
    "a": offsb.chem.types.AngleGraph,
    "i": offsb.chem.types.OutOfPlaneGraph,
    "t": offsb.chem.types.TorsionGraph,
}
prim_to_group = {
    "n": offsb.chem.types.AtomType,
    "b": offsb.chem.types.BondGroup,
    "a": offsb.chem.types.AngleGroup,
    "i": offsb.chem.types.OutOfPlaneGroup,
    "t": offsb.chem.types.TorsionGroup,
}

key_transformer = {
    "vdW": lambda x: x,
    "Bonds": ValenceDict.key_transform,
    "Angles": ValenceDict.key_transform,
    "ProperTorsions": ValenceDict.key_transform,
    "ImproperTorsions": ImproperDict.key_transform,
}


generic_smirnoff_params = {
    "vdW": vdWHandler.vdWType(
        smirks="[*:1]",
        epsilon="0.01 * kilocalorie/mole",
        rmin_half="1.5 * angstrom",
        id="n0",
    ),
    "Bonds": BondHandler.BondType(
        smirks="[*:1]~[*:2]",
        k="500.0 * kilocalorie/(angstrom**2*mole)",
        length="1.3 * angstrom",
        id="b0",
    ),
    "Angles": AngleHandler.AngleType(
        smirks="[*:1]~[*:2]~[*:3]",
        k="50.0 * kilocalorie/(radian**2*mole)",
        angle="109.5 * degree",
        id="a0",
    ),
    "ProperTorsions": ProperTorsionHandler.ProperTorsionType(
        smirks="[*:1]~[*:2]~[*:3]~[*:4]",
        periodicity=[1],
        k=[
            "0.0 * kilocalorie/mole",
        ],
        phase=["0.0 * degree"],
        id="t0",
        idivf=list([1.0] * 1),
    ),
    "ImproperTorsions": ImproperTorsionHandler.ImproperTorsionType(
        smirks="[*:1]~[*:2](~[*:3])~[*:4]",
        periodicity=[1],
        k=[
            "0.0 * kilocalorie/mole",
        ],
        phase=["0.0 * degree"],
        id="i0",
        idivf=list([1.0] * 1),
    ),
}


def parameter_data(
    i, entry, mols, graph, ic_db, fc_db, labeler, params_db, prim_db, ignore_params
):
    vtable = [
        "Bonds",
        "Angles",
        "ImproperTorsions",
        "ProperTorsions",
    ]

    labels = {
        key_transformer[ic_type](aidx): val
        for ic_type in vtable
        for aidx, val in labeler[ic_type].items()
        if val not in ignore_params
    }

    param_data = {}
    for ic_type, ic in ic_db.items():

        params = params_db[ic_type]

        for param_name, param in params.items():
            # if select is not None and param_name not in select:
            #     continue
            if param_name not in param_data:
                param_data[param_name] = {}
            for mol in mols:
                ic_data = ic.db[mol.payload]
                for aidx, vals in ic_data.items():  # for aidx in labels:

                    if aidx not in labels:
                        continue

                    if labels[aidx] == param_name:
                        # param_vals.extend(vals)
                        if aidx not in prim_db:
                            breakpoint()
                            print("HI!")

                        key = prim_db[aidx]
                        # if aidx not in self._fc[mol.payload]:
                        #     breakpoint()
                        #     print("HI!")

                        payload = param_data[param_name].setdefault(
                            key, dict(measure=list(), force=list())
                        )
                        payload["smarts"] = key.to_smarts()
                        # if key not in param_data[param_name]:
                        #     param_data[param_name][key] = {
                        #         "measure": [],
                        #         "force": [],
                        #     }

                        payload["measure"].extend(vals)
                        # param_data[param_name][key]["measure"].extend(vals)

                        mol_fc = fc_db.get(mol.payload)
                        if mol_fc is not None:
                            force_vals = mol_fc[aidx]

                            payload["force"].append(force_vals)
                            # param_data[param_name][key]["force"].append(
                            #     force_vals
                            # )
    return i, param_data


def ff_param_subtract(
    i,
    ff_param,
    lbl,
    bit,
    group,
    mapping,
    bit_visited,
    ignore_bits,
    operation,
    param_data_lbl,
):

    ff_subtract = ff_param.subtract(bit, map=mapping, trim=True)
    # ff_subtract = ff_subtract.intersection(group)

    early_return = False

    visited, ignore, valid, blank, occluding = [False] * 5

    visited = any(bit.equal(x) for x in bit_visited)
    early_return |= visited
    if early_return:
        return i, ff_subtract, bit, visited, ignore, valid, blank, occluding

    ignore = any(
        [
            x[0] == lbl
            and (x[1] is None or x[1].equal(ff_subtract, map=None))
            and x[2] == operation
            for x in ignore_bits
        ]
    )
    early_return |= ignore
    if early_return:
        return i, ff_subtract, bit, visited, ignore, valid, blank, occluding

    valid = ff_subtract.is_valid()
    early_return |= not valid
    if early_return:
        return i, ff_subtract, bit, visited, ignore, valid, blank, occluding

    in_new = [x in ff_subtract for x in param_data_lbl]

    blank = not any(in_new)
    early_return |= blank
    if early_return:
        return i, ff_subtract, bit, visited, ignore, valid, blank, occluding

    occluding = all(in_new)

    return i, ff_subtract, bit, visited, ignore, valid, blank, occluding


def bit_split_and_score(
    idx,
    param_data,
    group,
    key,
    lbl,
    mode,
    eps,
    denoms,
    stats_fn,
    angle_mode,
    angle_cosine_frequency_residual_max,
    quiet,
):
    idx, mask, lhs, rhs = bit_split(idx, param_data, group, key)
    idx, vals, success = bit_score(
        idx,
        lhs,
        rhs,
        lbl,
        mode,
        eps,
        key,
        denoms,
        stats_fn,
        angle_mode,
        angle_cosine_frequency_residual_max,
        quiet,
    )

    return idx, mask, vals, success


def bit_split(idx, param_data_lbl, group, key):

    mask = []
    old_data = []
    new_data = []
    for i, (prim, dat) in enumerate(param_data_lbl.items()):

        sma = dat["smarts"]
        if key is not None:
            dat = dat[key]
        # pdb.set_trace()

        if len(dat) == 0:
            continue

        if prim in group:
            mask.append(True)
            new_data.extend(dat)
        else:
            mask.append(False)
            old_data.extend(dat)

    return idx, mask, old_data, new_data


def bit_score(
    idx,
    lhs,
    rhs,
    lbl,
    mode,
    eps,
    key,
    denoms,
    stats_fn,
    angle_mode,
    angle_cosine_frequency_residual_max,
    quiet,
):

    vals = {}
    success = False
    if rhs and lhs:
        for mode_i in mode:
            val = 0.0
            if mode_i == "sum_difference":
                val = np.sum(rhs, axis=0) - np.sum(lhs, axis=0)
            if mode_i == "mag_difference":
                val = np.linalg.norm(np.sum(rhs, axis=0) - np.sum(lhs, axis=0))
            elif mode_i == "mean_difference":
                if key == "measure":
                    stats_fn.set_circular_from_label(lbl)
                    if lbl[0] in "ait":
                        if lbl[0] == "a" or angle_mode == "angle":
                            A, B = map(stats_fn.mean, [rhs, lhs])
                            A, B = map(lambda x: x - 180 if x > 180 else x, [A, B])
                            val = A - B
                        elif angle_mode == "cosine_residuals":
                            n = np.arange(angle_cosine_frequency_residual_max).reshape(
                                1, -1
                            )
                            A = np.cos(n * np.radians(np.atleast_2d(rhs).T)).sum(
                                axis=0
                            ) / len(rhs)
                            B = np.cos(n * np.radians(np.atleast_2d(lhs).T)).sum(
                                axis=0
                            ) / len(lhs)
                            val = np.linalg.norm(A - B)
                        # lhs = list(map(lambda x: x - 360 if x < 0 else x, lhs))
                        # rhs = list(map(lambda x: x - 360 if x < 0 else x, rhs))
                    else:
                        A, B = map(stats_fn.mean, [rhs, lhs])
                        val = A - B
                else:
                    stats_fn.set_circular(False)
                    A, B = map(stats_fn.mean, [rhs, lhs])
                    val = A - B

                # if lbl[0] in "ait":
                #     lhs = list(map(lambda x: x + 180 if x < 180 else x, lhs))
                #     rhs = list(map(lambda x: x + 180 if x < 180 else x, rhs))
                # val = np.mean(rhs, axis=0) - np.mean(lhs, axis=0)

            # this should only activate to scale the values
            if key == "measure":
                if denoms[lbl[0]] == 0.0:
                    val = 0.0
                else:
                    # only scale if using the raw angle values
                    val = val / denoms[lbl[0]]
            vals[mode_i] = val

        # default is to score by the first mode supplied
        val = vals[mode[0]]
        if eps is None:
            success = True
        else:
            success = np.abs(val) > eps

        try:
            success = bool(success)
        except ValueError:
            # we could use all or any here. Using any allows, for a bond for
            # example, either the length or the force to accepted. If we use
            # all, both terms must be above eps. This is only important for
            # sum_difference, which could provide multiple values.
            success = any(success)
    return idx, vals, success


def bit_gradient_factory():
    return offsb.tools.sorted_collection.SortedCollection(
        key=lambda x: (
            np.min(np.abs(x[5])),
            len(x[1].nodes) if x[6] in ["delete"] else -len(x[1].nodes),
            x[1].bits(maxbits=True),
            x[7],
        )
    )


class ChemicalSpace(offsb.treedi.tree.Tree):
    class stats_fn:

        mean = np.mean
        var = np.var

        @classmethod
        def set_circular(cls, val: bool):
            if val:
                cls.mean, cls.var = cls.circmean, cls.circvar
            else:
                cls.mean, cls.var = np.mean, np.var

        @classmethod
        def circvar(cls, *args, **kwargs):
            rad = np.radians(*args)
            val = scipy.stats.circvar(rad, **kwargs)
            return np.degrees(val)

        @classmethod
        def circmean(cls, *args, **kwargs):
            rad = np.radians(*args)
            val = scipy.stats.circmean(rad, **kwargs)
            return np.degrees(val)

        @classmethod
        def stats(cls, dat):
            return cls.mean(dat, axis=0), cls.var(dat, axis=0), np.sum(dat, axis=0)

        @classmethod
        def set_circular_from_label(cls, lbl: str):

            stats_fn_tab = {
                "a": True,
                "t": True,
                "i": True,
            }
            circular = stats_fn_tab.get(lbl[0], False)
            cls.set_circular(circular)

    def __init__(
        self,
        ff_fname,
        obj,
        root_payload=None,
        node_index=None,
        db=None,
        payload=None,
        verbose=False,
        openff_compat=True,
    ):
        self.verbose = verbose
        if self.verbose:
            print("Building ChemicalSpace")
        if isinstance(obj, str):
            super().__init__(
                obj,
                root_payload=root_payload,
                node_index=node_index,
                db=db,
                payload=payload,
                verbose=verbose,
            )

        self.ffname = ff_fname

        self.openff_compat = openff_compat

        self._prim_clusters = dict()
        self._param_data_cache = None
        self._ref_data_cache = None
        self._labeler = None

        self._ic = None
        self._fc = None
        self._prim = None
        self._ic_prim = None

        self._ignore_parameters = []

        # default try to make new types for these handlers
        self.parameterize_handlers = [
            "vdW",
            "Bonds",
            "Angles",
            "ImproperTorsions",
            "ProperTorsions",
        ]

        self.bit_search_limit = 4
        self.optimize_candidate_limit = 1
        self.split_candidate_limit = None  # None to disable
        self.score_candidate_limit = None  # None to disable
        self.gradient_assigned_only = True
        self.reject_limit = 0  # unlimited

        # if we optimize each split, the split must fall below e.g -.1 (-10%)
        self.split_keep_threshhold = -0.1

        # Calculate the theoretical best splits and rank scores
        self.calculate_score_rank = False

        self.score_mode = [
            "min",
            "max",
            "abs_min",
            "abs_max",
            "single-point-gradient-max",
        ]

        self.split_mode = self.score_mode[3]
        self.score_mode = self.score_mode[3]

        self.trust0 = 0.25
        self.eig_lowerbound = 1e-5
        self.finite_difference_h = 0.01

        self.atom_universe = offsb.chem.types.AtomType()
        if self.openff_compat:
            self.atom_universe.disable("aA")
        self.bond_universe = offsb.chem.types.BondType()

        self.denoms = {
            "n": VDW_DENOM,
            "b": 0.1 * offsb.tools.const.angstrom2bohr,
            "a": 10,
            "i": 10,
            "t": 10,
        }

        self.angle_mode = ["angle", "cosine_residuals"]
        self.angle_mode = self.angle_mode[0]
        self.angle_cosine_frequency_residual_max = 30

        self._po = None
        self._to = None

    def to_pickle(self, db=True, index=True, name=None):
        po, self._po = self._po, None
        to, self._to = self._to, None
        super().to_pickle(db=db, index=index, name=name)
        self._po = po
        self._to = to

    def to_pickle_str(self):
        pass

    def to_smirnoff(
        self, verbose=True, renumber=False, sanitize=False, hide_ignored=True
    ):

        ff = self.db["ROOT"]["data"]

        ph_nodes = [self[x] for x in self.root().children]

        ph_to_letter = {
            "vdW": "n",
            "Angles": "a",
            "Bonds": "b",
            "ProperTorsions": "t",
            "ImproperTorsions": "i",
        }

        # since we expanded all bits which point to the same param
        # only print the most general parameter (FIFO)
        visited = set()

        def visible_node_depth(node):
            return len(set(list([n.payload for n in self.node_iter_to_root(node)]))) - 2

        if verbose:
            print("Here is the current FF hierarchy")
        for ph_node in ph_nodes:
            # if ph_node.payload != "Bonds":
            #     continue
            if verbose:
                print("  " * self.node_depth(ph_node), ph_node)
            # print("Parsing", ph_node)
            ff_ph = ff.get_parameter_handler(ph_node.payload)
            params = []

            # ah, breadth first doesn't take the current level, depth does
            # so it is pulling the ph_nodes into the loop
            # but guess what! depth first puts it into the entirely incorrect order

            num_map = {}

            # for i, param_node in enumerate(self.node_iter_breadth_first(ph_node), 1):
            p_idx = 1
            for i, param_node in enumerate(self.node_iter_dive(ph_node), 1):
                if param_node.payload not in self.db:
                    print("This param not in the db???!", param_node.payload)
                    continue
                if param_node == ph_node:
                    continue
                param = self.db[param_node.payload]["data"]["parameter"]
                if param.id in visited:
                    # print("PARAM VISITED; SKIPPING", param.id)
                    continue
                visited.add(param.id)
                param = copy.deepcopy(param)
                if renumber:
                    num_map[param.id] = ph_to_letter[ph_node.payload] + str(p_idx)
                    p_idx += 1
                    param.id = num_map[param.id]
                if sanitize:
                    param._cosmetic_attribs.clear()
                # print("FF Appending param", param.id, param.smirks)
                params.append(param)
            ff_ph._parameters = ParameterList(params)

            # this is for printing
            if not verbose:
                continue
            p_idx = 1
            for i, param_node in enumerate(self.node_iter_dive(ph_node), 1):
                if param_node.payload not in self.db:
                    continue
                if param_node == ph_node:
                    continue
                if hide_ignored and param_node.payload in self._ignore_parameters:
                    continue
                param = copy.deepcopy(self.db[param_node.payload]["data"]["parameter"])

                if visible_node_depth(param_node) > 0:
                    print("->", end="")
                print(
                    "    " * (visible_node_depth(param_node) - 1)
                    + "{:8d} ".format(self.node_depth(param_node))
                    + "{}".format(param_node)
                )
                ff_param = copy.deepcopy(
                    self.db[param_node.payload]["data"]["parameter"]
                )
                if renumber:
                    ff_param.id = num_map[param.id]
                # ff_param.smirks = ""
                # print(
                #     "  " + "      " * visible_node_depth(param_node),
                #     self.db[param_node.payload]["data"]["group"],
                # )
                smarts = (
                    ff_param.smirks
                )  # self.db[param_node.payload]["data"]["group"].to_smarts()
                print(
                    "  " + "      " * visible_node_depth(param_node),
                    "{:14s} : {}".format("smarts", smarts),
                )
                print(
                    "  " + "      " * visible_node_depth(param_node),
                    "{:14s} : {}".format("id", ff_param.id),
                )
                for k, v in ff_param.__dict__.items():
                    k = str(k).lstrip("_")
                    if any([k.startswith(x) for x in ["cosmetic", "smirks", "id"]]):
                        continue

                    # list of Quantities...
                    if issubclass(type(v), list):
                        v = " ".join(["{}".format(x.__str__()) for x in v])

                    print(
                        "  " + "      " * visible_node_depth(param_node),
                        "{:12s} : {}".format(k, v),
                    )

        return ff

    def to_smirnoff_xml(
        self, output, verbose=True, renumber=False, sanitize=False, hide_ignored=True
    ):

        ff = self.to_smirnoff(
            verbose=verbose,
            renumber=renumber,
            sanitize=sanitize,
            hide_ignored=hide_ignored,
        )
        if output:
            ff.to_file(output)

    @classmethod
    def from_smirnoff_xml_graph(
        cls, input, name=None, add_generics=False, verbose=False
    ):

        prim_to_graph = {
            "n": offsb.chem.graph.AtomEnvironment,
            "b": offsb.chem.graph.BondEnvironment,
            "a": offsb.chem.graph.AngleEnvironment,
            "i": offsb.chem.graph.OutOfPlaneEnvironment,
            "t": offsb.chem.graph.TorsionEnvironment,
        }
        ff = ForceField(input, allow_cosmetic_attributes=True)
        cls = cls(input, name, root_payload=ff, verbose=False)

        cls.openff_compat = True
        cls.atom_universe.disable("aA")
        # need to read in all of the parameters

        handlers = ["vdW", "Bonds", "Angles", "ProperTorsions", "ImproperTorsions"]

        label_re = re.compile("([a-z])([0-9]+)(.*)")

        def sort_handler(x):
            x = label_re.fullmatch(
                cls.db[cls[x].payload]["data"]["parameter"].id
            ).groups()
            return (x[0], int(x[1]), x[2])

        root = cls.root()
        for ph_name in [
            ph for ph in ff.registered_parameter_handlers if ph in handlers
        ]:
            ph = ff.get_parameter_handler(ph_name)

            node = offsb.treedi.node.Node(name=ph_name, payload=ph_name)
            node = cls.add(root.index, node)
            cls.db[ph_name] = DEFAULT_DB({"data": ph})

            generic = []
            if add_generics:
                param = generic_smirnoff_params.get(ph_name)
                if param:
                    generic = [param]

            nodes = {}

            if cls.verbose:
                print("Loading parameters for", ph)
            for param in generic + ph.parameters:
                param_letter = param.id[0]
                # group_type = prim_to_group[param_letter]
                graph_type = prim_to_graph[param_letter]
                # if param.id ==  "b3":
                #     breakpoint()
                try:
                    # group = group_type.from_string(param.smirks)
                    graph = graph_type.from_smarts(param.smirks)
                    if cls.openff_compat:
                        graph.atom_disable("aA")
                    group = graph
                except Exception as e:
                    breakpoint()
                    print(
                        "Could not build group for id",
                        param.id,
                        "\nsmirks",
                        param.smirks,
                    )
                    continue

                if not graph.is_valid():
                    print(
                        "Invalid graph produced by id",
                        param.id,
                        "\nsmirks",
                        param.smirks,
                        "\ngraph",
                        graph,
                    )
                    continue

                pnode = offsb.treedi.node.Node(
                    name=str(type(param)).split(".")[-1], payload=param.id
                )

                # in order to separate out similar terms (like b83 and b84),
                # look for the marginals and split from that
                # take the sum, then iterate over the sum marginal
                # if both have them, need to repeat with other node
                # repeat until both are not in a group

                # print("For this group \n", group, "\n the hash is ", hash(group))
                if group in nodes:
                    # print("This group already covered! param id is", param.id, "Group is \n", group)
                    # print([p[1].id for p in nodes[group]])
                    if not any([param.id == x[1].id for x in nodes[group]]):
                        nodes[group].append((pnode, param))
                    else:
                        print(
                            "Warning: parameter",
                            param.id,
                            "already in tree; refusing to overwrite",
                        )
                else:
                    # print("Adding group for id", param.id, "group is \n", group)
                    nodes[group] = [(pnode, param)]

            # now iterate through the nodes and create the hierarchy

            # do a pairwise contains test, largest is one who is a subset of all the others

            all_nodes = list(nodes.keys())

            node_list = list(nodes.keys())
            compare = np.zeros(len(node_list))
            if cls.verbose:
                print("Determining hierarchy for", ph)
            for i, groupA in enumerate(node_list):
                for j, groupB in enumerate(all_nodes):
                    if groupB in groupA:
                        compare[i] += 1
            s = np.argsort(compare)[::-1]
            for i in s:

                largest_group = node_list[i]  # sorted(nodes, reverse=True)[0]
                parent = node

                for parent_node in cls.node_iter_depth_first(node):
                    if parent_node == node:
                        continue
                    if largest_group in cls.db[parent_node.payload]["data"]["group"]:
                        parent = parent_node
                        break

                param_nodes = nodes.pop(largest_group)

                for (pnode, param) in param_nodes:
                    if cls.verbose:
                        print("Adding parameter", param.id, "under", parent)
                    pnode = cls.add(parent.index, pnode)
                    cls.db[param.id] = DEFAULT_DB(
                        {"data": {"parameter": param, "group": largest_group}}
                    )

            # next stage: for nodes on the same level, sort by their param name
            # for example, if a level has a44, a22, a11, then the SMIRNOFF
            # spec is ordered such that a11 would appear first (least precedence)

            # also try to deal with those pesky a22, a22a things

            # maybe want to sorted based on first-in appearance, since we could
            # technically have a22b come before a22a...

            all_nodes = [n for n in cls.node_iter_depth_first(node)]

            if cls.verbose:
                print("Sorting parameters as they appear in the XML...")
            for param_node in all_nodes:
                if len(param_node.children) > 1:

                    if cls.verbose:
                        print("Sorting parameter", param_node.payload)

                    param_ids = [
                        cls.db[cls[x].payload]["data"]["parameter"].id
                        for x in param_node.children
                    ]
                    if cls.verbose:
                        print("Existing sort:", param_ids)

                    param_node.children = sorted(param_node.children, key=sort_handler)

                    param_ids = [
                        cls.db[cls[x].payload]["data"]["parameter"].id
                        for x in param_node.children
                    ]
                    if cls.verbose:
                        print("sorted:", param_ids)
                else:
                    if cls.verbose:
                        print("Only one child for parameter", param_node.payload)

            # last stage
            # In order to keep the right track of things, we need to expand
            # a node by all the bits, so it is placed on the correct level
            # This is needed when we create new parameters, since a split
            # could create a new level which would override many other parameters
            # even if those parameters are more specific

            # take a parent and its child, and add a list of nodes that turn off
            # a bit one by one, establishing the correct level in the tree

            # this is too expensive for general graphs, and the fact that
            # each param has a different number of nodes, so counting bits
            # is less meaningful

            if False:
                for parent_node in all_nodes:
                    if parent_node == node:
                        continue
                    if cls.verbose:
                        print("Expanding bit hierarchy for param", parent_node)
                    g_parent = cls.db[parent_node.payload]["data"]["group"]
                    if cls.verbose:
                        print(g_parent)
                    param_parent = cls.db[parent_node.payload]["data"]["parameter"]
                    for idx, child in enumerate([cls[x] for x in parent_node.children]):
                        if cls.verbose:
                            print(idx, child)
                        g_child = cls.db[child.payload]["data"]["group"]

                        # enumerate the marginal
                        bit_node = None
                        new_group = g_parent.copy()
                        prev_node = parent_node
                        n_idx = idx

                        M = g_parent.map_to(g_child)
                        if False:
                            # this generates the long chain of params with
                            # single bits turned off
                            for bit in (g_parent - g_child).iter_bits():

                                new_group = new_group.subtract(bit, map=M)
                                # make sure we dont add the child node
                                if not new_group.subtract(g_child, map=M).any():
                                    break

                                bit_node = offsb.treedi.node.Node(
                                    name=str(type(param_parent)),
                                    payload=param_parent.id,
                                )
                                # print("      bit",new_group)

                                # replace the position in the parent, then for subsequent
                                # nodes, just append
                                bit_node = cls.add(
                                    prev_node.index, bit_node, index=n_idx
                                )
                                prev_node = bit_node
                                n_idx = None
                        else:
                            bit_node = offsb.treedi.node.Node(
                                name=str(type(param_parent)), payload=param_parent.id
                            )

                        # if we have a string of bit nodes, then proceed
                        # "pop" the child from the parent; we need to be the child
                        # of the last bit node we enter
                        if bit_node:
                            # del parent_node.children[idx]
                            child = cls.add(parent_node.index, child)
                        # print(
                        #     "    bit delta:",
                        #     (g_parent - g_child).bits(maxbits=True),
                        #     "for child",
                        #     cls.db[child.payload]["data"]["parameter"],
                        # )
                        # print("    child", cls.db[child.payload]['data']['group'])
                        # print("    child", cls.db[child.payload]['data']['parameter'])

        return cls

    @classmethod
    def from_smirnoff_xml(
        cls, input, name=None, add_generics=False, graph=False, verbose=False
    ):

        """
        add parameter ids to db
        each db entry has a ff param (smirks ignored) and a group
        """

        ff = ForceField(input, allow_cosmetic_attributes=True)
        cls = cls(input, name, root_payload=ff, verbose=verbose)

        if graph:
            return cls.from_smirnoff_xml_graph(
                input, name=name, add_generics=add_generics, verbose=verbose
            )

        # need to read in all of the parameters

        handlers = ["vdW", "Bonds", "Angles", "ProperTorsions", "ImproperTorsions"]

        label_re = re.compile("([a-z])([0-9]+)(.*)")

        def sort_handler(x):
            x = label_re.fullmatch(
                cls.db[cls[x].payload]["data"]["parameter"].id
            ).groups()
            return (x[0], int(x[1]), x[2])

        root = cls.root()
        for ph_name in [
            ph for ph in ff.registered_parameter_handlers if ph in handlers
        ]:
            ph = ff.get_parameter_handler(ph_name)

            node = offsb.treedi.node.Node(name=ph_name, payload=ph_name)
            node = cls.add(root.index, node)
            cls.db[ph_name] = DEFAULT_DB({"data": ph})

            generic = []
            if add_generics:
                param = generic_smirnoff_params.get(ph_name)
                if param:
                    generic = [param]

            nodes = {}

            for param in generic + ph.parameters:
                param_letter = param.id[0]
                group_type = prim_to_group[param_letter]
                graph_type = prim_to_graph[param_letter]
                # if param.id ==  "b3":
                #     breakpoint()
                try:
                    group = group_type.from_string(param.smirks)
                    graph = graph_type.from_string(param.smirks)
                    group = graph
                except Exception as e:
                    print(
                        "Could not build group for id",
                        param.id,
                        "\nsmirks",
                        param.smirks,
                    )
                    continue

                if not graph.is_valid():
                    print(
                        "Invalid graph produced by id",
                        param.id,
                        "\nsmirks",
                        param.smirks,
                        "\ngraph",
                        graph,
                    )
                    continue

                pnode = offsb.treedi.node.Node(
                    name=str(type(param)).split(".")[-1], payload=param.id
                )

                # in order to separate out similar terms (like b83 and b84),
                # look for the marginals and split from that
                # take the sum, then iterate over the sum marginal
                # if both have them, need to repeat with other node
                # repeat until both are not in a group

                # print("For this group \n", group, "\n the hash is ", hash(group))
                if group in nodes:
                    # print("This group already covered! param id is", param.id, "Group is \n", group)
                    # print([p[1].id for p in nodes[group]])
                    if not any([param.id == x[1].id for x in nodes[group]]):
                        nodes[group].append((pnode, param))
                    else:
                        print(
                            "Warning: parameter",
                            param.id,
                            "already in tree; refusing to overwrite",
                        )
                else:
                    # print("Adding group for id", param.id, "group is \n", group)
                    nodes[group] = [(pnode, param)]

            # now iterate through the nodes and create the hierarchy

            # do a pairwise contains test, largest is one who is a subset of all the others

            all_nodes = list(nodes.keys())

            node_list = list(nodes.keys())
            compare = np.zeros(len(node_list))
            for i, groupA in enumerate(node_list):
                for j, groupB in enumerate(all_nodes):
                    if groupB in groupA:
                        compare[i] += 1
            s = np.argsort(compare)[::-1]
            for i in s:

                largest_group = node_list[i]  # sorted(nodes, reverse=True)[0]
                parent = node

                for parent_node in cls.node_iter_depth_first(node):
                    if parent_node == node:
                        continue
                    if largest_group in cls.db[parent_node.payload]["data"]["group"]:
                        parent = parent_node
                        break

                param_nodes = nodes.pop(largest_group)

                for (pnode, param) in param_nodes:
                    print("Adding parameter", param.id, "under", parent)
                    pnode = cls.add(parent.index, pnode)
                    cls.db[param.id] = DEFAULT_DB(
                        {"data": {"parameter": param, "group": largest_group}}
                    )

            # next stage: for nodes on the same level, sort by their param name
            # for example, if a level has a44, a22, a11, then the SMIRNOFF
            # spec is ordered such that a11 would appear first (least precedence)

            # also try to deal with those pesky a22, a22a things

            # maybe want to sorted based on first-in appearance, since we could
            # technically have a22b come before a22a...

            all_nodes = [n for n in cls.node_iter_depth_first(node)]

            print("Sorting parameters as they appear in the XML...")
            for param_node in all_nodes:
                if len(param_node.children) > 1:

                    print("Sorting parameter", param_node.payload)

                    param_ids = [
                        cls.db[cls[x].payload]["data"]["parameter"].id
                        for x in param_node.children
                    ]
                    print("Existing sort:", param_ids)

                    param_node.children = sorted(param_node.children, key=sort_handler)

                    param_ids = [
                        cls.db[cls[x].payload]["data"]["parameter"].id
                        for x in param_node.children
                    ]
                    print("sorted:", param_ids)
                else:
                    print("Only one child for parameter", param_node.payload)

            # last stage
            # In order to keep the right track of things, we need to expand
            # a node by all the bits, so it is placed on the correct level
            # This is needed when we create new parameters, since a split
            # could create a new level which would override many other parameters
            # even if those parameters are more specific

            # take a parent and its child, and add a list of nodes that turn off
            # a bit one by one, establishing the correct level in the tree
            for parent_node in all_nodes:
                if parent_node == node:
                    continue
                print("Expanding bit hierarchy for param", parent_node)
                g_parent = cls.db[parent_node.payload]["data"]["group"]
                print(g_parent)
                param_parent = cls.db[parent_node.payload]["data"]["parameter"]
                for idx, child in enumerate([cls[x] for x in parent_node.children]):
                    g_child = cls.db[child.payload]["data"]["group"]

                    # enumerate the marginal
                    bit_node = None
                    new_group = g_parent.copy()
                    prev_node = parent_node
                    n_idx = idx
                    for bit in g_parent - g_child:

                        new_group -= bit
                        # make sure we dont add the child node
                        if not (new_group - g_child).any():
                            break

                        bit_node = offsb.treedi.node.Node(
                            name=str(type(param_parent)), payload=param_parent.id
                        )
                        # print("      bit",new_group)

                        # replace the position in the parent, then for subsequent
                        # nodes, just append
                        bit_node = cls.add(prev_node.index, bit_node, index=n_idx)
                        prev_node = bit_node
                        n_idx = None

                    # if we have a string of bit nodes, then proceed
                    # "pop" the child from the parent; we need to be the child
                    # of the last bit node we enter
                    if bit_node:
                        del parent_node.children[idx]
                        child = cls.add(bit_node.index, child)
                    print(
                        "    bit delta:",
                        (g_parent - g_child).bits(maxbits=True),
                        "for child",
                        cls.db[child.payload]["data"]["parameter"],
                    )
                    # print("    child", cls.db[child.payload]['data']['group'])
                    # print("    child", cls.db[child.payload]['data']['parameter'])

        return cls

    @classmethod
    def default_from_smirnoff_xml(cls, input, name=None):

        """
        add parameter ids to db
        each db entry has a ff param (smirks ignored) and a group
        """
        ff = ForceField(input, allow_cosmetic_attributes=True)

        name = input if name is None else name
        cls = cls(input, name, root_payload=ff)
        # root = offsb.treedi.node.Node()

        root = cls.root()
        for pl_name in ff.registered_parameter_handlers:

            if pl_name == "vdW":

                node = offsb.treedi.node.Node(name=pl_name, payload=pl_name)
                node = cls.add(root.index, node)
                ph = vdWHandler(version="0.3")
                cls.db[cls.name + "-" + pl_name] = DEFAULT_DB({"data": ph})

                smirks = "[*:1]"
                param_dict = dict(
                    epsilon="0.05 * kilocalorie/mole",
                    rmin_half="1.2 * angstrom",
                    smirks=smirks,
                    id="n1",
                )

                ff.get_parameter_handler(pl_name)._parameters = ParameterList()
                ff.get_parameter_handler(pl_name).add_parameter(param_dict)

                param = vdWHandler.vdWType(**param_dict)

                param_name = param.id

                pnode = offsb.treedi.node.Node(
                    name=str(type(param)).split(".")[-1], payload=param_name
                )
                pnode = cls.add(node.index, pnode)
                cls.db[param_name] = DEFAULT_DB(
                    {
                        "data": {
                            "parameter": param,
                            "group": offsb.chem.types.AtomType.from_string(
                                param.smirks
                            ),
                        }
                    }
                )
            if pl_name == "Bonds":

                node = offsb.treedi.node.Node(name=pl_name, payload=pl_name)
                node = cls.add(root.index, node)
                ph = BondHandler(version="0.3")
                cls.db[cls.name + "-" + pl_name] = DEFAULT_DB({"data": ph})

                smirks = "[*:1]~[*:2]"

                param_dict = dict(
                    k="20.0 * kilocalorie/(angstrom**2*mole)",
                    length="1.2 * angstrom",
                    smirks=smirks,
                    id="b1",
                )

                param = BondHandler.BondType(**param_dict)

                ff.get_parameter_handler(pl_name)._parameters = ParameterList()
                ff.get_parameter_handler(pl_name).add_parameter(param_dict)

                param_name = param.id

                pnode = offsb.treedi.node.Node(
                    name=str(type(param)).split(".")[-1], payload=param_name
                )
                pnode = cls.add(node.index, pnode)
                group = offsb.chem.types.BondGroup.from_string(param.smirks)
                cls.db[param_name] = DEFAULT_DB(
                    {
                        "data": {
                            "parameter": param,
                            "group": group,
                        }
                    }
                )
            if pl_name == "Angles":

                node = offsb.treedi.node.Node(name=pl_name, payload=pl_name)
                node = cls.add(root.index, node)
                ph = AngleHandler(version="0.3")
                cls.db[cls.name + "-" + pl_name] = DEFAULT_DB({"data": ph})

                smirks = "[*:1]~[*:2]~[*:3]"
                param_dict = dict(
                    k="10.00 * kilocalorie/(degree**2*mole)",
                    angle="110.0 * degree",
                    smirks=smirks,
                    id="a1",
                )
                param = AngleHandler.AngleType(**param_dict)

                ff.get_parameter_handler(pl_name)._parameters = ParameterList()
                ff.get_parameter_handler(pl_name).add_parameter(param_dict)

                param_name = param.id

                pnode = offsb.treedi.node.Node(
                    name=str(type(param)).split(".")[-1], payload=param_name
                )
                pnode = cls.add(node.index, pnode)
                cls.db[param_name] = DEFAULT_DB(
                    {
                        "data": {
                            "parameter": param,
                            "group": offsb.chem.types.AngleGroup.from_string(
                                param.smirks
                            ),
                        }
                    }
                )
            if pl_name == "ProperTorsions":

                node = offsb.treedi.node.Node(name=pl_name, payload=pl_name)
                node = cls.add(root.index, node)
                ph = ProperTorsionHandler(version="0.3")
                cls.db[cls.name + "-" + pl_name] = DEFAULT_DB({"data": ph})

                smirks = "[*:1]~[*:2]~[*:3]~[*:4]"
                # param_dict = dict(
                #     periodicity=[1, 2, 3],
                #     k=[
                #         "0 * kilocalorie/mole",
                #         "0 * kilocalorie/mole",
                #         "0 * kilocalorie/mole",
                #     ],
                #     phase=["0.0 * degree", "0 * degree", "0 * degree"],
                #     smirks=smirks,
                #     id="t1",
                #     idivf=list([1.0] * 3),
                # )
                param_dict = dict(
                    periodicity=[1],
                    k=[
                        "0 * kilocalorie/mole",
                    ],
                    phase=["0.0 * degree"],
                    smirks=smirks,
                    id="t1",
                    idivf=list([1.0] * 1),
                )
                param = ProperTorsionHandler.ProperTorsionType(**param_dict)
                ff.get_parameter_handler(pl_name)._parameters = ParameterList()
                ff.get_parameter_handler(pl_name).add_parameter(param_dict)

                param_name = param.id

                pnode = offsb.treedi.node.Node(
                    name=str(type(param)).split(".")[-1], payload=param_name
                )
                pnode = cls.add(node.index, pnode)
                cls.db[param_name] = DEFAULT_DB(
                    {
                        "data": {
                            "parameter": param,
                            "group": offsb.chem.types.TorsionGroup.from_string(
                                param.smirks
                            ),
                        }
                    }
                )
            if pl_name == "ImproperTorsions":

                node = offsb.treedi.node.Node(name=pl_name, payload=pl_name)
                node = cls.add(root.index, node)
                ph = ImproperTorsionHandler(version="0.3")
                cls.db[cls.name + "-" + pl_name] = DEFAULT_DB({"data": ph})

                smirks = "[*:1]~[*:2](~[*:3])~[*:4]"
                param_dict = dict(
                    periodicity=[1],
                    k=["0.0 * kilocalorie/mole"],
                    phase=["0.000 * degree"],
                    smirks=smirks,
                    id="i1",
                    idivf=[1.0],
                )
                param = ImproperTorsionHandler.ImproperTorsionType(**param_dict)
                ff.get_parameter_handler(pl_name)._parameters = ParameterList()
                ff.get_parameter_handler(pl_name).add_parameter(param_dict)

                param_name = param.id

                pnode = offsb.treedi.node.Node(
                    name=str(type(param)).split(".")[-1], payload=param_name
                )
                pnode = cls.add(node.index, pnode)
                cls.db[param_name] = DEFAULT_DB(
                    {
                        "data": {
                            "parameter": param,
                            "group": offsb.chem.types.OutOfPlaneGroup.from_string(
                                param.smirks
                            ),
                        }
                    }
                )

        return cls

    def combine_parameter(self, node):

        parent = self[node.parent]

        children = []

        for i, c in enumerate(parent.children):
            if c == node.index:
                # this replaces the child with the children of the child into
                # the parent in the same position in the parent
                children = node.children
                parent.children[i : i + 1] = node.children
                break
        # self.db.pop(node.payload)

        ret = self.node_index.pop(node.index, None)
        print("Removed parameter", node, "status:", ret is not None)

        for c in children:
            self[c].parent = parent.index

        # if len(children) == 0:
        #     print("ERROR on 1066:")
        #     print("    Children of parent param", parent.payload, "is", parent.children)
        #     print("   ", [self[c] for c in parent.children])
        #     print("    Node to delete is", node)
        #     print("    Error due to no node index in the children being equal to the node index to delete")

        return children

    def split_parameter(self, label, bit, smarts=None):

        node = list(
            [x for x in self.node_iter_depth_first(self.root()) if x.payload == label]
        )[0]
        data = self.db[node.payload]["data"]
        group = data["group"]
        param = data["parameter"]
        # child = group.subtract(bit)

        # child = group.union(bit, add_nodes=True)
        # child = group.intersection(bit)

        child = bit

        # if child == group:
        #     return None
        # if not child.is_valid():
        #     return None

        new_param = copy.deepcopy(param)

        param_name = label[0] + str(np.random.randint(1000, 9999))

        pnode = offsb.treedi.node.Node(
            name=str(type(param)).split(".")[-1], payload=param_name
        )

        # check to make sure the group isn't already present

        # for n in self.node_iter_depth_first(node):
        #     grp = self.db[n.payload]['data']['group']
        #     if grp == child:
        #         # this parameter is already in the FF
        #         return None

        # In order to try to modify the FF as little as possible by splits,
        # give the new param the least precendence on the level
        # This gives ties between this new param and existing params point to
        # the old param
        pnode = self.add(node.index, pnode, index=0)

        # param.smirks = group.drop(child).to_smarts()
        # param.smirks = group.to_smarts(tag=True, atom_universe=self.atom_universe, bond_universe=self.bond_universe)
        try:
            if smarts is None:
                new_param.smirks = child.to_smarts(
                    tag=True,
                    atom_universe=self.atom_universe,
                    bond_universe=self.bond_universe,
                )
            else:
                new_param.smirks = smarts
        except Exception as e:
            breakpoint()
        new_param.id = param_name

        self.db[param_name] = DEFAULT_DB(
            {"data": {"parameter": new_param, "group": child}}
        )
        return pnode

    def _join_param(
        self,
        param_data,
        parent,
        child,
        key=None,
        eps=1.0,
        mode="mag_difference",
        bit_gradients=None,
        ignore_bits=None,
    ):

        # just need to go through the
        ys_bit = []
        no_bit = []

        operation = "delete"

        denoms = self.denoms

        verbose = self.verbose

        if type(mode) is str:
            mode = [mode]

        if bit_gradients is None:
            bit_gradients = bit_gradient_factory()

        # bit_gradients = sorted(
        #     bit_gradients,
        #     key=,
        #     reverse=True,
        # )

        if ignore_bits is None:
            ignore_bits = {}

        # ys_bit is the parent, no_bit is the child
        ys_lbl = parent.payload
        no_lbl = child.payload
        lbl = child.payload
        group = self.db[child.payload]["data"]["group"]
        smarts = group.to_smarts(
            atom_universe=self.atom_universe, bond_universe=self.bond_universe
        )

        # ignore_key = (lbl, group, operation)
        if any(x[0] == lbl and x[2] == "delete" for x in ignore_bits):
            # print("DEBUG 2: SKIPPING")
            return

        new_val = [lbl, group, {}, parent, child, np.inf, "delete", len(smarts)]

        if lbl not in param_data:

            if not any(x[0] == lbl and x[2] == "delete" for x in bit_gradients):
                # ignore_bits.setdefault((lbl, group, operation), list()).append(
                #     (None, None)
                # )
                print(f"Added {new_val}")
                print(f"Because {lbl} did not have any data")
                bit_gradients.insert(tuple(new_val))
            return

        stats_fn = self.stats_fn
        if lbl[0] in "ti" and key == "measure":
            stats_fn.set_circular(True)
        else:
            stats_fn.set_circular(False)

        if ys_lbl in param_data:
            for i, (prim, dat) in enumerate(param_data[ys_lbl].items()):
                if key is not None:
                    dat = dat[key]
                # pdb.set_trace()

                if len(dat) == 0:
                    continue
                if verbose:
                    try:
                        smarts = prim.to_smarts()
                    except Exception as e:
                        breakpoint()
                # ff_subtract = group.subtract(bit)

                # not in means that it is not in the new parameter, meaning it should
                # still have the parent parameter assigned
                if verbose:
                    _ = ""
                    print(
                        f"{_:20s}parent_param (N={len(dat):4d})",
                        "mean: {:12.5e} var: {:12.5e} sum: {:12.5e}".format(
                            *stats_fn.stats(dat)
                        ),
                        smarts,
                    )
                ys_bit.extend(dat)

        if no_lbl in param_data:
            for i, (prim, dat) in enumerate(param_data[no_lbl].items()):
                if key is not None:
                    dat = dat[key]
                # pdb.set_trace()

                if len(dat) == 0:
                    continue

                if verbose:
                    try:
                        smarts = prim.to_smarts()
                    except Exception as e:
                        breakpoint()

                else:
                    if verbose:
                        print(
                            f"{_:20s} child_param (N={len(dat):4d})",
                            "mean: {:8.4e} var: {:8.4e} sum: {:8.4e}".format(
                                *stats_fn.stats(dat)
                            ),
                            smarts,
                        )
                no_bit.extend(dat)

        if verbose:
            print("{:18s}Parent group (N=".format(""), len(ys_bit), ") :", end=" ")
        if len(ys_bit) > 0:
            pass
            if verbose:
                print("mean: {} var: {} sum: {}".format(*stats_fn.stats(ys_bit)))
        else:
            pass
            if verbose:
                print("None")
        # print(ys_bit)
        if verbose:
            print("{:18s}Child group    (N=".format(""), len(no_bit), ") :", end=" ")
        if len(no_bit) > 0 and len(ys_bit) > 0:
            if verbose:
                print("mean: {} var: {} sum: {}".format(*stats_fn.stats(no_bit)))

            lhs = no_bit
            rhs = ys_bit

            vals = {}
            for mode_i in mode:
                val = 0.0
                if mode_i == "sum_difference":
                    val = np.sum(rhs, axis=0) - np.sum(lhs, axis=0)
                if mode_i == "mag_difference":
                    val = np.linalg.norm(np.sum(rhs, axis=0) - np.sum(lhs, axis=0))
                elif mode_i == "mean_difference":
                    if key == "measure":
                        stats_fn.set_circular_from_label(lbl)
                        if lbl[0] in "ait":
                            if lbl[0] == "a" or self.angle_mode == "angle":
                                A, B = map(stats_fn.mean, [rhs, lhs])
                                A, B = map(lambda x: x - 180 if x > 180 else x, [A, B])
                                val = A - B
                            elif self.angle_mode == "cosine_residuals":
                                n = np.arange(
                                    self.angle_cosine_frequency_residual_max
                                ).reshape(1, -1)
                                A = np.cos(n * np.radians(np.atleast_2d(rhs).T)).sum(
                                    axis=0
                                ) / len(rhs)
                                B = np.cos(n * np.radians(np.atleast_2d(lhs).T)).sum(
                                    axis=0
                                ) / len(lhs)
                                val = np.linalg.norm(A - B)
                            # A, B = map(stats_fn.mean, [rhs, lhs])
                            # A = stats_fn.mean(rhs)
                            # B = stats_fn.mean(lhs)
                            # A, B = map(lambda x: x if x > 0 else x - 360, [A,B])
                            # lhs = list(map(lambda x: x - 360 if x < 0 else x, lhs))
                            # rhs = list(map(lambda x: x - 360 if x < 0 else x, rhs))
                        else:
                            A, B = map(stats_fn.mean, [rhs, lhs])
                            val = A - B
                    else:
                        stats_fn.set_circular(False)
                        A, B = map(stats_fn.mean, [rhs, lhs])
                        val = A - B

                if key == "measure":
                    if denoms[lbl[0]] == 0.0:
                        val = 0.0
                    else:
                        val = val / denoms[lbl[0]]

                vals[mode_i] = val

            # default is to score by the first mode supplied
            val = vals[mode[0]]

            # note this is essentially inverted from the split success
            if eps is None:
                success = True
            else:
                success = np.abs(val) <= eps

            try:
                success = bool(success)
            except ValueError:
                # we could use all or any here. Using any allows, for a bond for
                # example, either the length or the force to accepted. If we use
                # all, both terms must be above eps. This is only important for
                # sum_difference, which could provide multiple values.
                success = any(success)
            if verbose:
                print("{:20s}Score: {} eps: {} split? {}".format("", val, eps, success))
            if success:

                # invert the values so they prioritized in reverse order compared
                # to splits
                new_val[2] = vals
                new_val[5] = 1 / vals[mode[0]]

                if not any(x[0] == ys_lbl and x[6] == "delete" for x in bit_gradients):

                    # ignore_bits.setdefault((lbl, group, operation), list()).append(
                    #     (None, None)
                    # )
                    bit_gradients.insert(tuple(new_val))
        elif len(no_bit) == 0 and len(ys_bit) > 0:
            if not any(x[0] == no_lbl and x[6] == "delete" for x in bit_gradients):
                # ignore_bits.setdefault((lbl, group, operation), list()).append(
                #     (None, None)
                # )
                bit_gradients.insert(tuple(new_val))
        elif len(ys_bit) > 0 and len(no_bit) == 0:

            if not any(x[0] == ys_lbl and x[6] == "delete" for x in bit_gradients):
                # ignore_bits.setdefault((lbl, group, operation), list()).append(
                #     (None, None)
                # )
                bit_gradients.insert(tuple(new_val))

    def _rank_scores(
        self,
        param_data,
        lbl,
        group,
        bit,
        key=None,
        eps=1.0,
        mode="mag_difference",
        bit_gradients=None,
        ignore_bits=None,
        quiet=True,
    ):
        groups = self._prim_clusters.get(lbl, None)

        denoms = self.denoms
        verbose = self.verbose

        if self.calculate_score_rank and groups is None:
            prims = list(param_data[lbl])

            # hashing the prims is expensive so cache it
            param_array = [param_data[lbl][prims[j]] for j in range(len(prims))]

            groups = {}
            # if lbl == 'b6':
            #     breakpoint()
            if verbose:
                print("This label has {:d} primitives".format(len(prims)))
            n_groups = len(range(1, (len(prims) + 1) // 2 + 1))
            for i in range(1, (len(prims) + 1) // 2 + 1):
                iterable = list(itertools.combinations(range(len(prims)), i))
                n_iterable = len(iterable)
                for j, group_a in tqdm.tqdm(
                    enumerate(iterable),
                    total=n_iterable,
                    ncols=80,
                    desc="Group scan {}/{}".format(i, n_groups),
                    disable=not verbose,
                ):
                    group_b = tuple(j for j in range(len(prims)) if j not in group_a)
                    group_key = tuple(sorted((group_a, group_b)))
                    if group_key in groups:
                        continue

                    lhs = np.vstack([param_array[j] for j in group_a])
                    # lhs = np.sum(lhs, axis=0)
                    rhs = np.vstack([param_array[j] for j in group_b])
                    # rhs = np.sum(rhs, axis=0)

                    vals = {}
                    for mode_i in mode:
                        val = 0.0
                        if mode_i == "sum_difference":
                            val = np.sum(rhs, axis=0) - np.sum(lhs, axis=0)
                        if mode_i == "mag_difference":
                            val = np.linalg.norm(
                                np.sum(rhs, axis=0) - np.sum(lhs, axis=0)
                            )
                        elif mode_i == "mean_difference":
                            if key == "measure":
                                if lbl[0] in "ait":
                                    lhs = list(
                                        map(lambda x: x + 180 if x < 180 else x, lhs)
                                    )
                                    rhs = list(
                                        map(lambda x: x + 180 if x < 180 else x, rhs)
                                    )
                            val = np.mean(rhs, axis=0) - np.mean(lhs, axis=0)
                        if key == "measure":
                            if denoms[lbl[0]] == 0.0:
                                val = 0.0
                            else:
                                val = val / denoms[lbl[0]]
                        vals[mode_i] = val

                    groups[group_key] = vals
            groups = [(a, b, c) for (a, b), c in groups.items()]

            for mode_i in mode:
                groups_sorted = sorted(
                    groups, key=lambda x: np.max(np.abs(x[2][mode_i])), reverse=True
                )
                print(
                    "\n\nCluster analysis of primitives; max,min predicted split would be for mode {:s}".format(
                        mode_i
                    )
                )
                if len(groups_sorted) > 0:
                    # for grp in groups_sorted:
                    #     print(grp)
                    print(groups_sorted[0])
                    print(groups_sorted[-1])
                    print(
                        "Total permutations for {} primitives: {}".format(
                            len(groups_sorted[0][0]) + len(groups_sorted[0][1]),
                            len(groups_sorted),
                        )
                    )
                else:
                    print("None")
                print("\n")
            # for line in groups:
            #     print(line)

            # not that this whole thing was ripped out, so it likely doesn't work
            # self._prim_clusters[lbl] = groups
            #     groups = self._prim_clusters
            #     if len(groups):
            #         group_a = sorted(chosen_split[0])
            #         group_b = sorted(chosen_split[1])
            #         chosen_split = sorted(map(tuple, [group_a, group_b]))
            #         chosen_split = tuple(chosen_split)
            #         groups_sorted = sorted(
            #             groups,
            #             key=lambda x: np.max(np.abs(x[2][mode[0]])),
            #             reverse=True,
            #         )
            #         rank_pos = [
            #             i
            #             for i, v in enumerate(groups_sorted)
            #             if v[2][mode[0]] == vals[mode[0]]
            #         ][0]
            #         score_0_to_1 = (
            #             (vals[mode[0]] - groups_sorted[-1][2][mode[0]])
            #             / (groups_sorted[0][2][mode[0]] - groups_sorted[-1][2][mode[0]])
            #             if (
            #                 groups_sorted[0][2][mode[0]] - groups_sorted[-1][2][mode[0]]
            #             )
            #             else 1.0
            #         )
            #         rank = {
            #             "rank": rank_pos + 1,
            #             "rank_of": len(groups),
            #             "rank_1": groups_sorted[0][2][mode[0]],
            #             "score_0_to_1": score_0_to_1,
            #         }

    def split_from_mask(self, mask):
        A = []
        B = []
        for i, v in enumerate(mask):
            if v:
                A.append(i)
            else:
                B.append(v)
        return tuple(A), tuple(B)

    def bit_add_if_new(
        self,
        bit,
        group,
        smarts,
        lbl,
        vals,
        mode,
        mask,
        operation,
        bit_gradients,
        ignore_bits,
    ):

        verbose = True  # self.verbose

        rank = None
        chosen_split = self.split_from_mask(mask)
        new_val = (
            lbl,
            group.copy(),
            vals,
            chosen_split,
            None,
            vals[mode[0]],
            operation,
            -len(smarts),
        )
        duplicate = False
        success = False

        for i, x in enumerate(bit_gradients):
            # results show that even though we might have a specific split
            # pattern, the final assignments may differ if our SMARTS
            # were incomplete
            if (
                x[0] == lbl
                # and (any([x[1] == y for y in bit]) or bit not in x[1])
                and chosen_split == x[3]
            ):
                if group.bits() < x[1].bits():
                    # if verbose:
                    #     print(
                    #         "{:18s}Swapping existing result (bits=".format(""),
                    #         x[1].bits(),
                    #         ")",
                    #         x,
                    #         "with (bits=",
                    #         group.bits(),
                    #         ")",
                    #         new_val,
                    #     )
                    bit_gradients.remove(x)
                    bit_gradients.insert(new_val)
                    success = True

                    # if verbose:
                    #     print(
                    #         "{:18s}Ignoring".format(""),
                    #         lbl,
                    #         bit.compact_str(),
                    #     )
                    if (lbl, group, operation) in ignore_bits:
                        ignore_bits[(lbl, group, operation)].append((vals, None))
                    else:
                        ignore_bits[(lbl, group, operation)] = [(vals, None)]

                    if (lbl, x[1], operation) in ignore_bits:
                        ignore_bits[(x[0], x[1], operation)].append((x[2], None))
                    else:
                        ignore_bits[(x[0], x[1], operation)] = [(x[2], None)]

                    # only mark as a duplicate if we swapped
                    duplicate = True
                    break

        if not duplicate:
            # if verbose:
            #     print(
            #         "{:18s}Appending result (bits={})".format("", bit.bits()),
            #         bit.bits(),
            #         new_val[0],
            #         group.to_smarts(),
            #     )
            bit_gradients.insert(new_val)
            success = True
        else:
            # if verbose:
            #     print(
            #         "{:18s}Not appending result since a lower bit split produces the same result".format(
            #             ""
            #         )
            #     )
            if group != x[1]:
                # if verbose:
                #     print(
                #         "{:18s}Ignoring".format(""),
                #         lbl,
                #         group.to_smarts(),
                #     )
                if (lbl, group, operation) in ignore_bits:
                    ignore_bits[(lbl, group, operation)].append((vals, None))
                else:
                    ignore_bits[(lbl, group, operation)] = [(vals, None)]
        # if verbose:
        #     print("\n")
        return success

    def _scan_param_with_bit(
        self,
        param_data,
        lbl,
        group,
        bit,
        key=None,
        eps=1.0,
        mode="mag_difference",
        bit_gradients=None,
        ignore_bits=None,
        quiet=True,
    ):

        ys_bit = []
        no_bit = []

        operation = "new"

        denoms = self.denoms

        verbose = self.verbose

        if type(mode) is str:
            mode = [mode]

        if bit_gradients is None:
            bit_gradients = bit_gradient_factory()

        # bit_gradients = sorted(
        #     bit_gradients,
        #     key=,
        #     reverse=True,
        # )

        smarts = group.to_smarts(
            atom_universe=self.atom_universe, bond_universe=self.bond_universe
        )

        if ignore_bits is None:
            ignore_bits = {}

        stats_fn = self.stats_fn
        if lbl[0] in "ti" and key == "measure":
            stats_fn.set_circular(True)
        else:
            stats_fn.set_circular(False)

        if verbose:
            print("{:18s}group is".format(""), group.to_smarts())
        chosen_split = [[], []]
        for i, (prim, dat) in enumerate(param_data[lbl].items()):
            sma = dat["smarts"]
            if key is not None:
                dat = dat[key]
            # pdb.set_trace()

            if len(dat) == 0:
                continue

            if verbose:
                try:
                    smarts = sma  # prim.to_smarts()
                except Exception as e:
                    breakpoint()

            # not in means that it is not in the new parameter, meaning it should
            # still have the parent parameter assigned
            if prim not in group:
                if verbose:
                    _ = ""
                    print(
                        f"{_:20s}old_param (N={len(dat):4d})",
                        "mean: {:12.5e} var: {:12.5e} sum: {:12.5e}".format(
                            *stats_fn.stats(dat)
                        ),
                        smarts,
                    )
                chosen_split[0].append(i)
                ys_bit.extend(dat)
            else:
                if verbose:
                    _ = ""
                    print(
                        f"{_:20s}new_param (N={len(dat):4d})",
                        "mean: {:12.5e} var: {:12.5e} sum: {:12.5e}".format(
                            *stats_fn.stats(dat)
                        ),
                        smarts,
                    )
                chosen_split[1].append(i)
                no_bit.extend(dat)
        if not quiet:
            print("{:18s}Parent group (N=".format(""), len(ys_bit), ") :", end=" ")
        if len(ys_bit) > 0:
            if not quiet:
                print("mean: {} var: {} sum: {}".format(*stats_fn.stats(ys_bit)))
        else:
            if not quiet:
                print("None")
        # print(ys_bit)
        if not quiet:
            print("{:18s}New group    (N=".format(""), len(no_bit), ") :", end=" ")
        if len(no_bit) > 0 and len(ys_bit) > 0:
            if not quiet:
                print("mean: {} var: {} sum: {}".format(*stats_fn.stats(no_bit)))

            lhs = no_bit
            # lhs = np.sum(lhs, axis=0)
            rhs = ys_bit
            # rhs = np.sum(rhs, axis=0)

            angle_mode = self.angle_mode

            vals = {}
            for mode_i in mode:
                val = 0.0
                if mode_i == "sum_difference":
                    val = np.sum(rhs, axis=0) - np.sum(lhs, axis=0)
                if mode_i == "mag_difference":
                    val = np.linalg.norm(np.sum(rhs, axis=0) - np.sum(lhs, axis=0))
                elif mode_i == "mean_difference":
                    if key == "measure":
                        stats_fn.set_circular_from_label(lbl)
                        if lbl[0] in "ait":
                            if lbl[0] == "a" or self.angle_mode == "angle":
                                A, B = map(stats_fn.mean, [rhs, lhs])
                                A, B = map(lambda x: x - 180 if x > 180 else x, [A, B])
                                val = A - B
                            elif angle_mode == "cosine_residuals":
                                n = np.arange(
                                    self.angle_cosine_frequency_residual_max
                                ).reshape(1, -1)
                                A = np.cos(n * np.radians(np.atleast_2d(rhs).T)).sum(
                                    axis=0
                                ) / len(rhs)
                                B = np.cos(n * np.radians(np.atleast_2d(lhs).T)).sum(
                                    axis=0
                                ) / len(lhs)
                                val = np.linalg.norm(A - B)
                            # lhs = list(map(lambda x: x - 360 if x < 0 else x, lhs))
                            # rhs = list(map(lambda x: x - 360 if x < 0 else x, rhs))
                        else:
                            A, B = map(stats_fn.mean, [rhs, lhs])
                            val = A - B
                    else:
                        stats_fn.set_circular(False)
                        A, B = map(stats_fn.mean, [rhs, lhs])
                        val = A - B

                    # if lbl[0] in "ait":
                    #     lhs = list(map(lambda x: x + 180 if x < 180 else x, lhs))
                    #     rhs = list(map(lambda x: x + 180 if x < 180 else x, rhs))
                    # val = np.mean(rhs, axis=0) - np.mean(lhs, axis=0)

                # this should only activate to scale the values
                if key == "measure":
                    if denoms[lbl[0]] == 0.0:
                        val = 0.0
                    else:
                        # only scale if using the raw angle values
                        val = val / denoms[lbl[0]]
                vals[mode_i] = val

            # default is to score by the first mode supplied
            val = vals[mode[0]]
            if eps is None:
                success = True
            else:
                success = np.abs(val) > eps

            try:
                success = bool(success)
            except ValueError:
                # we could use all or any here. Using any allows, for a bond for
                # example, either the length or the force to accepted. If we use
                # all, both terms must be above eps. This is only important for
                # sum_difference, which could provide multiple values.
                success = any(success)
            if not quiet:
                print("{:20s}Score: {} eps: {} split? {}".format("", val, eps, success))
            if success:
                rank = None

                new_val = (
                    lbl,
                    group.copy(),
                    vals,
                    chosen_split,
                    None,
                    vals[mode[0]],
                    "new",
                    -len(smarts),
                )
                duplicate = False
                for i, x in enumerate(bit_gradients):
                    # results show that even though we might have a specific split
                    # pattern, the final assignments may differ if our SMARTS
                    # were incomplete
                    if (
                        x[0] == lbl
                        # and (any([x[1] == y for y in bit]) or bit not in x[1])
                        and chosen_split == x[3]
                    ):
                        if group.bits() < x[1].bits():
                            if verbose:
                                print(
                                    "{:18s}Swapping existing result (bits=".format(""),
                                    x[1].bits(),
                                    ")",
                                    x,
                                    "with (bits=",
                                    group.bits(),
                                    ")",
                                    new_val,
                                )
                            bit_gradients.remove(x)
                            bit_gradients.insert(new_val)

                            if verbose:
                                print(
                                    "{:18s}Ignoring".format(""),
                                    lbl,
                                    bit.compact_str(),
                                )
                            if (lbl, group, operation) in ignore_bits:
                                ignore_bits[(lbl, group, operation)].append(
                                    (vals, None)
                                )
                            else:
                                ignore_bits[(lbl, group, operation)] = [(vals, None)]

                            if (lbl, x[1], operation) in ignore_bits:
                                ignore_bits[(x[0], x[1], operation)].append(
                                    (x[2], None)
                                )
                            else:
                                ignore_bits[(x[0], x[1], operation)] = [(x[2], None)]

                            # only mark as a duplicate if we swapped
                            duplicate = True
                            break

                if not duplicate:
                    if verbose:
                        print(
                            "{:18s}Appending result (bits={})".format("", bit.bits()),
                            bit.bits(),
                            new_val[0],
                            group.to_smarts(),
                        )
                    bit_gradients.insert(new_val)
                else:
                    if verbose:
                        print(
                            "{:18s}Not appending result since a lower bit split produces the same result".format(
                                ""
                            )
                        )
                    if group != x[1]:
                        if verbose:
                            print(
                                "{:18s}Ignoring".format(""),
                                lbl,
                                group.to_smarts(),
                            )
                            if (lbl, group, operation) in ignore_bits:
                                ignore_bits[(lbl, group, operation)].append(
                                    (vals, None)
                                )
                            else:
                                ignore_bits[(lbl, group, operation)] = [(vals, None)]
                if verbose:
                    print("\n")
        else:
            if not quiet:
                print("None")

    def _check_overlapped_parameter(self, pre, post, node):

        parent_param = self[node.parent].payload
        child_param = node.payload

        n_pre = 0
        for entry in pre.db.values():
            entry = entry["data"]
            n_pre += sum(
                [
                    1
                    for ic_type in entry
                    for atoms, lbl in entry[ic_type].items()
                    if lbl == parent_param
                ]
            )

        n_post = 0
        for entry in post.db.values():
            entry = entry["data"]
            n_post += sum(
                [
                    1
                    for ic_type in entry
                    for atoms, lbl in entry[ic_type].items()
                    if lbl == parent_param
                ]
            )

        n_new_post = 0
        for entry in post.db.values():
            entry = entry["data"]
            n_new_post += sum(
                [
                    1
                    for ic_type in entry
                    for atoms, lbl in entry[ic_type].items()
                    if lbl == child_param
                ]
            )

        if n_pre >= 0 and n_post == 0 and n_new_post == n_pre:
            return True

        else:
            return False

    def _find_next_join(
        self,
        param_data,
        key=None,
        ignore_bits=None,
        mode=None,
        eps=1.0,
        bit_gradients=None,
        bit_cache=None,
    ):

        verbose = self.verbose

        operation = "delete"

        if mode is None:
            mode = ["mean_difference"]

        if bit_gradients is None:
            bit_gradients = bit_gradient_factory()
        if ignore_bits is None:
            ignore_bits = {}
        if bit_cache is None:
            bit_cache = {}

        handlers = [
            self[x]
            for x in self.root().children
            if self[x].payload in self.parameterize_handlers
        ]
        nodes = list(self.node_iter_breadth_first(handlers))

        QCA = self._po.source.source

        labeler = self.labels()  # self._labeler

        # labels = [entry['data'][ph].values() for entry in labeler.db.values() for ph in handlers]

        # this only has the labels that we plan to modify, since we optionally
        # ignore entire handlers and/or parameters
        labels = [
            ph_lbls
            for ph in handlers
            for ph_lbls in labeler.db["ROOT"]["data"][ph.payload]
        ]

        # n_bits = {}
        # fundamental = {}

        if verbose:
            print(f"\n\nDetermining next join...")
            print("\nCandidates are")
            print([x.payload for x in nodes])
            print("Have label data for", labels)
            print("Param data labels is", list(param_data))

        # total_hits = 0
        # GRAPH = True

        for node in tqdm.tqdm(nodes, total=len(nodes), desc="joining", ncols=80):
            lbl = node.payload

            sys.stdout.flush()
            if len(node.children) == 0:
                # easier to just take a parent and iterate children
                continue

            group = self.db[node.payload]["data"]["group"]

            for child in node.children:

                if any(x[0] == lbl and x[2] == "delete" for x in ignore_bits):
                    # print("DEBUG 2: SKIPPING")
                    continue

                print("\n########################################################\n")
                print("Considering for joining child", self[child], "to parent", node)

                child_lbl = self[child].payload

                if any(child_lbl == lbl and x[2] == "delete" for x in ignore_bits):
                    print(f"Ignoring {child_lbl}")
                    continue

                self._join_param(
                    param_data,
                    node,
                    self[child],
                    key=key,
                    mode=mode,
                    eps=eps,
                    bit_gradients=bit_gradients,
                    ignore_bits=ignore_bits,
                )

        n_candidates = len(bit_gradients)

        if n_candidates:

            # I think we can just pop the last since they are always sorted?
            # The check with all in ignore is expensive

            # bit_gradients.remove(bit_gradients[-1])
            # n_candidates -= 1

            print(
                "\n\n### Here are the {} candidates, ordered by priority (mode: {}) ###".format(
                    n_candidates, mode[0]
                )
            )
            for j, bit_gradient in enumerate(reversed(bit_gradients), 1):
                print(
                    "{:5d}/{:5d} bits={}".format(
                        j, n_candidates, bit_gradient[1].bits(maxbits=True)
                    ),
                    bit_gradient[6],
                    bit_gradient[0],
                    bit_gradient[1].to_smarts(
                        atom_universe=self.atom_universe,
                        bond_universe=self.bond_universe,
                    ),
                    bit_gradient[5],
                )

            if self.score_candidate_limit is not None:
                print(
                    "Limiting candidates to the top {}".format(
                        self.score_candidate_limit
                    )
                )
                for n, val in enumerate(bit_gradients):
                    if n >= self.score_candidate_limit:
                        bit_gradients.remove(val)
                # bit_gradients = bit_gradients[: self.score_candidate_limit]

        # print("Ignore is")
        # for i in ignore_bits.items():
        #     print(i)

        for bit_gradient in reversed(bit_gradients):
            # split_bit = bit_gradients[0][1]
            lbl = bit_gradient[0]
            split_bit = bit_gradient[1]
            split_combination = bit_gradient
            if all([not (x[0] == lbl and x[2] == operation) for x in ignore_bits]):
                parent = bit_gradient[3]
                child = bit_gradient[4]

                print("Ignoring ", lbl, split_bit.compact_str())
                ignore_bits.setdefault((lbl, split_bit, operation), list()).append(
                    (bit_gradient[2], bit_gradient[3])
                )

                print(
                    "\n=====\nCombining",
                    parent,
                    "with",
                    child,
                    "\n\n",
                )
                print(
                    "\n\nParameter SMARTS:",
                    self.db[parent.payload]["data"]["group"].to_smarts(),
                    "\n\nRemoved SMARTS:",
                    self.db[child.payload]["data"]["group"].to_smarts(),
                    "\n\n",
                )

                _ = self.combine_parameter(child)
                # node.children = children
                node = child
                bit_gradients.remove(bit_gradients[-1])

                return node, bit_gradient[2]

        return None, None

    def _find_next_split(
        self,
        param_data,
        key=None,
        ignore_bits=None,
        mode=None,
        eps=1.0,
        bit_gradients=None,
        bit_cache=None,
    ):

        verbose = self.verbose

        operation = "new"

        if mode is None:
            mode = ["mean_difference"]

        if bit_gradients is None:
            bit_gradients = bit_gradient_factory()
        if ignore_bits is None:
            ignore_bits = {}
        if bit_cache is None:
            bit_cache = {}

        handlers = [
            self[x]
            for x in self.root().children
            if self[x].payload in self.parameterize_handlers
        ]
        nodes = list(self.node_iter_breadth_first(handlers))

        QCA = self._po.source.source

        labeler = self.labels()  # self._labeler

        # labels = [entry['data'][ph].values() for entry in labeler.db.values() for ph in handlers]

        # this only has the labels that we plan to modify, since we optionally
        # ignore entire handlers and/or parameters
        labels = [
            ph_lbls
            for ph in handlers
            for ph_lbls in labeler.db["ROOT"]["data"][ph.payload]
        ]

        n_bits = {}
        fundamental = {}

        if verbose:
            print("\n\nDetermining next split...")
            print("\nCandidates are")
            print([x.payload for x in nodes])
            print("Have label data for", labels)
            print("Param data labels is", list(param_data))

        total_hits = 0
        GRAPH = True

        processes = os.cpu_count()
        pool = concurrent.futures.ProcessPoolExecutor(processes)

        for node in tqdm.tqdm(nodes, total=len(nodes), desc="bit scanning", ncols=80):
            lbl = node.payload

            if lbl in self._ignore_parameters:
                # print("Skipping case 6 (filtered)")
                continue

            sys.stdout.flush()

            print("\n########################################################\n")
            print("Considering for bit scans", node)

            if lbl not in labels:
                print("Skipping case 1 (not applied)")
                continue
            # if lbl[0] != 't':
            #     continue

            if lbl not in param_data:
                print("Skipping case 2 (no data: this is possibly a bug)")
                breakpoint()
                continue

            if lbl in bit_cache:
                print("Skipping case 3: cached")
                continue

            n_vals = len(param_data[lbl])
            if key is not None:
                n_data = sum([len(x[key]) for x in param_data[lbl].values()])
            else:
                n_data = sum([len(x) for x in param_data[lbl].values()])
            if n_vals == 0 or n_data == 0:
                print(
                    f"Skipping case 5 (no data assigned; dead parameter for key {key:s})"
                )
                bit_cache[lbl] = None
                continue

            # print("\nConsidering for bit scans", node)

            ff_param = self.db[lbl]["data"]["group"]
            # print("FF param:", ff_param)

            if not GRAPH:
                # try to reorder the prim to the ff prim. note this modifies in-place
                # print("Aligning to FF")
                for prim in param_data[lbl]:
                    # print("before:", prim)
                    prim.align_to(ff_param)
                    # print("after :", prim)

                # Now that all prims are ordered to match the FF prim, try to
                # add them together as tighly as possible. For example, if the param
                # is *-C-*, and we have H-C-C and C-C-H, we just order to the first
                # one visited such that the group will always be H-C-C in this case.
                # This allows preventing cases where it says C-C-C and H-C-H is in
                # the group, when no match prims were either.
                prims = list(param_data[lbl])
                group = prims[0]

                # print("Aligning to group")
                for prim in param_data[lbl]:
                    # print("before:", prim)
                    prim.align_to(group)
                    # print("after :", prim)
                    group += prim

                # group = functools.reduce(lambda x, y: x + y, param_data[lbl])

                # this I already know isn't the best since the primitives are
                # in any order, and symmetry messes this up
                fundamental[lbl] = (
                    self.db[lbl]["data"]["parameter"].smirks,
                    self.db[lbl]["data"]["group"].to_smarts(),
                    group.to_smarts(),
                    (self.db[lbl]["data"]["group"] & group).to_smarts(),
                )
                # print("\n\nFundamental SMARTS (idx, lbl, FF, FF group, DATA group, FF & DATA):")
                # print("{:3d} {:5s} {}".format(0, lbl, fundamental[lbl]))

                # only iterate on bits that matched from the smirks from the data
                # we are considering

                # The data should be a subset of the parameter that covers it

                # this does not cover degeneracies; use the in keyword

                # if (group - self.db[lbl]["data"]["group"]).reduce() != 0:
            else:
                prims = list(param_data[lbl])
                group = prims[0].union_list(prims[1:], executor=pool)

            if len(group.nodes) < len(ff_param.nodes):
                group.append(ff_param)

            # try:
            #     # Checking the sum does not work for torsions, so check each individually
            #     # if group not in self.db[lbl]["data"]["group"]:
            #     if any(
            #         [x not in self.db[lbl]["data"]["group"] for x in param_data[lbl]]
            #     ):
            #         print("ERROR: data is not covered by param!")
            #         print("Group is", group.to_smarts())
            #         print("FF Group is", self.db[lbl]["data"]["group"].to_smarts())
            #         print(
            #             "marginal is ",
            #             (group.subtract(self.db[lbl]["data"]["group"])).compact_str(),
            #         )
            #         # breakpoint()
            # except Exception as e:
            #     print(e)
            #     # breakpoint()

            # Now that we know all params are covered by this FF param,
            # sanitize the group by making it match the FF group, so things
            # stay tidy
            # group = group & self.db[lbl]["data"]["group"]

            # assert (group - self.db[lbl]['data']['group']).reduce() == 0

            # this indicates what this smirks covers, but we don't represent in the
            # current data
            # try:
            #     uncovered = self.db[lbl]["data"]["group"].subtract(group)
            #     if uncovered.bits() == 0:
            #         print("This dataset completely covers", lbl, ". Nice!")
            #     else:
            #         print("This dataset does not cover this information:")
            #         print(uncovered.compact_str())
            # except Exception as e:
            #     print("error at 1568", e)
            #     pass
            #     # breakpoint()

            # iterate bits that we cover (AND them just to be careful)
            # group = group & self.db[lbl]["data"]["group"]
            if verbose:
                print("\nContinuing with param ", lbl, "this information:")
                for data in param_data[lbl]:
                    print(data.to_smarts())
            print(
                "\nThis parameter has {:d} unique graphs".format(len(param_data[lbl]))
            )

            print(
                "\nSum: atoms= {} bits= {}\n".format(
                    len(group.nodes), group.bits(maxbits=True)
                ),
                group.to_smarts(),
            )

            u_group = group.copy()
            for atom in u_group.nodes:
                u_group.nodes[atom]["primitive"] &= self.atom_universe
            for bond in u_group.edges:
                u_group.edges[bond]["primitive"] &= self.bond_universe

            print("\nUniverse-intersected sum:\n\n", u_group.to_smarts(), "\n\n")

            print(
                "\nUniverse-reference sum:\n\n",
                group.to_smarts(
                    atom_universe=self.atom_universe, bond_universe=self.bond_universe
                ),
                "\n\n",
            )

            print("\nFF param for ", lbl, "is:")
            print(ff_param.to_smarts())
            print("--------------------------------------------------------")

            # Try to find the magical split from the raw param_data
            # breakpoint()
            # for prim_data in param_data[lbl]:
            #     x = 5

            #     bit_gradients.append(x)
            # This means I want to get all of the data for a parameter term
            # this is in param data, which is a {prim: data} structure

            # The original tier 1 score; go through every bit and try to find the
            # best split
            # Need to support a way that inverts this: look at the data, then try
            # to segment it
            # This means trying to take the raw data, then try to cluster it

            n_bits[lbl] = group.bits(maxbits=True)
            bit_visited = {}

            # manipulations = set([bit for bit in param_group if bit in group])
            # manipulations = set(group)
            # try:
            branched_group = prims[0].union_list(
                prims[1:], add_nodes=True, executor=pool
            )
            # except Exception as e:
            #     breakpoint()
            #     print("OUCH")

            if verbose:
                print(
                    "Total unabided union: atoms= {} bits= {}".format(
                        len(branched_group.nodes), branched_group.bits(maxbits=True)
                    )
                )
                print(branched_group.to_smarts())
                print()

            # print("Group bits:")
            # for b in group:
            # #     print(b.compact_str())
            # Mff = ff_param.map_to(group)
            # Mff = {n:n for n in ff_param.nodes}
            bit_group = ff_param.intersection(
                group, add_nodes=False, fill_new_nodes=False
            )
            mapping = ff_param.map_to(bit_group)

            manipulations = [
                (bit, mapping) for bit in bit_group.iter_bits(skip_ones=True)
            ]
            M = branched_group.map_to(ff_param)
            visited = []

            # print("param has", len(ff_param.nodes), "nodes")
            new_branches = []

            # go over the large, branched union, and look at the neighbors
            sys.stdout.flush()
            if len(branched_group.nodes) != len(ff_param.nodes):
                for m in M:
                    for nbr in branched_group.adj[m]:

                        # we want to find a node that is not in the group, but connected
                        # to it. M[nbr] == None is when the node is not in the group
                        # so we want nbr to be outside, and m to be inside
                        if M[nbr] is None and M[m] is not None and nbr not in visited:

                            # now we know we have a new branch with a single outside
                            # node
                            new_branch = branched_group.copy()
                            for n in branched_group.nodes:

                                # nbr is outside, so remove all others
                                if M[n] is None and n != nbr:
                                    new_branch.remove_node(n)
                            # print("new_branch has", len(new_branch.nodes), "nodes")
                            if len(new_branch.nodes) == len(ff_param.nodes):
                                continue
                            # here should have a graph that matches the group
                            # but has one node extending out

                            # this wipes out everything that is accounted for
                            # so we only keep the "new" bits in the branched node
                            # To be more clear, we want to clear out anything that
                            # has a mapping, and save only unmapped nodes. Then,
                            # we will iterate the bits in the unmapped nodes next
                            Minv = {v: k for k, v in M.items() if v is not None}
                            for m2, v2 in M.items():
                                if v2 is not None:
                                    new_branch.nodes[m2]["primitive"].clear()
                                    for nbr2 in [
                                        x for x in ff_param.adj[v2] if x in Minv
                                    ]:
                                        new_branch.edges[m2, Minv[nbr2]][
                                            "primitive"
                                        ].clear()
                            # new_branch.clear_primitives()
                            if new_branch not in new_branches:

                                new = True

                                for i, existing in enumerate(new_branches):

                                    existing_map = existing.map_to(new_branch)

                                    # a quick way to estimate a 1to1 mapping
                                    mapped_atoms = [
                                        x is not None for x in existing_map.values()
                                    ]

                                    if all(mapped_atoms) and len(
                                        new_branch.nodes
                                    ) == sum(mapped_atoms):
                                        old = new_branches[i]
                                        new_branches[i] = existing.union(
                                            new_branch, add_nodes=False
                                        )
                                        new = False
                                        if verbose and not new_branches[i].equal(old):
                                            print(
                                                "    Updated new branch",
                                                i + 1,
                                                "to",
                                                new_branches[i].compact_str()
                                                # "using",
                                                # new_branch.compact_str(),
                                            )
                                        break

                                if new:
                                    new_branches.append(new_branch)
                                    if verbose:
                                        l = len(new_branches)
                                        print(
                                            f"New branch point {l:d} adding bits from node",
                                            new_branch.compact_str(),
                                        )

                                    # print("   tags:", new_branch.graph)
                                    # print("   nodes:", new_branch.nodes)
                                    # print("   edges:", new_branch.edges)
                            visited.append(nbr)
            if len(new_branches) > 0:

                # we set skip_ones to false because I don't know how to combine
                # symmetric cases yet, and two overlapping nodes have different
                # bits
                branched_bits = []
                for branch in new_branches:
                    mapping = ff_param.map_to(branch)
                    branched_bits.extend(
                        iter((bit, mapping) for bit in branch.iter_bits(skip_ones=True))
                    )
                # branched_bits = [
                #     b
                #     for branch in new_branches
                #     for b in branch.iter_bits(skip_ones=True)
                # ]

                l = len(new_branches)
                print(f"\nTotal branches added: {l:d}")

                l = len(branched_bits)
                print(f"\nTotal branched bits added: {l:d}")
                manipulations.extend(branched_bits)

                print("\n")

                # only want to union those that line up

                # manipulations.extend(
                #     [
                #         b
                #         for b in new_branches[0]
                #         .union_list(new_branches, add_nodes=True)
                #         .iter_bits(skip_ones=True)
                #     ]
                # )

            M = group.map_to(ff_param)
            todo = len(manipulations) + 1
            completed = 0
            hits = 0

            single_bit_manips = manipulations.copy()
            self._prim_clusters.clear()
            maxbits = 1
            total = len(manipulations)

            debug = True
            # if self.bit_search_limit is None:
            #     estimated_grand_total = total ** total
            # else:
            #     estimated_grand_total = total ** self.bit_search_limit
            # print(f"\nEstimated max bits to scan: {estimated_grand_total:d}")

            pbar = tqdm.tqdm(
                ncols=80, total=total, desc="manipulation search", disable=verbose
            )

            chunksize = processes
            # chunksize = 1

            operation = "new"
            good_manips = {}
            idx = 0
            while idx < len(manipulations):
                sys.stdout.flush()

                # bit = manipulations.pop()

                if chunksize > 1:
                    bit_visited_copy = bit_visited.copy()
                    work_list = iter(
                        pool.submit(
                            ff_param_subtract,
                            ii,
                            ff_param,
                            lbl,
                            bit,
                            group,
                            mapping,
                            bit_visited_copy,
                            ignore_bits,
                            operation,
                            list(param_data[lbl]),
                        )
                        for ii, (bit, mapping) in enumerate(
                            manipulations[idx : idx + chunksize], idx
                        )
                    )
                else:
                    work_list = [
                        ff_param_subtract(
                            idx,
                            ff_param,
                            lbl,
                            manipulations[idx][0],
                            group,
                            manipulations[idx][1],
                            bit_visited,
                            ignore_bits,
                            operation,
                            list(param_data[lbl]),
                        )
                    ]

                idx += min(len(manipulations) - idx, chunksize)

                iterable = (
                    concurrent.futures.as_completed(work_list)
                    if chunksize > 1
                    else work_list
                )

                for j, work in enumerate(iterable, idx):
                    sys.stdout.flush()

                    if chunksize > 1:
                        (
                            i,
                            ff_subtract,
                            bit,
                            visited,
                            ignore,
                            valid,
                            blank,
                            occluding,
                        ) = work.result()
                    else:
                        (
                            i,
                            ff_subtract,
                            bit,
                            visited,
                            ignore,
                            valid,
                            blank,
                            occluding,
                        ) = work

                    todo -= 1

                    completed += 1
                    pbar.update(1)
                    if debug:
                        print(
                            "{:8d}/{:8d}".format(todo, completed),
                            "Scanning (atoms: {:d} bits {:d})".format(
                                len(bit.nodes), bit.bits()
                            ),
                            "{:36s}".format(bit.compact_str()),
                            end=" ",
                        )

                    if visited:
                        if debug:
                            print("Reject: already visited")
                        continue

                    bit_visited[bit] = None

                    if ignore:
                        if debug:
                            print("Reject: Ignoring list")
                        continue

                    if not valid:  # caseB:
                        # this case is technically allowed for odd reasons
                        if debug:

                            print(
                                "Reject: Invalid SMARTS",
                            )
                            # print(ff_subtract.compact_str())
                        continue

                    if blank:
                        if debug:
                            print(
                                "Reject: This bit would not separate the data",
                            )
                        continue

                    if occluding:  # occluding_split:

                        if (
                            self.bit_search_limit is None
                            or bit.bits() <= self.bit_search_limit
                        ):
                            mapping = ff_param.map_to(
                                bit.union(single_bit_manips[0][0], add_nodes=False)
                            )
                            if chunksize > 1:
                                new_manips = []
                                for new_bit, old_mapping in single_bit_manips:
                                    work = (
                                        pool.submit(
                                            new_bit.union,
                                            bit,
                                            add_nodes=False,
                                            map=None,
                                        ),
                                        old_mapping,
                                    )
                                    new_manips.append(work)
                                new_manips = [
                                    (ret.result(), mapping)
                                    for ret, mapping in new_manips
                                ]
                            else:

                                new_manips = list(
                                    [
                                        (
                                            new_bit.union(
                                                bit, add_nodes=False, map=None
                                            ),
                                            old_mapping,
                                        )
                                        for new_bit, old_mapping in single_bit_manips
                                    ]
                                )

                            new_manips = list(
                                filter(lambda x: x[0].bits() > bit.bits(), new_manips)
                            )
                            n_new_manips = 0

                            if chunksize > 1:
                                for m in new_manips:
                                    work_list = concurrent.futures.as_completed(
                                        pool.submit(x[0].__eq__, m[0])
                                        for x in manipulations
                                    )
                                    exists = map(lambda x: x.result(), work_list)
                                    if not any(exists):
                                        manipulations.append(m)
                                        n_new_manips += 1

                                    # since `any` can short-circuit, cancel any remaining items in the generator
                                    for work in work_list:
                                        work.cancel()

                            else:
                                for m in new_manips:
                                    if m not in manipulations:
                                        manipulations.append(m)
                                        n_new_manips += 1

                            # n_new_manips = len(manipulations) - n_new_manips
                            todo += n_new_manips
                            total += n_new_manips
                            pbar.total = total
                            print(
                                "Adding another bit (",
                                bit.bits() + 1,
                                "), producing {:d} more splits to check (total= {:d})".format(
                                    n_new_manips, total
                                ),
                            )
                        elif debug:
                            print("Reject: Occluding but max bits reached")
                        continue

                    # TODO: Since we are doing parallel comparisons, some
                    # ignore and/or cached bits can pass through twice ore more
                    # Need to go over each work_list and make sure there arent
                    # any intra-overlaps
                    good_manips.setdefault(j, (bit, ff_subtract))

                    # for ignore in ignore_bits:
                    #     if type(ignore) == type(bit) and :
                    #         if verbose:
                    #             print("Ignoring since it is in the ignore list. Matches this ignore:")
                    #             print(ignore)
                    #         continue
                    hits += 1
                    maxbits = max(maxbits, bit.bits())

                    print("Accept: {} hits".format(hits))

                # manipulations.extend(new_manip_chunks)

            if verbose or debug:
                print("\n\n")
                sys.stdout.flush()

            if 1:
                work_list = [
                    pool.submit(
                        bit_split_and_score,
                        i,
                        param_data[lbl],
                        group,
                        key,
                        lbl,
                        mode,
                        eps,
                        self.denoms,
                        self.stats_fn,
                        self.angle_mode,
                        self.angle_cosine_frequency_residual_max,
                        not verbose,
                    )
                    for i, (bit, group) in good_manips.items()
                ]

                for idx, unit in enumerate(
                    concurrent.futures.as_completed(work_list), 1
                ):
                    i, mask, vals, success = unit.result()
                    bit, group = good_manips[i]
                    if verbose:
                        print(
                            "{:8d}/{:8d}".format(idx, len(good_manips)),
                            "Scanning (atoms: {:d} bits {:d})".format(
                                len(bit.nodes), bit.bits()
                            ),
                            "{:40s}".format(bit.compact_str()),
                            end=" ",
                        )
                        val_str = "{:>10s}".format("none")
                        val = vals.get(mode[0])
                        if val:
                            val_str = "{:10.5f}".format(val)
                        print(
                            "{:20s}Score: {} eps: {} split? {}".format(
                                "", val_str, eps, success
                            ),
                            end=" ",
                        )
                    if success:
                        smarts = group.to_smarts(
                            atom_universe=self.atom_universe,
                            bond_universe=self.bond_universe,
                        )
                        added = self.bit_add_if_new(
                            bit,
                            group,
                            smarts,
                            lbl,
                            vals,
                            mode,
                            mask,
                            operation,
                            bit_gradients,
                            ignore_bits,
                        )
                        if verbose and added:
                            print("*** ADDED {}".format(smarts), end="")
                    if verbose:
                        print()
                        sys.stdout.flush()
            else:

                for i, (bit, ff_subtract) in enumerate(good_manips.items(), 1):
                    if verbose:
                        print(
                            "{:8d}/{:8d}".format(i, len(good_manips)),
                            "Scanning (atoms: {:d} bits {:d})".format(
                                len(bit.nodes), bit.bits()
                            ),
                            bit.compact_str(),
                        )

                    self._scan_param_with_bit(
                        param_data,
                        lbl,
                        ff_subtract,
                        bit,
                        key=key,
                        mode=mode,
                        eps=eps,
                        bit_gradients=bit_gradients,
                        ignore_bits=ignore_bits,
                        quiet=not verbose,
                    )

            bit_cache[lbl] = True

            print(
                "Evaluated {:d} bit splits, producing {:d} hits of max bit depth of {:d}".format(
                    completed, hits, maxbits
                )
            )
            total_hits += hits
            pbar.close()

        if len(bit_gradients) == 0:
            print("No values! Returning nothing")
            pool.shutdown()
            return None, None

        if total_hits > 0:
            print("Total hits to evaluate: {}".format(total_hits))

        if len(n_bits):
            print("\n\nBits per parameter:")
            for i, (b, v) in enumerate(n_bits.items(), 1):
                print("| {:5} {:4d}".format(b, v), end="")
                if i % 7 == 0:
                    print(" |\n", end="")
            print("\nTotal parameters:", len(n_bits))

        if len(fundamental):
            print(
                "\n\nFundamental SMARTS (idx, lbl, FF, FF group, DATA group, FF & DATA):"
            )
            for i, (lbl, v) in enumerate(fundamental.items(), 1):
                print("{:3d} {:5s} {}".format(i, lbl, v))

        # bit_gradients = [bit_gradient for ignore in ignore_bits for bit_gradient in bit_gradients if not (bit_gradient[0] == ignore[0] and bit_gradient[1] == ignore[1])]

        bg = []
        # print("ignore is")
        # for ib in ignore_bits:
        #     print(ib[0], ib[1].compact_str())
        # n_bg = len(bit_gradients)
        # for bit_gradient in bit_gradients:
        #     keep = True
        #     # print("bg:", bit_gradient)
        #     for ignore in ignore_bits:

        #         # map={
        #         #     m: n
        #         #     for m, n in zip(
        #         #         bit_gradient[1].nodes, ignore[1].nodes
        #         #     )
        #         # },
        #         same_prim = bit_gradient[1].equal(ignore[1])
        #         # print("Same?", same_prim)
        #         # print("    new_bit", bit_gradient[1].compact_str())
        #         # print("    ignored", ignore[1].compact_str())
        #         if bit_gradient[0] == ignore[0] and same_prim:
        #             bit_gradients.remove(bit_gradient)
        #             break
        # if keep:
        #     bg.append(bit_gradient)

        # print("Bit gradients changed from", n_bg, "to", len(bit_gradients))
        # bit_gradients = bg

        n_candidates = len(bit_gradients)
        if n_candidates:
            # I think we can just pop the last since they are always sorted?
            # The check with all in ignore is expensive
            # bit_gradients.remove(bit_gradients[-1])
            # n_candidates -= 1

            print(
                "\n\n### Here are the {} candidates to split, ordered by priority (mode: {}) ###".format(
                    n_candidates, mode[0]
                )
            )
            for j, bit_gradient in enumerate(reversed(bit_gradients), 1):
                print(
                    "{:5d}/{:5d} bits={}".format(
                        j, n_candidates, bit_gradient[1].bits(maxbits=True)
                    ),
                    bit_gradient[6],
                    bit_gradient[0],
                    "{:12.5f}".format(bit_gradient[5]),
                    bit_gradient[1].to_smarts(
                        atom_universe=self.atom_universe,
                        bond_universe=self.bond_universe,
                    ),
                )

            if self.score_candidate_limit is not None:
                print(
                    "Limiting candidates to the top {}".format(
                        self.score_candidate_limit
                    )
                )
                for n, val in enumerate(bit_gradients):
                    if n >= self.score_candidate_limit:
                        bit_gradients.remove(val)
                # bit_gradients = bit_gradients[: self.score_candidate_limit]

        # QCA = self._po.source.source
        # self.to_smirnoff_xml("tmp.offxml", verbose=False)
        # coverage_pre = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA).assign_labels_from_openff("tmp.offxml", "tmp.offxml")
        # need this for measuring geometry
        # should only need to do it once

        # if verbose:
        #     print("\n\nHere is ignore:")
        #     for ignore in ignore_bits:
        #         print(ignore[0], ignore[1].compact_str())
        #     print("\n")

        for bit_gradient in reversed(bit_gradients):
            # split_bit = bit_gradients[0][1]
            lbl = bit_gradient[0]
            split_bit = bit_gradient[1]
            split_combination = bit_gradient
            smarts = bit_gradient[4]
            if all(
                [
                    not (
                        (x[0] == lbl and x[1] == split_bit and x[2] == operation)
                        and (
                            x[0] == lbl
                            and any([split_combination == y[1] for y in ignore_bits[x]])
                        )
                    )
                    for x in ignore_bits
                ]
            ):

                # child = group - split_bit
                # smarts = split_bit.to_smarts(tag=True, atom_universe=self.atom_universe, bond_universe=self.bond_universe)
                node = self.split_parameter(lbl, split_bit, smarts=smarts)

                if node is None:
                    continue

                #

                # if lbl[0] in "ti":
                #     self._guess_periodicities(
                #         lbl, [x for x in param_data[lbl]['measure']], report_only=False
                #     )

                # self.to_smirnoff_xml("tmp.offxml", verbose=False)
                # coverage_post = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA).assign_labels_from_openff("tmp.offxml", "tmp.offxml")
                # overlapped = self._check_overlapped_parameter(coverage_pre, coverage_post, node)
                print("Ignoring ", lbl, split_bit.compact_str())
                if (lbl, split_bit, operation) in ignore_bits:
                    ignore_bits[(lbl, split_bit, operation)].append(
                        (bit_gradient[2], bit_gradient[3])
                    )
                else:
                    ignore_bits[(lbl, split_bit, operation)] = [
                        (bit_gradient[2], bit_gradient[3])
                    ]
                print("\n=====\nSplitting", lbl, "to", node.payload, "\n\n")
                # self.db[lbl]["data"]["group"],
                # "using\n",
                # split_bit,
                # "\nvals (key=",
                # key,
                # ") eps=",
                # eps,
                # bit_gradient[2],
                # "\nresult is\n",
                # self.db[lbl]["data"]["group"] - split_bit,
                # )
                print(
                    "\n\nParent SMARTS:",
                    self.db[lbl]["data"]["group"].to_smarts(),
                    "\nNew SMARTS   :",
                    split_bit.to_smarts(
                        atom_universe=self.atom_universe,
                        bond_universe=self.bond_universe,
                    ),
                    "\n\n",
                )
                bit_gradients.remove(bit_gradients[-1])

                # if overlapped:

                #     print("This param occluded its parent; narrowing the parent and continuing")
                #     coverage_pre = coverage_post
                #     continue

                # print("The parent is")
                # print(group.drop(child))
                # print("Smarts is")
                # print(group.drop(child).to_smarts())
                # print("The child is")
                # print(child)
                # print("Smarts is")
                # print(child.to_smarts())

                pool.shutdown()
                return node, bit_gradient[2]
            else:
                print("This parameter is in the ignore list:", bit_gradient)

        pool.shutdown()

        return None, None

    def _cache_ic_prim(self):

        if hasattr(self, "_ic_prim"):
            return
        self._ic_prim = {}

        import geometric.internal
        import geometric.molecule
        import offsb.ui.qcasb

        QCA = self._po.source.source

        prim_table = {
            "Bonds": geometric.internal.Distance,
            "Angles": geometric.internal.Angle,
            "ImproperTorsions": geometric.internal.OutOfPlane,
            "ProperTorsions": geometric.internal.Dihedral,
        }

        labeler = self.labels()

        n_entries = len(list(QCA.iter_entry()))
        for entry in tqdm.tqdm(
            QCA.iter_entry(), total=n_entries, desc="IC generation", ncols=80
        ):

            self._prim[entry.payload] = {}

            # need to unmap... sigh
            smi = QCA.db[entry.payload]["data"].attributes[
                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
            ]
            rdmol = offsb.rdutil.mol.build_from_smiles(smi)
            atom_map = offsb.rdutil.mol.atom_map(rdmol)
            map_inv = offsb.rdutil.mol.atom_map_invert(atom_map)

            primitives = self._to.db[entry.payload]["data"]

            graph_ic = None
            graph = self._to.db[entry.payload]["data"].get("graph", None)

            for mol in QCA.node_iter_depth_first(entry, select="Molecule"):
                if graph is not None and graph_ic is None:
                    graph_ic = {
                        "b": graph.bonds(),
                        "a": graph.angles(),
                        "i": graph.outofplanes(),
                        "t": graph.torsions(),
                        "n": graph.atom,
                    }
                    graph_ic = {k: v for x in graph_ic.values() for k, v in x.items()}

                # with tempfile.NamedTemporaryFile(mode="wt") as f:
                #     offsb.qcarchive.qcmol_to_xyz(
                #         QCA.db[mol.payload]["data"], fnm=f.name
                #     )
                #     gmol = geometric.molecule.Molecule(f.name, ftype="xyz")
                # # with open("out.xyz", mode="wt") as f:
                # #     offsb.qcarchive.qcmol_to_xyz(QCA.db[mol.payload]["data"], fd=f)

                # ic_prims = geometric.internal.PrimitiveInternalCoordinates(
                #     gmol,
                #     build=True,
                #     connect=True,
                #     addcart=False,
                #     constraints=None,
                #     cvals=None,
                # )
                ic_prims = ic_op.db[entry.payload]["data"]

                for ic_type, prim_fn in prim_table.items():
                    for unmapped, param_name in labeler.db[entry.payload]["data"][
                        ic_type
                    ].items():
                        # forward map to QCA index, which is what ICs use

                        aidx = [atom_map[i] - 1 for i in unmapped]

                        if prim_fn is geometric.internal.OutOfPlane:
                            aidx = ImproperDict.key_transform(aidx)

                            # this is needed for the code below, where we use
                            # unmapped as the key, which must be sorted (in the
                            # same way openff sorts
                            unmapped = ImproperDict.key_transform(unmapped)
                            param_name = "i"
                        else:
                            unmapped = ValenceDict.key_transform(unmapped)
                            aidx = ValenceDict.key_transform(aidx)

                        new_ic = prim_fn(*aidx)
                        if new_ic not in ic_prims.Internals:
                            # Is it ok if we add extra ICs to the primitive list?
                            # if prim_fn is not geometric.internal.OutOfPlane:
                            #     print("Adding an IC:", new_ic, "which may indicate missing coverage by the FF!")
                            ic_prims.add(new_ic)

                        # if the tuple has no param (impropers), skip it
                        try:
                            if self._to.chembit:
                                prim = primitives[unmapped]
                            elif graph_ic is not None:
                                prim = graph_ic[unmapped]
                            else:
                                prim = prim_to_graph[param_name[0]].from_string_list(
                                    primitives[unmapped], sorted=True
                                )

                            # prim_key = self._prim_tab.get(prim)
                            # prim_key = self._get_prim_key(prim)
                            prim_map = self._prim.get(entry.payload)
                            if prim_map is None:
                                self._prim[entry.payload] = {unmapped: prim}
                            else:
                                prim_map[unmapped] = prim

                        except Exception as e:
                            breakpoint()
                            print("Issue with assigning primitive! Error message:")
                            print(e)

                def _get_prim_key(self, prim):
                    self._prim_tab

                # to estimate the FCs, we need the gradients and the hessians
                if QCA[mol.parent].payload == "Gradient":
                    for hessian_node in QCA.node_iter_depth_first(
                        mol, select="Hessian"
                    ):

                        xyz = QCA.db[mol.payload]["data"].geometry
                        hess = QCA.db[hessian_node.payload]["data"].return_result
                        grad = np.array(
                            QCA.db[hessian_node.payload]["data"].extras["qcvars"][
                                "CURRENT GRADIENT"
                            ]
                        )

                        ic_hess = ic_prims.calcHess(xyz, grad, hess)

                        # eigs = np.linalg.eigvalsh(ic_hess)
                        # s = np.argsort(np.diag(ic_hess))
                        # force_vals = eigs
                        force_vals = np.diag(ic_hess)
                        # ic_vals = [ic_prims.Internals[i] for i in s]
                        ic_vals = ic_prims.Internals
                        for aidx, val in zip(ic_vals, force_vals):
                            key = tuple(
                                map(
                                    lambda x: map_inv[int(x) - 1],
                                    str(aidx).split()[1].split("-"),
                                )
                            )

                            if type(aidx) is geometric.internal.OutOfPlane:
                                key = ImproperDict.key_transform(key)
                            else:
                                key = ValenceDict.key_transform(key)

                            # no conversion, this is done elsewhere, e.g. set_parameter
                            # if type(aidx) is geometric.internal.Distance:
                            #     val = val / (offsb.tools.const.bohr2angstrom ** 2)
                            # val *= offsb.tools.const.hartree2kcalmol

                            # if mol.payload == "QCM-1396980":
                            #     breakpoint()
                            #     print("HI")
                            if mol.payload not in self._fc:
                                self._fc[mol.payload] = {key: val}
                            else:
                                self._fc[mol.payload][key] = val

    def _calculate_ic_force_constants(self):

        """
        TODO: Make an operation out of this rather than do it here
        """

        if self._ic is not None:
            return

        self._ic = {}
        self._fc = {}
        self._prim = {}

        import geometric.internal
        import geometric.molecule
        import offsb.ui.qcasb

        QCA = self._po.source.source
        # need this for measuring geometry
        # should only need to do it once
        qcasb = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA)

        vtable = {
            "Bonds": qcasb.measure_bonds,
            "Angles": qcasb.measure_angles,
            "ImproperTorsions": qcasb.measure_outofplanes,
            "ProperTorsions": qcasb.measure_dihedrals,
        }
        prim_table = {
            "Bonds": geometric.internal.Distance,
            "Angles": geometric.internal.Angle,
            "ImproperTorsions": geometric.internal.OutOfPlane,
            "ProperTorsions": geometric.internal.Dihedral,
        }

        ignore_tab = {
            "b": "Bonds",
            "a": "Angles",
            "i": "ImproperTorsions",
            "t": "ProperTorsions",
        }

        ignore = [ignore_tab[x] for x in self._ignore_parameters if x in ignore_tab]

        # should return a dict of the kind
        #
        # well, the measure already collects what is needed
        # need a way to transform indices to primitives st we have prim: measure
        # so we could just keep a indices -> prim, which is what _to is

        ic_op = offsb.op.internal_coordinates.InteralCoordinateGeometricOperation(
            QCA, "ic", verbose=True
        )
        ic_op.processes = None
        ic_op.apply()

        # need this for measuring geometry
        # should only need to do it once
        qcasb = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA)
        qcasb.verbose = True

        labeler = self.labels()
        # self.to_smirnoff_xml("tmp.offxml", verbose=False)
        # labeler = qcasb.assign_labels_from_openff("tmp.offxml", "tmp.offxml")
        # self._labeler = labeler

        self._ic = qcasb.measure_internal_coordinates(ignore=ignore)
        # for ic_type, measure_function in vtable.items():
        #     # the geometry measurements
        #     ic = measure_function(ic_type)
        #     ic.source.source = QCA
        #     ic.verbose = True
        #     ic.apply()
        #     self._ic[ic_type] = ic

        n_entries = len(list(QCA.iter_entry()))
        for entry in tqdm.tqdm(
            QCA.iter_entry(), total=n_entries, desc="IC generation", ncols=80
        ):

            self._prim[entry.payload] = {}

            # need to unmap... sigh
            smi = QCA.db[entry.payload]["data"].attributes[
                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
            ]
            rdmol = offsb.rdutil.mol.build_from_smiles(smi)
            atom_map = offsb.rdutil.mol.atom_map(rdmol)
            map_inv = offsb.rdutil.mol.atom_map_invert(atom_map)

            primitives = self._to.db[entry.payload]["data"]

            graph_ic = None
            graph = self._to.db[entry.payload]["data"].get("graph", None)

            for mol in QCA.node_iter_depth_first(entry, select="Molecule"):
                if graph is not None and graph_ic is None:
                    graph_ic = {
                        "b": graph.bonds(),
                        "a": graph.angles(),
                        "i": graph.outofplanes(),
                        "t": graph.torsions(),
                        "n": graph.atom,
                    }
                    graph_ic = {k: v for x in graph_ic.values() for k, v in x.items()}

                # with tempfile.NamedTemporaryFile(mode="wt") as f:
                #     offsb.qcarchive.qcmol_to_xyz(
                #         QCA.db[mol.payload]["data"], fnm=f.name
                #     )
                #     gmol = geometric.molecule.Molecule(f.name, ftype="xyz")
                # # with open("out.xyz", mode="wt") as f:
                # #     offsb.qcarchive.qcmol_to_xyz(QCA.db[mol.payload]["data"], fd=f)

                # ic_prims = geometric.internal.PrimitiveInternalCoordinates(
                #     gmol,
                #     build=True,
                #     connect=True,
                #     addcart=False,
                #     constraints=None,
                #     cvals=None,
                # )
                ic_prims = ic_op.db[entry.payload]["data"]

                for ic_type, prim_fn in prim_table.items():
                    for unmapped, param_name in labeler.db[entry.payload]["data"][
                        ic_type
                    ].items():
                        # forward map to QCA index, which is what ICs use

                        aidx = [atom_map[i] - 1 for i in unmapped]

                        if prim_fn is geometric.internal.OutOfPlane:
                            aidx = ImproperDict.key_transform(aidx)

                            # this is needed for the code below, where we use
                            # unmapped as the key, which must be sorted (in the
                            # same way openff sorts
                            unmapped = ImproperDict.key_transform(unmapped)
                            param_name = "i"
                        else:
                            unmapped = ValenceDict.key_transform(unmapped)
                            aidx = ValenceDict.key_transform(aidx)

                        new_ic = prim_fn(*aidx)
                        if new_ic not in ic_prims.Internals:
                            # Is it ok if we add extra ICs to the primitive list?
                            # if prim_fn is not geometric.internal.OutOfPlane:
                            #     print("Adding an IC:", new_ic, "which may indicate missing coverage by the FF!")
                            ic_prims.add(new_ic)

                        # if the tuple has no param (impropers), skip it
                        try:
                            if self._to.chembit:
                                prim = primitives[unmapped]
                            elif graph_ic is not None:
                                prim = graph_ic[unmapped]
                            else:
                                prim = prim_to_graph[param_name[0]].from_string_list(
                                    primitives[unmapped], sorted=True
                                )

                            prim_map = self._prim.get(entry.payload)
                            if prim_map is None:
                                self._prim[entry.payload] = {unmapped: prim}
                            else:
                                prim_map[unmapped] = prim

                        except Exception as e:
                            breakpoint()
                            print("Issue with assigning primitive! Error message:")
                            print(e)

                # to estimate the FCs, we need the gradients and the hessians
                if QCA[mol.parent].name == "Gradient":
                    for hessian_node in QCA.node_iter_depth_first(
                        mol, select="Hessian"
                    ):

                        xyz = QCA.db[mol.payload]["data"].geometry
                        hess = QCA.db[hessian_node.payload]["data"].return_result
                        grad = np.array(
                            QCA.db[hessian_node.payload]["data"].extras["qcvars"][
                                "CURRENT GRADIENT"
                            ]
                        )

                        ic_hess = ic_prims.calcHess(xyz, grad, hess)

                        # eigs = np.linalg.eigvalsh(ic_hess)
                        # s = np.argsort(np.diag(ic_hess))
                        # force_vals = eigs
                        force_vals = np.diag(ic_hess)
                        # ic_vals = [ic_prims.Internals[i] for i in s]
                        ic_vals = ic_prims.Internals
                        for aidx, val in zip(ic_vals, force_vals):
                            key = tuple(
                                map(
                                    lambda x: map_inv[int(x) - 1],
                                    str(aidx).split()[1].split("-"),
                                )
                            )

                            if type(aidx) is geometric.internal.OutOfPlane:
                                key = ImproperDict.key_transform(key)
                            else:
                                key = ValenceDict.key_transform(key)

                            # no conversion, this is done elsewhere, e.g. set_parameter
                            # if type(aidx) is geometric.internal.Distance:
                            #     val = val / (offsb.tools.const.bohr2angstrom ** 2)
                            # val *= offsb.tools.const.hartree2kcalmol

                            # if mol.payload == "QCM-1396980":
                            #     breakpoint()
                            #     print("HI")
                            if mol.payload not in self._fc:
                                self._fc[mol.payload] = {key: val}
                            else:
                                self._fc[mol.payload][key] = val

    def print_label_assignments(self, hide_ignored=True):

        """
        print the entry, atoms, prim, label, label smarts
        """

        handlers = [
            self[x]
            for x in self.root().children
            if self[x].payload in self.parameterize_handlers
        ]

        QCA = self._po.source.source
        ciehms = "canonical_isomeric_explicit_hydrogen_mapped_smiles"

        print("\nPRINT OUT OF MOLECULE ASSIGNMENTS\n")

        self.to_smirnoff_xml("tmp.offxml", verbose=False, hide_ignored=hide_ignored)
        labeler = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA).assign_labels_from_openff(
            "tmp.offxml", "tmp.offxml"
        )

        params = {
            lbl: param
            for ph in handlers
            for lbl, param in labeler.db["ROOT"]["data"][ph.payload].items()
        }

        n_entries = 0
        for entry in QCA.iter_entry():
            n_entries += 1

        for i, entry in enumerate(QCA.iter_entry(), 1):

            labels = {
                key_transformer[ph.payload](aidx): lbl
                for ph in handlers
                for aidx, lbl in labeler.db[entry.payload]["data"][ph.payload].items()
            }
            index_str = "{:6d}/{:6d}".format(i, n_entries)
            print("   ", index_str, entry.payload)
            print(
                "        ",
                index_str,
                QCA.db[entry.payload]["data"].attributes[ciehms],
            )

            # prims = self._prim[entry.payload]
            graph = self._to.db[entry.payload]["data"].get("graph")
            graph_ic = None
            if graph is not None:
                graph_ic = {
                    "b": graph.bonds(),
                    "a": graph.angles(),
                    "i": graph.outofplanes(),
                    "t": graph.torsions(),
                    "n": graph.atoms(),
                }
                graph_ic = {k: v for x in graph_ic.values() for k, v in x.items()}

            for j, aidx in enumerate(labels, 1):
                lbl = labels[aidx]
                if lbl is None:
                    breakpoint()
                    continue
                if hide_ignored and lbl in self._ignore_parameters:
                    continue
                if graph_ic is not None:
                    smarts = graph_ic[aidx].to_smarts()
                elif self._to.chembit:
                    smarts = self._to.db[entry.payload]["data"][aidx].to_smarts()
                else:
                    smarts = "".join(self._to.db[entry.payload]["data"][aidx])

                measurement = self._ic

                index_str = "{:6d}/{:6d}".format(j, len(labels))
                print(
                    "        ",
                    index_str,
                    f"{lbl:5s}",
                    aidx,
                    params[lbl]["smirks"],
                    smarts,
                    # prims[aidx],
                )
            print("---------------------------------")

        print("#################################")
        sys.stdout.flush()

    ###

    def labels(self, force=False):

        QCA = self._po.source.source
        self._qca = QCA
        qcasb = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA)
        qcasb.verbose = True

        if self._labeler is None or force:
            self.to_smirnoff_xml("tmp.offxml", verbose=False)
            labeler = qcasb.assign_labels_from_openff("tmp.offxml", "tmp.offxml")
            self._labeler = labeler
        else:
            return self._labeler

        return labeler

    def _create_universe(self):

        QCA = self._qca

        n_entries = len(list(QCA.iter_entry()))

        for entry in tqdm.tqdm(
            QCA.iter_entry(),
            total=n_entries,
            desc="Generating universe",
            ncols=80,
            disable=True,
        ):
            graph = self._to.db[entry.payload]["data"].get("graph", None)

            if graph is None:
                continue

            self.atom_universe = functools.reduce(
                lambda x, y: x + y,
                iter(x.nodes[x._primary[0]]["primitive"] for x in graph.atom.values()),
                self.atom_universe,
            )
            self.bond_universe = functools.reduce(
                lambda x, y: x + y,
                iter(
                    x.edges[x._primary[0], x._primary[1]]["primitive"]
                    for x in graph.bonds().values()
                ),
                self.bond_universe,
            )

        print("Atom universe: ", self.atom_universe.to_smarts())
        print("Bond universe: ", self.bond_universe.to_smarts())

    def _combine_reference_data(self, QCA=None, prims=True):
        """
        Collect the expected parameter values directly from the reference QM
        molecules
        """
        # import geometric.internal
        # import offsb.ui.qcasb

        # import geometric.molecule

        if self._ref_data_cache is not None:
            print("ref_data is cached:", self._ref_data_cache.keys())
            return self._ref_data_cache

        if QCA is None:
            QCA = self._po.source.source

        self._qca = QCA

        if hasattr(self, "_qca"):
            if self._qca is not QCA and hasattr(self, "_ic"):
                del self._ic

        self._calculate_ic_force_constants()

        self._create_universe()

        # need this for measuring geometry
        # should only need to do it once

        # qcasb = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA)

        # vtable = {
        #     "Bonds": qcasb.measure_bonds,
        #     "Angles": qcasb.measure_angles,
        #     "ImproperTorsions": qcasb.measure_outofplanes,
        #     "ProperTorsions": qcasb.measure_dihedrals,
        # }

        labeler = self.labels()

        # graphs = {}

        pool = concurrent.futures.ProcessPoolExecutor()

        param_data = {}
        work_list = []
        for i, entry in enumerate(QCA.iter_entry()):
            mols = list(QCA.node_iter_depth_first(entry, select="Molecule"))
            graph = self._to.db[entry.payload]["data"].get("graph", None)

            work = pool.submit(
                parameter_data,
                i,
                entry,
                mols,
                graph,
                self._ic,
                self._fc,
                labeler.db[entry.payload]["data"],
                labeler.db["ROOT"]["data"],
                self._prim[entry.payload],
                self._ignore_parameters,
            )
            work_list.append(work)

        for work in tqdm.tqdm(
            concurrent.futures.as_completed(work_list),
            total=len(work_list),
            desc="Parameter data",
            ncols=80,
            disable=False,
        ):
            i, entry_data = work.result()
            for lbl, dat in entry_data.items():
                if lbl not in param_data:
                    param_data[lbl] = dat
                else:
                    for prim_key, prim_data in dat.items():
                        payload = param_data[lbl].setdefault(
                            prim_key,
                            dict(measure=[], force=[], smarts=prim_data.get("smarts")),
                        )
                        for measure_key in ["measure", "force"]:
                            payload[measure_key].extend(prim_data[measure_key])

        # n_entries = len(list(QCA.iter_entry()))
        # for entry in tqdm.tqdm(
        #     QCA.iter_entry(),
        #     total=n_entries,
        #     desc="Parameter data",
        #     ncols=80,
        #     disable=False,
        # ):
        #     # need to unmap... sigh
        #     # smi = QCA.db[entry.payload]['data'].attributes['canonical_isomeric_explicit_hydrogen_mapped_smiles']
        #     # rdmol = offsb.rdutil.mol.build_from_smiles(smi)
        #     # atom_map = offsb.rdutil.mol.atom_map(rdmol)
        #     # map_inv = offsb.rdutil.mol.atom_map_invert(atom_map)
        #     labels = {
        #         key_transformer[ic_type](aidx): val
        #         for ic_type in vtable
        #         for aidx, val in labeler.db[entry.payload]["data"][ic_type].items()
        #         if val not in self._ignore_parameters
        #     }
        #     graph = self._to.db[entry.payload]["data"].get("graph", None)
        #     if graph is not None:
        #         graphs[entry.payload] = graph

        #     for ic_type, ic in self._ic.items():

        #         params = labeler.db["ROOT"]["data"][ic_type]

        #         for param_name, param in params.items():
        #             # if select is not None and param_name not in select:
        #             #     continue
        #             if param_name not in param_data:
        #                 param_data[param_name] = {}
        #             for mol in QCA.node_iter_depth_first(entry, select="Molecule"):
        #                 ic_data = ic.db[mol.payload]
        #                 for aidx, vals in ic_data.items():  # for aidx in labels:

        #                     if aidx not in labels:
        #                         continue

        #                     if labels[aidx] == param_name:
        #                         # param_vals.extend(vals)
        #                         if aidx not in self._prim[entry.payload]:
        #                             breakpoint()
        #                             print("HI!")

        #                         key = self._prim[entry.payload][aidx]
        #                         # if aidx not in self._fc[mol.payload]:
        #                         #     breakpoint()
        #                         #     print("HI!")

        #                         payload = param_data[param_name].setdefault(
        #                             key, dict(measure=list(), force=list())
        #                         )
        #                         payload["smarts"] = key.to_smarts()
        #                         # if key not in param_data[param_name]:
        #                         #     param_data[param_name][key] = {
        #                         #         "measure": [],
        #                         #         "force": [],
        #                         #     }

        #                         payload["measure"].extend(vals)
        #                         # param_data[param_name][key]["measure"].extend(vals)

        #                         mol_fc = self._fc.get(mol.payload)
        #                         if mol_fc is not None:
        #                             force_vals = mol_fc[aidx]

        #                             payload["force"].append(force_vals)
        #                             # param_data[param_name][key]["force"].append(
        #                             #     force_vals
        #                             # )

        self._ref_data_cache = param_data

        # This has the graph info for the ICs in param data
        with open("chemper.p", "wb") as f:
            pickle.dump(self._to, f)

        print("Reference data generated:", self._ref_data_cache.keys())
        pool.shutdown()
        return param_data

    ###

    def _combine_optimization_data(self):

        QCA = self._po.source.source

        # smi_to_label = self._po._setup.labeler.db["ROOT"]["data"]
        # smi_to_label = {
        #     k: v["smirks"] for keys in smi_to_label.values() for k, v in keys.items()
        # }
        # smi_to_label = {v: k for k, v in smi_to_label.items()}

        if self._param_data_cache is not None:
            print("Parameter data cached:", self._param_data_cache[0].keys())
            return self._param_data_cache

        param_names = self._po.db["ROOT"]["data"]

        param_labels = [
            param.split("/")[-1]
            for param in param_names
            if param not in self._ignore_parameters
        ]
        # param_labels = [x.payload for x in self.node_iter_depth_first(self.root()) if x.payload[0] in ['nbait']]

        # try:
        #     param_labels = [smi_to_label[param.radian("/")[-1]] for param in param_names]
        # except KeyError as e:
        #     print("KeyError! Dropping to a debugger")
        #     breakpoint()
        #     print("KeyError: these keys were from FB")
        #     print([param.split("/")[-1] for param in param_names])
        #     print("The FF keys are:")
        #     print(smi_to_label)

        param_data = {}
        all_data = {}
        print(
            "Parsing data from physical optimizer, smarts generator, and molecule data"
        )

        print("po params is", param_names)

        labeler = self.labels()

        # print("Labels are")
        # print(labeler.db['ROOT'])

        n_entries = len(list(QCA.iter_entry()))
        for i, entry in enumerate(QCA.iter_entry()):

            print("    {:8d}/{:8d} : {}".format(i + 1, n_entries, entry))
            if entry.payload not in labeler.db:
                print("\nSkipping case 1")
                continue
            labels = labeler.db[entry.payload]["data"]
            labels = {
                k: v
                for keys in labels.values()
                for k, v in keys.items()
                if v not in self._ignore_parameters
            }
            print("{:18s} Params: ".format(""), end="")
            for p in set(labels.values()):
                print(
                    "{}: {}".format(p, sum([1 for x in labels.values() if p == x])),
                    end=" ",
                )
            print()

            if entry.payload not in self._to.db:
                print("\nSkipping case 2")
                continue

            primitives = self._to.db[entry.payload]["data"]

            graph_ic = None
            graph = self._to.db[entry.payload]["data"].get("graph", None)

            if graph is not None:
                graph_ic = {
                    "b": graph.bonds(),
                    "a": graph.angles(),
                    "i": graph.outofplanes(),
                    "t": graph.torsions(),
                    "n": graph.atoms(),
                }

            # This is for per-molecule info
            for molecule in QCA.node_iter_depth_first(entry):
                mol_id = molecule.payload

                obj = self._po.db.get(mol_id)
                if obj is None:
                    # probably an initial molecule
                    # print("case 3: physical optimizer had no input")
                    continue

                obj = obj["data"].copy()

                # This is for an optgeo style, where there is a direct
                # objective per IC
                # IC keys (not vdW)
                ref_prims = {}
                for key in [k for k in obj if type(k) == tuple and k in labels]:
                    matched_params = []
                    for j, val in enumerate(obj[key]["dV"]):
                        if labels[key] == param_labels[j]:
                            matched_params.append(val)

                    lbl = labels[key]

                    if len(matched_params) == 0:
                        print(
                            "This label didn't match:", key, "actual label", labels[key]
                        )
                        print("Debug: param_labels is", param_labels)
                        # if we have no matches, then likely we are not trying to
                        # fit to it, so we can safely skip
                        continue

                    # if lbl == 't3':
                    #     breakpoint()
                    # for groups

                    # for graphs
                    prim = None
                    if graph_ic is not None:
                        prim = graph_ic[lbl[0]][key]
                    else:
                        prim = prim_to_graph[lbl[0]].from_string_list(primitives[key])

                    if lbl not in ref_prims:
                        ref_prims[lbl] = prim
                    elif graph is None:
                        # don't want to align entire graphs
                        prim.align_to(ref_prims[lbl])

                    if lbl not in param_data:
                        param_data[lbl] = {}
                        all_data[lbl] = {}

                    sma = prim.to_smarts()
                    obj["smarts"] = sma
                    if prim not in param_data[lbl]:
                        param_data[lbl][prim] = [matched_params]
                        all_data[lbl][prim] = [obj[key]["dV"]]
                        all_data[lbl][prim] = [obj[key]["dV"]]
                    else:
                        param_data[lbl][prim].append(matched_params)
                        all_data[lbl][prim].append(obj[key]["dV"])

                # vdW keys (gradient spread out evenly over atoms)
                # sums the two terms (e.g. epsilon and rmin_half)
                # then divides by the number of atoms and the number
                # of ff terms. This aims to make the FF param gradient sum
                # and the IC sum equivalent (but spread over atoms), so
                # for example the gradient sum for n1 epsilon and rmin_half
                # will be equal to the sum of all atoms that match n1 (0,), (1,), etc.
                vdw_keys = [x for x in labels if len(x) == 1]
                for key in vdw_keys:
                    ff_grad = []
                    matched_params_idx = [
                        i for i, lbl in enumerate(param_labels) if lbl == labels[key]
                    ]
                    if len(matched_params_idx) == 0:
                        # This means that we are not considering this param,
                        # so we can safely skip
                        continue

                    for i in matched_params_idx:
                        try:
                            ff_grad.append(
                                sum([obj[k]["dV"][i] for k in obj if type(k) == tuple])
                            )
                        except Exception as e:
                            breakpoint()
                            print("error")

                    # since we are applying the entire FF gradient to this single
                    # IC, we need to divide by the number of ICs which also match
                    # this param. This causes the gradient to be spread out over
                    # all matched ICs
                    # Also, we need to spread out over the entire parameter list
                    # so divide by number of FF params as well
                    # Finally, we need a denom-like scalar to make the magnitudes
                    # similar to existing parameters; it is currently chosen to
                    # be on the scale of bonds (very sensitive)

                    denom = 1.0 if len(key) > 1.0 else VDW_DENOM
                    for i, _ in enumerate(ff_grad):
                        ff_grad[i] /= (
                            len([x for x in vdw_keys if labels[x] == labels[key]])
                            * denom
                            # * len(param_labels)
                        )

                    matched_params = ff_grad
                    all_params = list([np.mean(ff_grad)] * len(param_labels))
                    lbl = labels[key]
                    prim = prim_to_graph[lbl[0]].from_string_list(primitives[key])
                    if lbl not in param_data:
                        param_data[lbl] = {}
                        all_data[lbl] = {}
                    if prim not in param_data[lbl]:
                        param_data[lbl][prim] = [matched_params]
                        all_data[lbl][prim] = [all_params]
                    else:
                        param_data[lbl][prim].append(matched_params)
                        all_data[lbl][prim].append(all_params)
            # for object in QCA.node_iter_depth_first(entry):
            #     if object.name == "Molecule":
            #         continue
            #     object_id = object.payload

            #     obj = self._po.db.get(object_id)
            #     if obj is None:
            #         continue

            #     obj = obj["data"]

            #     # IC keys (not vdW)
            #     for key in [k for k in obj if type(k) == tuple and k in labels]:
            #         matched_params = []
            #         for j, val in enumerate(obj[key]["dV"]):
            #             if labels[key] == param_labels[j]:
            #                 matched_params.append(val)

            #         lbl = labels[key]

            #         if len(matched_params) == 0:
            #             # print("This label didn't match:", key, "actual label", labels[key])
            #             # print("Debug: param_labels is", param_labels )
            #             # if we have no matches, then likely we are not trying to
            #             # fit to it, so we can safely skip
            #             continue

            #         # if lbl == 't3':
            #         #     breakpoint()
            #         prim = prim_to_graph[lbl[0]].from_string_list(primitives[key])
            #         if lbl not in param_data:
            #             param_data[lbl] = {}
            #             all_data[lbl] = {}
            #         if prim not in param_data[lbl]:
            #             param_data[lbl][prim] += [matched_params]
            #             all_data[lbl][prim] += [obj[key]["dV"]]

            #         param_data[lbl][prim].append(matched_params)
            #         all_data[lbl][prim].append(obj[key]["dV"])

            #     # vdW keys (gradient spread out evenly over atoms)
            #     # sums the two terms (e.g. epsilon and rmin_half)
            #     # then divides by the number of atoms and the number
            #     # of ff terms. This aims to make the FF param gradient sum
            #     # and the IC sum equivalent (but spread over atoms), so
            #     # for example the gradient sum for n1 epsilon and rmin_half
            #     # will be equal to the sum of all atoms that match n1 (0,), (1,), etc.
            #     vdw_keys = [x for x in labels if len(x) == 1]
            #     for key in vdw_keys:
            #         ff_grad = []
            #         matched_params_idx = [
            #             i for i, lbl in enumerate(param_labels) if lbl == labels[key]
            #         ]
            #         if len(matched_params_idx) == 0:
            #             # This means that we are not considering this param,
            #             # so we can safely skip
            #             continue

            #         # this is the sum over all IC, for FF param i (the vdW terms)
            #         for i in matched_params_idx:
            #             ff_grad.append(
            #                 sum([obj[k]["dV"][i] for k in obj if type(k) == tuple])
            #             )

            #         # since we are applying the entire FF gradient to this single
            #         # IC, we need to divide by the number of ICs which also match
            #         # this param. This causes the gradient to be spread out over
            #         # all matched ICs
            #         # Also, we need to spread out over the entire parameter list
            #         # so divide by number of FF params as well
            #         # Finally, we need a denom-like scalar to make the magnitudes
            #         # similar to existing parameters; it is currently chosen to
            #         # be on the scale of bonds (very sensitive)

            #         for i, _ in enumerate(ff_grad):
            #             ff_grad[i] /= (
            #                 len([x for x in vdw_keys if labels[x] == labels[key]])
            #                 * VDW_DENOM
            #                 # * len(param_labels)
            #             )

            #         matched_params = ff_grad
            #         all_params = list([np.mean(ff_grad)] * len(param_labels))
            #         lbl = labels[key]
            #         prim = prim_to_graph[lbl[0]].from_string(primitives[key][0])
            #         if lbl not in param_data:
            #             param_data[lbl] = {}
            #             all_data[lbl] = {}
            #         if prim not in param_data[lbl]:
            #             param_data[lbl][prim] = [matched_params]
            #             all_data[lbl][prim] = [all_params]

            #         param_data[lbl][prim].append(matched_params)
            #         all_data[lbl][prim].append(all_params)
        self._param_data_cache = (param_data, all_data)
        print("Parameter data generated:", self._param_data_cache[0].keys())
        return param_data, all_data

    def _plot_gradients(self, fname_prefix=""):

        import matplotlib
        import matplotlib.pyplot as plt

        labeler = self._po._setup.labeler
        QCA = self._po.source.source
        params = self._po._forcefield.plist

        # now we need to plot the data

        # plot the raw gradient matrix

        matplotlib.use("Qt5Agg")

        off = self.db["ROOT"]["data"]
        labels = {}
        for param in params:
            ph_name, _, _, pid = param.split("/")
            labels[param] = pid
            # ff_params = off.get_parameter_handler(ph_name).parameters
            # for ff_param in ff_params:
            #     if ff_param.smirks == smirks:
            #         labels[param] = ff_param.id
            #         break

        prim_dict = {}

        n_mols = len(list(QCA.node_iter_depth_first(QCA.root(), select="Molecule")))
        for mol_node in tqdm.tqdm(
            QCA.node_iter_depth_first(QCA.root(), select="Molecule"),
            total=n_mols,
            desc="Gradient plot",
        ):

            if mol_node.payload not in self._po.db:
                continue

            fb_data = self._po.db.get(mol_node.payload, None)
            if fb_data is None:
                continue

            entry = next(QCA.node_iter_to_root(mol_node, select="Entry"))

            if entry.payload not in self._to.db:
                continue

            smi = QCA.db[entry.payload]["data"].attributes[
                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
            ]
            qcmol = QCA.db[mol_node.payload]["data"]

            rdmol = offsb.rdutil.mol.rdmol_from_smiles_and_qcmol(smi, qcmol)
            rdmol.RemoveAllConformers()
            for atom in rdmol.GetAtoms():
                atom.SetAtomMapNum(0)
            offsb.rdutil.mol.save2d(rdmol, mol_node.payload + ".2d.png", indices=True)

            fb_data = fb_data["data"]

            ic_indices = [ic for ic in fb_data if type(ic) == tuple]

            dV = np.vstack([x["dV"] for ic, x in fb_data.items() if type(ic) == tuple])
            ic_names = [
                "-".join(map(str, [i + 1 for i in ic]))
                for ic in fb_data
                if type(ic) == tuple
            ]

            lim = np.abs(dV).max()
            lim = 10

            n_ic = len(ic_names)
            x_multiple = 0.3
            label_rhs = True
            # if len(params) / n_ic > 3:
            #     label_rhs = False
            #     x_multiple = 0.8
            y_multiple = 0.4
            # if len(params) / n_ic < 1 / 3:
            #     y_multiple = 1.0

            leading_dim = max(x_multiple * n_ic, y_multiple * len(params))
            dpi = min(100, int(2 ** 16 / leading_dim))
            dpi = 300
            # fig = plt.figure(
            #     figsize=(.0+x_multiple * n_ic, 1.0 + y_multiple * len(params)), dpi=dpi
            # )
            fig = plt.figure(figsize=(0.5 * 9 / 10 * n_ic, 0.6 * len(params)), dpi=dpi)
            matplotlib.rc("font", size=12, family="monospace")
            ax = fig.add_subplot(111)

            image = ax.imshow(dV.T, vmin=-lim, vmax=lim, cmap=plt.cm.get_cmap("bwr_r"))

            points_x = []
            points_y = []
            labeled_params = []
            labeled_params_no_smi = []
            last_name = ""
            sep_line_letters = []
            for i, param_rows in enumerate(dV.T):
                found = False
                for j, dv_val in enumerate(param_rows):
                    ph_name = params[i].split("/")[0]
                    try:
                        ff_label = labeler.db[entry.payload]["data"][ph_name].get(
                            ic_indices[j]
                        )
                    except:
                        print(ph_name, ic_indices[j])
                    if ff_label is None:
                        continue
                    if j == 0:
                        last_name = ff_label[0]
                    elif (
                        last_name != ff_label[0] and ff_label[0] not in sep_line_letters
                    ):
                        ax.axvline(x=j - 0.5, lw=1, alpha=0.8, ls="--", color="k")
                        last_name = ff_label[0]
                        sep_line_letters.append(ff_label[0])
                    # if last_name == "n":
                    #     breakpoint()
                    if labels[params[i]] == ff_label:
                        try:
                            smarts = self._to.db[entry.payload]["data"][ic_indices[j]]
                        except KeyError:
                            # breakpoint()
                            pass
                        param = params[i].split("/")[2]

                        if smarts not in prim_dict:
                            prim_dict[smarts] = {ff_label: {param: [dV[j][i]]}}
                        elif ff_label not in prim_dict[smarts]:
                            prim_dict[smarts][ff_label] = {param: [dV[j][i]]}
                        elif param not in prim_dict[smarts][ff_label]:
                            prim_dict[smarts][ff_label][param] = [dV[j][i]]
                        else:
                            prim_dict[smarts][ff_label][param].append(dV[j][i])

                        if ff_label not in ic_names[j]:
                            ic_names[j] += " " + ff_label
                        points_x.append(j)
                        points_y.append(i)
                        found = True

                labeled_params.append(
                    "{:<4s} {:>11s}".format(
                        labels[params[i]],
                        params[i].split("/")[2],
                    )
                )
                labeled_params_no_smi.append(
                    "{:<4s} {:<11s}".format(labels[params[i]], params[i].split("/")[2])
                )
                if not found:
                    ax.axhspan(
                        ymin=i - 0.5, ymax=i + 0.5, alpha=0.1, lw=0, color="gray"
                    )

            ax.plot(points_x, points_y, "o", ms=25, mfc="none", mec="k", lw=2)

            ax.set_xticks(range(n_ic))
            ax.set_xticklabels(
                ic_names, rotation=45, ha="right", rotation_mode="anchor"
            )

            ax.set_yticks(range(len(params)))
            ax.set_yticklabels(labeled_params)

            if label_rhs:
                ax2 = ax.secondary_yaxis("right")
                ax2.set_yticks(range(len(params)))
                ax2.set_yticklabels(labeled_params_no_smi)

            ax3 = ax.secondary_xaxis("top")
            ax3.set_xticks(range(n_ic))
            ax3.set_xticklabels(
                ic_names, rotation=45, ha="left", rotation_mode="anchor"
            )

            ax.grid(alpha=0.3)

            # fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(
                ".".join(
                    [mol_node.payload, "{}".format(fname_prefix), "gradient", "png"]
                ),
                dpi=dpi,
            )

            plt.close(fig)
            del fig

            # u, s, vh = np.linalg.svd(dV.T)
            # with open(mol_node.payload + ".eigvals.dat", "w") as fid:
            # [fid.write("{:12.8e}\n".format(si ** 0.5)) for si in s]

        self.prim_dict = prim_dict

    def _run_optimizer(self, jobtype, keep_state=False):

        verbose = True

        keep_trust = False
        newff_name = "tmp.offxml"

        options_override = {}
        if self.trust0 is not None:
            options_override["trust0"] = self.trust0
            print("Setting trust0 to", self.trust0)
        elif self._po._options:
            self.trust0 = self._po._options.get("trust0")

        if self.finite_difference_h is not None:
            options_override["finite_difference_h"] = self.finite_difference_h
            print("Setting finite_difference_h to", self.finite_difference_h)
        elif self._po._options:
            self.finite_difference_h = self._po._options.get("finite_difference_h")

        if self.eig_lowerbound:
            options_override["eig_lowerbound"] = self.eig_lowerbound
            print("Setting eig_lowerbound to", self.eig_lowerbound)
        elif self._po._options:
            self.eig_lowerbound = self._po._options.get("eig_lowerbound")

        trust = self.trust0
        fdh = self.finite_difference_h
        eps = self.eig_lowerbound

        while True:
            try:
                self._po.load_options(options_override=options_override)
                print("Running calculation! Below is the FF")
                self.to_smirnoff_xml(newff_name, verbose=verbose)
                self._po._setup.ff_fname = newff_name
                self._po.ff_fname = newff_name
                self._po._init = False

                self._po.logger.setLevel(logging.ERROR)
                self._po.apply(jobtype=jobtype)
                self._po.logger.setLevel(self.logger.getEffectiveLevel())
                self.load_new_parameters(self._po.new_ff)

                print("Calculation done! Below is the FF")
                self.to_smirnoff_xml(None, verbose=verbose)

                break
            except RuntimeError:
                self._bump_zero_parameters(1e-3, names="epsilon")
                self.to_smirnoff_xml(newff_name, verbose=False)
                self._po._setup.ff_fname = newff_name
                self._po.ff_fname = newff_name
                self._po._init = False

                self.trust0 = self._po._options["trust0"] / 2.0
                self.finite_difference_h = (
                    self._po._options["finite_difference_h"] / 2.0
                )
                print(
                    "Job failed; reducing trust radius to",
                    self.trust0,
                    "finite_difference_h",
                    self.finite_difference_h,
                )
                self._po._options["trust0"] = self.trust0
                self._po._options["finite_difference_h"] = self.finite_difference_h
                mintrust = self._po._options["mintrust"]
                if self.trust0 < mintrust:
                    return False

        if keep_state:
            self.trust0 = self._po._options.get("trust0")
            self.finite_difference_h = self._po._options.get("finite_difference_h")
            self.eig_lowerbound = self._po._options.get("eig_lowerbound")
        else:
            self.trust0 = trust
            self.finite_difference_h = fdh
            self.eig_lowerbound = eps

        if self.trust0 is None:
            self.trust0 = self._po._options.get("trust0")
        if self.finite_difference_h is None:
            self.finite_difference_h = self._po._options.get("finite_difference_h")
            fdh = self.finite_difference_h
        if self.eig_lowerbound is None:
            self.eig_lowerbound = self._po._options.get("eig_lowerbound")

        # self.trust0 = max(1e-5, self.trust0)
        # if self.trust0 < 1e-4:
        #     self.finite_difference_h = 1e-5

        return True

    def _optimize_type_iteration(
        self,
        optimize_during_typing=False,
        optimize_during_scoring=False,
        ignore_bits=None,
        use_gradients=True,
        split_strategy="spatial_reference",
        ignore_parameters=None,
        only_parameters=None,
        step=0,
        operation="new",
    ):

        jobtype = "GRADIENT"

        candidate_limit = np.inf

        if self.split_candidate_limit is not None:
            candidate_limit = self.split_candidate_limit

        candidates = []

        grad_new = 0
        grad = 0
        i = 0
        node = None
        base_ref_obj = self._po.X
        ref_obj = self._po.X
        has_ref_obj = base_ref_obj is not None
        grad_scale = 1.0

        if ignore_bits is None:
            ignore_bits = {}

        if ignore_parameters is None:
            ignore_parameters = []

        ignore_parameters.extend(self._ignore_parameters)

        if only_parameters is None:
            only_parameters = []

        if operation == "delete":
            ignore_parameters = ignore_parameters.copy()
            ignore_parameters.extend(only_parameters)
        # BREAK
        # breakpoint()

        if use_gradients and not has_ref_obj:
            print("Running reference gradient calculation...")

            success = self._run_optimizer(jobtype, keep_state=False)

            if not success:
                print("Reference gradient calculation failed; cannot continue!")
                return None, np.inf, -np.inf, None

            obj = self._po.X
            grad = self._po.G

            best = [
                None,
                grad * grad_scale,
                obj,
                None,
                None,
                -1,
                None,
                None,
                self.db.copy(),
                None,
                None,
                None,
                None,
                None,
                None,
            ]
        else:
            best = [
                None,
                self._po.G,
                self._po.X,
                None,
                None,
                -1,
                None,
                None,
                self.db.copy(),
                None,
                None,
                None,
                None,
                None,
                None,
            ]

        bit_gradients = bit_gradient_factory()
        bit_cache = {}

        del self._param_data_cache
        self._param_data_cache = None

        # self._calculate_ic_force_constants()
        # import cProfile
        # cProfile.runctx("self._combine_reference_data()", globals(), locals(), "profile.log")
        # assert False
        print("#### Generating Reference data ####")

        ref_data = self._combine_reference_data()
        with open("ref_data.p", "wb") as f:
            pickle.dump(ref_data, f)

        # print("Assignments from initial FF:")
        # self.print_label_assignments()

        current_ff = "current.offxml"
        self.to_smirnoff_xml(current_ff, verbose=False)

        olddb = copy.deepcopy(self._po.db)
        oldindex = copy.deepcopy(self.node_index)
        oldffdb = copy.deepcopy(self.db)
        oldfbff = copy.deepcopy(self._po._forcefield)

        eps = 1.0

        # # This generates the tier 1 scoring data
        if split_strategy == "spatial_reference":
            key = "measure"
            mode = ["mean_difference"]
            eps = 1.0
            # param_data = self._combine_reference_data()
            param_data = ref_data
        elif split_strategy == "force_reference":
            key = "force"
            mode = ["mean_difference"]
            eps = 10.0
            # param_data = self._combine_reference_data()
            param_data = ref_data
        elif use_gradients:

            # None means accept everything
            eps = None
            key = None
            # mode = "sum_difference"
            # mode = ["mag_difference"]
            mode = ["mean_difference"]
            # This needs a current FB ff and db
            param_data, all_data = self._combine_optimization_data()

        if not self.gradient_assigned_only:
            param_data = all_data

        if ignore_parameters is not None and len(ignore_parameters):
            # print("Stripping parameters:", ignore_parameters)
            param_data = {
                k: v for k, v in param_data.items() if k[0] not in ignore_parameters
            }
            ref_data = {
                k: v for k, v in ref_data.items() if k[0] not in ignore_parameters
            }

        if only_parameters is not None and len(only_parameters):
            # print("Stripping all other parameters except:", only_parameters)
            param_data = {k: v for k, v in param_data.items() if k in only_parameters}
            ref_data = {k: v for k, v in ref_data.items() if k in only_parameters}

        with open("param_data_{:04d}.p".format(step), "wb") as f:
            pickle.dump(param_data, f)
        # with open("ref_data.p", "wb") as f:
        #     pickle.dump(ref_data, f)

        # This associates the indices with the labels
        # The chemper.p will associates molecules with envs
        # The param_data.p associates label with envs and measurements

        # This means we can iterate over molecules, pull the ICs with indices,
        # then look up the label, giving a mol::idx::label,graph
        with open("labels_{:04d}.p".format(step), "wb") as f:
            pickle.dump(self.labels(), f)

        candidate_mode_choices = ["split_gradient_max"]
        # candidate_mode_choices.extend(mode)

        candidate_mode = candidate_mode_choices[0]

        node = None

        while len(candidates) < candidate_limit:
            if use_gradients:
                print("\n\nMicroiter", i)

            print("Reset hierarchy from operation on", node)
            self.node_index = copy.deepcopy(oldindex)
            self.db = copy.deepcopy(oldffdb)
            # self[node.parent].children.remove(node.index)
            # self.node_index.pop(node.index)
            self._po.db = copy.deepcopy(olddb)
            self._po._forcefield = copy.deepcopy(oldfbff)

            # this should be faster the copying everything, but need to debug
            # self.load_new_parameters(current_ff)

            newff_name = "tmp.offxml"
            self.to_smirnoff_xml(newff_name, verbose=False)
            # self._po._options["forcefield"] = [newff_name]

            self._po._setup.ff_fname = newff_name
            self._po.ff_fname = newff_name
            self._po._init = False

            i += 1

            # print("Finding new split...")
            # print("Ignore bits are")
            # for ignore, grads in ignore_bits.items():
            #     print(grads, ignore)

            # This is tier 1 scoring
            # Examines the entire set and tries to find the next split
            # This means the bits split are valid until a split is kept

            if operation == "new":
                node, score = self._find_next_split(
                    param_data,
                    key=key,
                    ignore_bits=ignore_bits,
                    mode=mode,
                    eps=eps,
                    bit_gradients=bit_gradients,
                    bit_cache=bit_cache,
                )
                node_operation = "new"
                print("Split is", node)

            elif operation == "delete":
                # TODO Implement the join in full

                node, score = self._find_next_join(
                    ref_data,
                    key="measure",
                    ignore_bits=ignore_bits,
                    mode=mode,
                    eps=eps,
                    bit_gradients=bit_gradients,
                    bit_cache=bit_cache,
                )
                node_operation = "delete"
                print("Combine is", node)

            print("Score is", score)

            if node is None:
                break

            # Only resets the new/modified parameter
            term = node.payload[0]
            if term in "ti":
                print("Reevaluating periodic terms")
                ref_data = self._combine_reference_data(
                    QCA=self._to.source.source, prims=False
                )
                self.initialize_parameters_from_data(
                    QCA=self._to.source.source,
                    only_parameters=[x for x in ref_data if x == term],
                    periodicities=True,
                    report_only=False,
                )

            # if verbose:
            #     print("Assignments post operation:")
            #     self.print_label_assignments()

            if use_gradients:

                # This starts the tier 2 scoring

                # self._po._options["forcefield"] = [newff_name]
                print("Calculating new gradient from operations (tier 2 scoring)")

                # self._po.logger.setLevel(logging.ERROR)
                success = self._run_optimizer(jobtype, keep_state=False)
                # self._po.logger.setLevel(logging.ERROR)
                # self._po.logger.setLevel(self.logger.getEffectiveLevel())

                if not success:
                    print("Gradient failed for this split; skipping")
                    # if there is an exception, the po will have no data
                    self._po.db = olddb
                else:
                    obj = self._po.X

                    grad_new = self._po.G
                    hit = np.abs(grad_new - grad * grad_scale)
                    if eps is None:
                        hit = True
                    else:
                        hit = hit > eps
                    print(
                        "\ngrad_new",
                        grad_new,
                        "grad",
                        grad,
                        "grad_new > abs(grad*scale)?",
                        hit,
                        grad_new - grad * grad_scale,
                        eps,
                    )
                    print("\nObjective: ", obj)
                    print("Reference: ", ref_obj)

                    # If doing a SP results in an large increase of the
                    # objective, then skip since we assume an optimization
                    # will not be able to improve objective
                    increase = 10
                    if obj > increase * ref_obj:
                        print(
                            "Skipping this candidate since the objective is over",
                            increase,
                            "times larger",
                        )
                        continue

                    grad_new_opt = np.inf
                    if optimize_during_scoring:

                        # Note that this section might be outdated
                        print("Candidates so far (unsorted):")
                        for c in candidates:
                            print(
                                "{:3d}".format(c[5]),
                                c[14],
                                oldindex[c[2]].payload,
                                "->",
                                c[0].payload,
                                oldffdb[oldindex[c[2]].payload]["data"][
                                    "group"
                                ].to_smarts(),
                                "->",
                                c[8][c[0].payload]["data"]["group"].to_smarts(),
                                "{:.6e}".format(c[9]),
                                "{:.6e}".format(c[1]),
                                "{:.6e}".format(c[10]),
                                "{}".format(c[4]),
                                "{:.6e}".format(c[6]),
                                "{:.6e}".format(c[7]),
                                "{:8.6f}%".format(100.0 * (c[7] - c[6]) / c[6]),
                            )
                        print("Performing micro optimization for candidate")

                        success = self._run_optimizer("OPTIMIZE", keep_state=False)

                        if success:
                            obj = self._po.X
                            grad_new_opt = self._po.G
                            print("Objective after minimization:", self._po.X)
                        else:
                            obj = ref_obj
                            grad_new_opt = grad
                            self.logger.info(
                                "Optimization failed; assuming bogus split"
                            )

                    node_copy = node.copy()
                    node_copy.parent = node.parent
                    node_copy.children = node.children.copy()
                    candidate = [
                        node_copy,
                        grad_new,
                        node.parent,
                        None,
                        score,
                        len(candidates),
                        ref_obj,
                        obj,
                        copy.deepcopy(self.db),
                        grad,
                        grad_new_opt,
                        copy.deepcopy(self._po._forcefield),
                        copy.deepcopy(self._po.db),
                        copy.deepcopy(self.node_index),
                        node_operation,
                    ]
                    candidates.append(candidate)

                    # turn off optimize_during_scoring to use the heuristic
                    # here we find that single point gradient increases are best
                    # this is nicer than brute force, but still need to eval
                    # every split

                    if optimize_during_scoring:
                        # if we are going through the trouble to optimize every
                        # split, always take the best
                        candidates = sorted(
                            candidates, key=lambda x: np.abs(x[7]), reverse=False
                        )
                    elif candidate_mode == "split_gradient_max":
                        candidates = sorted(
                            candidates, key=lambda x: x[1] - x[9], reverse=True
                        )
                    elif candidate_mode in [
                        "sum_difference",
                        "mag_difference",
                        "mean_difference",
                    ]:
                        candidates = sorted(
                            candidates, key=lambda x: np.abs(x[4]), reverse=True
                        )
                    print("Candidates so far (top wins):")
                    for c in candidates:
                        print(
                            "{:3d}".format(c[5]),
                            c[14],
                            oldindex[c[2]].payload,
                            "->",
                            c[0].payload,
                            oldffdb[oldindex[c[2]].payload]["data"][
                                "group"
                            ].to_smarts(),
                            "->",
                            c[8][c[0].payload]["data"]["group"].to_smarts(),
                            "{:.6e}".format(c[9]),
                            "{:.6e}".format(c[1]),
                            "{:.6e}".format(c[10]),
                            "{}".format(c[4]),
                            "{:.6e}".format(c[6]),
                            "{:.6e}".format(c[7]),
                            "{:8.6f}%".format(100.0 * (c[7] - c[6]) / c[6]),
                        )
                    print(
                        "Key is index, from_param, new_param, total_grad_ref, total_grad_split, total_grad_opt, grad_split_score initial_obj final_obj percent_change\n"
                    )

            else:
                node_copy = node.copy()
                node_copy.parent = node.parent
                node_copy.children = node.children.copy()
                best = [
                    node_copy,
                    np.inf,
                    node.parent,
                    self.db[node.payload].copy(),
                    score,
                    0,
                    np.inf,
                    np.inf,
                    copy.deepcopy(self.db),
                    np.inf,
                    np.inf,
                    copy.deepcopy(self._po._forcefield),
                    copy.deepcopy(self._po.db),
                    copy.deepcopy(self.node_index),
                    node_operation,
                ]
                candidates = [best]
                # hack so that we add it using the common path below
                if node_operation == "new":
                    self[node.parent].children.remove(node.index)
                    self.node_index.pop(node.index)
                    self.db.pop(node.payload)
                elif node_operation == "delete":
                    # do nothing since the node is there, and remove it below
                    pass

                break

        print("First scoring pass complete; assessing candidates with tier 3 scoring")
        print("Reset hierarchy")

        self.node_index = oldindex
        self.db = oldffdb
        self._po.db = olddb
        self._po._forcefield = oldfbff

        # self.print_label_assignments()
        # self[node.parent].children.remove(node.index)
        # self.node_index.pop(node.index)

        # self.load_new_parameters(current_ff)

        newff_name = "tmp.offxml"
        self.to_smirnoff_xml(newff_name, verbose=False)
        # self._po._options["forcefield"] = [newff_name]

        self._po._setup.ff_fname = newff_name
        self._po.ff_fname = newff_name
        self._po._init = False

        if len(candidates):

            if optimize_during_scoring:
                # if we are going through the trouble to optimize every
                # split, always take the best
                candidates = sorted(
                    candidates, key=lambda x: np.abs(x[7]), reverse=False
                )

                # but bail if we didn't find any splits below the cutoff
                candidates = [
                    c
                    for c in candidates
                    if (c[7] - c[6]) / c[6] < self.split_keep_threshhold
                ]
                if len(candidates) == 0:
                    return None, np.inf, np.inf, None

            elif candidate_mode == "split_gradient_max":
                candidates = sorted(candidates, key=lambda x: x[1] - x[9], reverse=True)
            elif candidate_mode in [
                "sum_difference",
                "mag_difference",
                "mean_difference",
            ]:
                candidates = (
                    sorted(candidates, key=lambda x: np.abs(x[4]), reverse=True),
                )
            grad_new_opt = np.inf

            # this means optimize the best we found, and return the objective
            # as the score
            if optimize_during_typing and not optimize_during_scoring:
                n_success = 0
                total_candidates = (
                    self.optimize_candidate_limit
                    if self.optimize_candidate_limit
                    else len(candidates)
                )
                total_candidates = min(len(candidates), total_candidates)
                for ii, candidate in enumerate(candidates[:total_candidates], 1):
                    self.db = candidate[8]
                    self.node_index = candidate[13]
                    self._po._forcefield = candidate[11]
                    self._po.db = candidate[12]
                    node = candidate[0]

                    if candidate[14] == "new":
                        print(
                            "Add candidate parameter",
                            node,
                            "to parent",
                            self[candidate[2]],
                        )
                        # candidate[0] = self.add(candidate[2], candidate[0], index=0)
                    elif candidate[14] == "delete":
                        print(
                            "Remove candidate parameter",
                            node,
                            "from parent",
                            self[candidate[2]],
                        )
                        # self.combine_parameter(node)

                    print(
                        "Performing tier 3 micro optimization for the best candidate score number {}/{}".format(
                            ii, total_candidates
                        )
                    )
                    newff_name = "newFF_" + str(i) + "." + str(ii) + ".offxml"
                    print("Wrote FF to file", newff_name)
                    self.to_smirnoff_xml(newff_name, renumber=True, verbose=False)
                    success = self._run_optimizer("OPTIMIZE", keep_state=False)

                    if success:
                        obj = self._po.X
                        grad_new_opt = self._po.G
                        print("Objective after minimization:", self._po.X)
                        # self.load_new_parameters(self._po.new_ff)
                        candidate[7] = obj
                        candidate[10] = grad_new_opt
                        candidate[11] = copy.deepcopy(self._po._forcefield)
                        candidate[12] = copy.deepcopy(self._po.db)

                        # since we are optimizing, overwrite the SP gradient
                        candidate[1] = grad_new_opt
                        n_success += 1
                    else:
                        self.logger.info("Optimization failed; assuming bogus split")

                    # print("Remove candidate parameter", node, "from parent", node.parent, candidate[2], node.parent == candidate[2])
                    # self[node.parent].children.remove(node.index)
                    # self.node_index.pop(node.index)

                    if (
                        self.optimize_candidate_limit is not None
                        and n_success == self.optimize_candidate_limit
                    ):
                        break

                candidates = sorted(
                    candidates, key=lambda x: np.abs(x[7]), reverse=False
                )

            print("All candidates (top wins; pre cutoff filter):")
            for c in candidates:
                print(
                    "{:3d}".format(c[5]),
                    "{:7.2f}%".format(100.0 * (c[7] - c[6]) / c[6]),
                    c[14],
                    oldindex[c[2]].payload,
                    "->",
                    c[0].payload,
                    oldffdb[oldindex[c[2]].payload]["data"]["group"].to_smarts(
                        atom_universe=self.atom_universe,
                        bond_universe=self.bond_universe,
                    ),
                    "->",
                    c[8][c[0].payload]["data"]["group"].to_smarts(
                        atom_universe=self.atom_universe,
                        bond_universe=self.bond_universe,
                    ),
                    "{:.6e}".format(c[6]),
                    "{:.6e}".format(c[7]),
                )
            print(
                "Key is index, from_param, new_param, total_grad_ref, total_grad_split, total_grad_opt, grad_split_score initial_obj final_obj percent_change\n"
            )

            # assume that for geometry based scoring, we keep all higher than eps
            candidates = [
                c
                for c in candidates
                if (not use_gradients)
                or ((c[7] - c[6]) / c[6] < self.split_keep_threshhold)
            ]
            if len(candidates) == 0:
                print(
                    "No candidates meet cutoff threshhold of {:6.2f}%".format(
                        self.split_keep_threshhold * 100
                    )
                )
                self.node_index = oldindex
                self.db = oldffdb
                self._po.db = olddb
                self._po._forcefield = oldfbff
                return None, np.inf, np.inf, None
            best = candidates[0]
            # only re-add if we did a complete scan, since we terminate that case
            # with no new node, and the best has to be re-added
            # if we break early, the node is already there

            # I think nodes need to be prepended to conserve hierarchy
            # for example, if we split a param, do we want it to override
            # all children? no, since we were only focused on the parent, so
            # we only care that the split node comes after *only* the parent,
            # which is true since it is a child
            # best[0] = self.add(best[2], best[0], index=0)
            self.node_index = best[13]
            self.db = best[8]
            self._po._forcefield = best[11]
            self._po.db = best[12]

            print("Best", best[14], "result")
            print(best[0])
            print("Best split gradient", best[1])
            print("Best split score", best[4])
            print(
                "Best split objective drop {}%".format(
                    100.0 * (best[7] - best[6]) / best[6]
                )
            )
            # print("Best combine parameter")
            # print(best[0])
            # breakpoint()
            # self.combine_parameter(best[0])
            # self.print_label_assignments()

        # newff_name = "newFF.offxml"
        # self.db = best[8]
        # self.to_smirnoff_xml(newff_name, verbose=False)

        # self._po._setup.ff_fname = newff_name
        # self._po._init = False

        return best[0], best[1], best[7], best[14]

    def _guess_periodicities(
        self,
        param_name,
        values,
        report_only=False,
        max_terms=3,
        period_max=30,
        cutoff=0.9,
    ):

        "assumes distances in Bohr"

        current = None
        new = None

        param = self.db[param_name]["data"]["parameter"]

        ptype = type(param)

        modify = not report_only

        # lets not set the phases to the average angle
        # try to determine if need to set to 0 or 180 based on data
        if ptype not in [
            ImproperTorsionHandler.ImproperTorsionType,
            ProperTorsionHandler.ProperTorsionType,
        ]:
            return

        assert max_terms > 0

        stats_fn = self.stats_fn
        stats_fn.set_circular(True)

        n = np.arange(1, period_max + 1, dtype=float).reshape(1, -1)
        A = np.cos(n * np.radians(np.atleast_2d(values).T)).sum(axis=0) / len(values)

        periods = []
        phases = []
        k = []
        idivf = []

        fname = "measures." + param_name + "." + str(time.time()) + ".p"
        dataset = {"data": list(values), "periods": list(n[0]), "overlap": list(A)}
        pickle.dump(dataset, open(fname, "wb"))
        print("DEBUG: dumped data, periods, and overlap for", param_name, "to", fname)

        # print(json.dumps(list(values)))
        # print("DEBUG: here are fourier values for ", param_name, "(periods, vals)")
        # print(json.dumps(list(n[0])))
        # print(json.dumps(list(A)))
        # print("DEBUG: max_terms is", max_terms)
        for i, val in enumerate(A, 1):

            if len(periods) >= max_terms:
                break
            if np.abs(val) > cutoff:
                phase = 0 if val < 0 else 180

                # skip=False
                # for per, pha in zip(param.periodicity, param.phase):
                #     if per != i and max(per, i) % min(per, i) == 0 and pha == phase * simtk.unit.degree:
                #         skip=True
                #         break
                # for per, pha in zip(periods, phases):
                #     if max(per, i) % min(per, i) == 0 and pha == phase * simtk.unit.degree:
                #         skip=True
                #         break
                # if skip:
                #     continue

                if i in param.periodicity:
                    idx = param.periodicity.index(i)
                    periods.append(param.periodicity[idx])
                    phases.append(param.phase[idx])
                    k.append(param.k[idx])
                    idivf.append(param.idivf[idx])
                else:
                    periods.append(i)
                    phases.append(phase * simtk.unit.degree)
                    k.append(0 * simtk.unit.kilocalories_per_mole)
                    idivf.append(1.0)

        # We choose to only modify if a good guess is found
        if len(periods) > 0:
            # if 1 in param.periodicity:
            #     idx = param.periodicity.index(1)
            #     periods.append(param.periodicity[idx])
            #     phases.append(param.phase[idx])
            #     k.append(param.k[idx])
            #     idivf.append(param.idivf[idx])
            # else:
            #     periods.append(1)
            #     phases.append(0.0 * simtk.unit.degree)
            #     k.append(param.k[0])
            #     idivf.append(1.0)

            current = (param.periodicity, param.phase)
            # n_params = len(param.phase)
            # if issubclass(type(value), list):
            #     new = list(
            #         [x * simtk.unit.degree for x, _ in zip(value, range(n_params))]
            #     )
            # else:
            #     new = list([value * simtk.unit.degree for _ in range(n_params)])
            if modify:
                param.phase = phases
                param.periodicity = periods
                param.k = k
                param.idivf = idivf
                new = (param.periodicity, param.phase)

            if report_only:
                print(
                    "Would change parameter in ",
                    param_name,
                    "from periodicities =",
                    current,
                    "to",
                    new,
                    "(reporting only; did not change)",
                )
            else:
                print(
                    "Changed periodicities in ", param_name, "from", current, "to", new
                )
        else:
            print(
                f"WARNING: Could not find a good guess for periodicity of parameter {param.id}"
            )

    def _set_parameter_spatial(self, param_name, value, report_only=False):

        "assumes distances in Bohr"

        current = None
        new = None

        param = self.db[param_name]["data"]["parameter"]

        ptype = type(param)

        modify = not report_only

        if ptype == BondHandler.BondType:
            current = param.length
            new = value * offsb.tools.const.bohr2angstrom * simtk.unit.angstrom
            if modify:
                param.length = new
        elif ptype == vdWHandler.vdWType:
            current = param.rmin_half
            new = value * offsb.tools.const.bohr2angstrom * simtk.unit.angstrom
            if modify:
                param.rmin_half = new

        elif ptype == AngleHandler.AngleType:
            current = param.angle
            new = value * simtk.unit.degree
            if modify:
                param.angle = new

        # lets not set the phases to the average angle
        # try to determine if need to set to 0 or 180 based on data
        elif ptype in [
            ImproperTorsionHandler.ImproperTorsionType,
            ProperTorsionHandler.ProperTorsionType,
        ]:
            pass
        #     current = param.phase
        #     n_params = len(param.phase)
        #     if issubclass(type(value), list):
        #         new = list(
        #             [x * simtk.unit.degree for x, _ in zip(value, range(n_params))]
        #         )
        #     else:
        #         new = list([value * simtk.unit.degree for _ in range(n_params)])
        #     if modify:
        #         param.phase = new
        else:
            raise NotImplementedError()
        if report_only:
            print(
                "Would change parameter in ",
                param_name,
                "from",
                current,
                "to",
                new,
                "(reporting only; did not change)",
            )
        else:
            print("Changed parameter in ", param_name, "from", current, "to", new)

    def _set_parameter_force(
        self, param_name, value, report_only=False, guess_periodicities=False
    ):

        param = self.db[param_name]["data"]["parameter"]

        current = None
        new = None

        ptype = type(param)

        modify = not report_only

        len_unit = (
            offsb.tools.const.hartree2kcalmol
            * simtk.unit.kilocalorie
            / simtk.unit.mole
            / (offsb.tools.const.bohr2angstrom * simtk.unit.angstrom) ** 2
        )
        rmin_half_unit = (
            offsb.tools.const.hartree2kcalmol * simtk.unit.kilocalorie / simtk.unit.mole
        )
        angle_unit = (
            offsb.tools.const.hartree2kcalmol
            * simtk.unit.kilocalorie
            / simtk.unit.mole
            / simtk.unit.radian ** 2
        )

        cosine_unit = (
            offsb.tools.const.hartree2kcalmol * simtk.unit.kilocalorie / simtk.unit.mole
        )

        if ptype == BondHandler.BondType:
            current = param.k
            new = value * len_unit
            if modify:
                param.k = new
        elif ptype == vdWHandler.vdWType:
            current = param.epsilon
            new = value * rmin_half_unit
            if modify:
                param.epsilon = new
        elif ptype == AngleHandler.AngleType:
            current = param.k
            new = value * angle_unit
            if modify:
                param.k = new
        elif ptype in [
            ImproperTorsionHandler.ImproperTorsionType,
            ProperTorsionHandler.ProperTorsionType,
        ]:
            current = param.k
            n_params = len(param.k)
            if issubclass(type(value), list):
                new = list(
                    [x / n_params * cosine_unit for x, _ in zip(value, range(n_params))]
                )
            else:
                new = list([value / n_params * cosine_unit for _ in range(n_params)])
            if modify:
                param.k = new
        else:
            raise NotImplementedError()

        if report_only:
            print(
                "Would change parameter in ",
                param_name,
                "from",
                current,
                "to",
                new,
                "(reporting only; did not change)",
            )
        else:
            print("Changed parameter in ", param_name, "from", current, "to", new)

    def initialize_parameters_from_data(
        self,
        QCA,
        ignore_parameters=None,
        only_parameters=None,
        report_only=False,
        periodicities=False,
        spatial=True,
        force=True,
    ):
        """
        get the ops from the data
        get the labels from the data
        combine
        use the qcasb module?!
        """
        print("\n\nInitializing parameters from data")

        if ignore_parameters is None:
            ignore_parameters = []

        ignore_parameters.extend(self._ignore_parameters)

        fn_map = {}

        if spatial:
            fn_map["measure"] = self._set_parameter_spatial

        if force:
            fn_map["force"] = self._set_parameter_force

        param_data = self._combine_reference_data(QCA=QCA, prims=False)

        stats_fn = self.stats_fn

        for param_name, param_types in param_data.items():
            if param_name in ignore_parameters or any(
                param_name[0] == x for x in ignore_parameters
            ):
                continue
            stats_fn.set_circular_from_label(param_name)
            if only_parameters is None or param_name in only_parameters:
                print(f"Modifying parameter {param_name}")
                param_types = list(param_types.values())
                for p_type, fn in fn_map.items():
                    new_vals = []
                    for param_dict in param_types:
                        for p, vals in param_dict.items():
                            if p == p_type:
                                new_vals.extend(vals)
                    # new_vals = param_data[param_name](p_type, None)
                    if periodicities and p_type == "measure":
                        self._guess_periodicities(
                            param_name, new_vals, report_only=report_only
                        )
                    if len(new_vals) > 0:
                        val = stats_fn.mean(new_vals)
                        fn(param_name, val, report_only=report_only)
                    else:
                        print(
                            "No values gathered for param",
                            param_name,
                            p_type,
                            "; cannot modify",
                        )

        return

        # qcasb = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA)

        # b = qcasb.measure_bonds("bonds")
        # a = qcasb.measure_angles("angles")
        # i = qcasb.measure_outofplanes("outofplanes")
        # t = qcasb.measure_dihedrals("dihedrals")

        # breakpoint()

        # vtable = {
        #     "Bonds": qcasb.measure_bonds,
        #     "Angles": qcasb.measure_angles,
        #     "ImproperTorsions": qcasb.measure_outofplanes,
        #     "ProperTorsions": qcasb.measure_dihedrals,
        # }

        # self.to_smirnoff_xml("tmp.offxml", verbose=False)
        # labeler = qcasb.assign_labels_from_openff("tmp.offxml", self.ffname)

        # for entry in QCA.iter_entry():
        #     print("ENTRY", entry)
        #     # need to unmap... sigh
        #     smi = QCA.db[entry.payload]["data"].attributes[
        #         "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        #     ]
        #     rdmol = offsb.rdutil.mol.build_from_smiles(smi)
        #     atom_map = offsb.rdutil.mol.atom_map(rdmol)
        #     map_inv = offsb.rdutil.mol.atom_map_invert(atom_map)
        #     labels = {
        #         aidx: val
        #         for ic_type in vtable
        #         for aidx, val in labeler.db[entry.payload]["data"][ic_type].items()
        #     }
        #     for ic_type, measure_function in vtable.items():

        #         # the geometry measurements
        #         ic = measure_function(ic_type)
        #         ic.source.source = QCA
        #         ic.apply()

        #         params = labeler.db["ROOT"]["data"][ic_type]

        #         for param_name, param in params.items():
        #             if param_name[0] in ignore_parameters:
        #                 continue
        #             if (
        #                 only_parameters is not None
        #                 and param_name not in only_parameters
        #             ):
        #                 continue
        #             param_vals = []
        #             for mol in QCA.node_iter_depth_first(entry, select="Molecule"):
        #                 ic_data = ic.db[mol.payload]
        #                 for aidx, vals in ic_data.items():
        #                     if labels[aidx] == param_name:
        #                         param_vals.extend(vals)
        #             new_val = np.mean(param_vals)
        #             self._set_parameter_spatial(param_name, new_val)

        #             param_vals = []
        #             for hessian_node in QCA.node_iter_depth_first(
        #                 entry, select="Hessian"
        #             ):
        #                 mol = QCA[hessian_node.parent]

        #                 with tempfile.NamedTemporaryFile(mode="wt") as f:
        #                     offsb.qcarchive.qcmol_to_xyz(
        #                         QCA.db[mol.payload]["data"], fnm=f.name
        #                     )
        #                     gmol = geometric.molecule.Molecule(f.name, ftype="xyz")
        #                 with open("out.xyz", mode="wt") as f:
        #                     offsb.qcarchive.qcmol_to_xyz(
        #                         QCA.db[mol.payload]["data"], fd=f
        #                     )

        #                 xyz = QCA.db[mol.payload]["data"].geometry
        #                 hess = QCA.db[hessian_node.payload]["data"].return_result
        #                 grad = np.array(
        #                     QCA.db[hessian_node.payload]["data"].extras["qcvars"][
        #                         "CURRENT GRADIENT"
        #                     ]
        #                 )

        #                 # IC = CoordClass(geometric_mol(mol_xyz_fname), build=True,
        #                 #                         connect=connect, addcart=addcart, constraints=Cons,
        #                 #                         cvals=CVals[0] if CVals is not None else None )
        #                 ic_prims = geometric.internal.PrimitiveInternalCoordinates(
        #                     gmol,
        #                     build=True,
        #                     connect=True,
        #                     addcart=False,
        #                     constraints=None,
        #                     cvals=None,
        #                 )

        #                 # for adding all of the ICs found by the labeler.

        #                 ic_hess = ic_prims.calcHess(xyz, grad, hess)

        #                 ic_data = {}
        #                 eigs = np.linalg.eigvalsh(ic_hess)
        #                 s = np.argsort(np.diag(ic_hess))
        #                 for aidx, val in zip([ic_prims.Internals[i] for i in s], eigs):
        #                     key = tuple(
        #                         map(
        #                             lambda x: map_inv[int(x) - 1],
        #                             str(aidx).split()[1].split("-"),
        #                         )
        #                     )

        #                     if ic_type == "ImproperTorsions" and len(key) == 4:
        #                         key = ImproperDict.key_transform(key)
        #                     else:
        #                         key = ValenceDict.key_transform(key)

        #                     ic_data[key] = val

        #                 # ic_data = ic.db[mol.payload]
        #                 # labels = {transform(tuple([atom_map[x]-1 for x in i])): v for i, v in labels.items()}
        #                 for aidx, vals in ic_data.items():
        #                     if aidx in labels and labels[aidx] == param_name:
        #                         # print("fc for key", aidx, "is", vals)
        #                         param_vals.append(vals)

        #             if len(param_vals) > 0:
        #                 new_val = np.mean(param_vals) * 0.80
        #                 # until we resolve generating an IC for each FF match,
        #                 # we skip setting params which the ICs don't make
        #                 # print("average val is", new_val)
        #                 self._set_parameter_force(param_name, new_val)
        # self.to_smirnoff_xml("tmp.offxml", verbose=False)

    def load_new_parameters(self, new_ff):

        if type(new_ff) == str:
            new_ff = ForceField(new_ff, allow_cosmetic_attributes=True)
        for ph in self.root().children:
            ph = self[ph]
            ff_ph = new_ff.get_parameter_handler(ph.payload)
            ff_params = {p.id: p for p in ff_ph._parameters}

            # directly grab the children, since we want to skip the top-level
            # parameter handler node (Bonds, Angles, etc.)
            params = [self[x] for x in ph.children]
            for param in self.node_iter_depth_first(params):
                try:
                    self.db[param.payload]["data"]["parameter"] = ff_params[
                        param.payload
                    ]
                except Exception as e:
                    breakpoint()
                    print("ERROR!", e)

    def _bump_zero_parameters(self, eps, names=None):

        ff = self.db["ROOT"]["data"]

        ph_nodes = [self[x] for x in self.root().children]

        ph_to_letter = {
            "vdW": "n",
            "Angles": "a",
            "Bonds": "b",
            "ProperTorsions": "t",
            "ImproperTorsions": "i",
        }
        for ph_node in ph_nodes:
            # if ph_node.payload != "Bonds":
            #     continue
            # print("Parsing", ph_node)
            ff_ph = ff.get_parameter_handler(ph_node.payload)
            params = []

            # ah, breadth first doesn't take the current level, depth does
            # so it is pulling the ph_nodes into the loop
            # but guess what! depth first puts it into the entirely incorrect order

            for i, param_node in enumerate(self.node_iter_dive(ph_node), 1):
                if param_node.payload not in self.db:
                    continue
                if param_node == ph_node:
                    continue
                param = copy.deepcopy(self.db[param_node.payload]["data"]["parameter"])

                ff_param = self.db[param_node.payload]["data"]["parameter"]

                for kval, v in ff_param.__dict__.items():
                    k = str(kval).lstrip("_")
                    if any([k.startswith(x) for x in ["cosmetic", "smirks"]]):
                        continue

                    if names is not None and k not in names:
                        continue

                    # list of Quantities...
                    if issubclass(type(v), list):
                        vals = v
                        for i, v in enumerate(vals):
                            if (
                                type(v) is simtk.unit.Quantity
                                and np.abs(v / v.unit) < eps
                            ):
                                print("bumped", k, param_node)
                                if v / v.unit == 0.0:
                                    vals[i] += (eps - v / v.unit) * v.unit
                                else:
                                    vals[i] += (
                                        np.sign(v / v.unit)
                                        * (eps - v / v.unit)
                                        * v.unit
                                    )
                        setattr(ff_param, kval, vals)
                    elif type(v) is simtk.unit.Quantity:
                        if np.abs(v / v.unit) < eps:
                            if v / v.unit == 0.0:
                                v += (eps - v / v.unit) * v.unit
                            else:
                                v += np.sign(v / v.unit) * (eps - v / v.unit) * v.unit
                            setattr(ff_param, kval, v)
                            print("bumped", k, param_node)

    def split_from_data(
        self,
        ignore_parameters=None,
        only_parameters=None,
        modify_parameters=False,
        join=True,
        join_protect_existing=True,
        maxiter=None,
    ):

        if ignore_parameters is None:
            ignore_parameters = []

        if only_parameters is None:
            only_parameters = []

        ignore_bits_ref = {}
        ignore_bits_optimized_ref = {}

        self.clear_caches()
        self._ignore_parameters = []

        if join and join_protect_existing and not self._ignore_parameters:

            print("\nProtecting existing parameters from deletion\n")
            params = iter(
                x.payload
                for y in self.root().children
                for x in self.node_iter_breadth_first(self[y])
            )

            params = filter(
                lambda x: x[0] in ignore_parameters
                or x in ignore_parameters
                or x not in only_parameters,
                params,
            )

            self._ignore_parameters.extend(params)

        ignore_bits_ref.update(
            ((x, None, "delete"), [None, None]) for x in self._ignore_parameters
        )

        # assume that if we want to split on these parameters, we don't want to delete them
        ignore_bits_ref.update(
            ((x, None, "delete"), [None, None]) for x in only_parameters
        )

        self._ignore_parameters.extend(filter(lambda x: len(x) == 1, ignore_parameters))

        ignore_bits_optimized_ref = ignore_bits_ref.copy()

        self._ignore_parameters = list(set(self._ignore_parameters))
        n_ignore = len(self._ignore_parameters)
        print(f"Ignoring {n_ignore} parameteres")

        print(self._ignore_parameters)

        print("Protecting from deletion:")
        print(ignore_bits_ref.keys())

        newff_name = "tmp.offxml"

        i = -1
        # self._to.apply()

        no_operation = False

        ops = ["new"]
        if join:
            ops.append("delete")

        strategies = ["spatial_reference", "force_reference"]
        strategies = ["spatial_reference"]

        if only_parameters is not None:
            only_parameters = []

        ignore_bits = ignore_bits_ref.copy()
        ignore_bits_optimized = ignore_bits_optimized_ref.copy()
        splits = 0
        print("Splitting parameters from data")
        moresteps = True if maxiter is None else i < (maxiter - 1)
        while moresteps:
            print("Splitting based on spatial difference")
            while no_operation is False:
                i += 1
                no_operation = True
                for opt_operation in ops:
                    for strategy in strategies:
                        print("\n####################################################")
                        print("# ----- Parameter operation:", opt_operation, strategy)
                        print("####################################################\n")
                        try:
                            (
                                node,
                                grad_split,
                                score,
                                operation,
                            ) = self._optimize_type_iteration(
                                optimize_during_typing=False,
                                ignore_bits=ignore_bits,
                                split_strategy=strategy,
                                use_gradients=False,
                                ignore_parameters=ignore_parameters,
                                only_parameters=only_parameters,
                                step=i,
                                operation=opt_operation,
                            )

                            if node is not None:
                                no_operation = False
                            else:
                                continue

                        except RuntimeError as e:
                            self.logger.error(str(e))
                            self.logger.error(
                                "Optimization failed; assuming bogus split"
                            )
                            # obj = np.inf
                            # grad_split = np.inf

                            # since we are keeping everything, use this to prevent adding
                            # the same node twice
                            # TODO: prevent adding same parameter twice
                            try:
                                parent = self[node.parent]
                            except KeyError as e:
                                # breakpoint()
                                pass
                            # bit = (
                            #     self.db[parent.payload]["data"]["group"]
                            #     - self.db[node.payload]["data"]["group"]
                            # )
                            bit = self.db[node.payload]["data"]["group"]

                            ignore_bits[(None, bit, opt_operation)] = [None, None]
                            # the db is in a null state, and causes failures
                            # TODO allow skipping an optimization
                            # reset the ignore bits since we found a good move

                        # ignore_bits_optimized = {}
                        # ignore_bits = {}
                        print("Keeping iteration", i)
                        print("Operation was", opt_operation, "on", node)
                        splits += 1
                        # ref = obj
                        # ref_grad = grad_split
                        # self._plot_gradients(fname_prefix=str(i) + ".accept")

                        self.to_smirnoff_xml("tmp.offxml", verbose=False)
                        self._labeler = offsb.ui.qcasb.QCArchiveSpellBook(
                            QCA=self._po.source.source
                        ).assign_labels_from_openff("tmp.offxml", "tmp.offxml")

                        # reset values of the new split
                        if operation == "new":
                            reinit = [node.payload, self[node.parent].payload]
                        elif operation == "delete":
                            reinit = [self[node.parent].payload]

                        del self._ref_data_cache
                        self._ref_data_cache = None
                        self._combine_reference_data()

                        print("modifying parameters", reinit)
                        self.initialize_parameters_from_data(
                            QCA=self._to.source.source,
                            only_parameters=reinit,
                            periodicities=True,
                            report_only=not modify_parameters,
                        )
                        # print("Assignments post operation:")
                        # self.print_label_assignments()

                        print("Hierarchy post operation:")
                        newff_name = "newFF" + str(i) + ".accept.offxml"
                        self.to_smirnoff_xml(newff_name, verbose=True, renumber=True)

            # print("Splitting based on force difference")
            # ignore_bits = {}
            while False:
                i += 1
                try:
                    node, grad_split, score, operation = self._optimize_type_iteration(
                        optimize_during_typing=False,
                        ignore_bits=ignore_bits,
                        split_strategy="force_reference",
                        use_gradients=False,
                        step=i,
                    )

                    if node is None:
                        print("No new parameter split, done!")
                        break

                except RuntimeError as e:
                    self.logger.error(str(e))
                    self.logger.error("Optimization failed; assuming bogus split")
                    # obj = np.inf
                    # grad_split = np.inf

                    # since we are keeping everything, use this to prevent adding
                    # the same node twice
                    # TODO: prevent adding same parameter twice
                    try:
                        parent = self[node.parent]
                    except KeyError as e:
                        # breakpoint()
                        pass
                    # bit = (
                    #     self.db[parent.payload]["data"]["group"]
                    #     - self.db[node.payload]["data"]["group"]
                    # )

                    bit = self.db[node.payload]["data"]["group"]

                    print("Ignoring ", bit)
                    ignore_bits[(None, bit, opt_operation)] = [None, None]
                    # the db is in a null state, and causes failures
                    # TODO allow skipping an optimization
                    # reset the ignore bits since we found a good move

                # ignore_bits_optimized = {}
                # ignore_bits = {}
                print("Keeping iteration", i)
                print("Operation {} on node".format(operation))
                print(node)
                splits += 1
                # ref = obj
                # ref_grad = grad_split
                # self._plot_gradients(fname_prefix=str(i) + ".accept")

                del self._param_data_cache
                self._param_data_cache = None

                self.labels(force=True)
                # self.to_smirnoff_xml("tmp.offxml", verbose=False)
                # self._labeler = offsb.ui.qcasb.QCArchiveSpellBook(
                #     QCA=self._po.source.source
                # ).assign_labels_from_openff("tmp.offxml", "tmp.offxml")

                self._combine_reference_data()
                # reset values of the new split
                reinit = [node.payload, self[node.parent].payload]
                print("modifiying parameters", reinit)
                self.initialize_parameters_from_data(
                    QCA=self._to.source.source,
                    only_parameters=reinit,
                    report_only=not modify_parameters,
                )

                newff_name = "newFF" + str(i) + ".accept.offxml"
                self.to_smirnoff_xml(newff_name, verbose=True)
                # reset values of the new split
            break

        # print("Final objective is", obj, "initial was", initial)
        # print("Total drop is", obj - initial)
        # self.print_label_assignments()
        self.to_smirnoff_xml(newff_name, verbose=True)

    def _optimize_microstep(self):
        """
        generate a possible manipulation

        this should return a generator?

        produces a manipulation as a partial or an obj with callable
        """

    def _optimize_step(self):
        """
        perform a single manipulation to the FF hierarchy which improves
        the agreement with the reference data

        this can be a split, combine, or swap
        this can be evaluated using FB, which uses either the objective or its
        following up with its gradient

        this should apply the step
            another subroutine should produce the candiates
            this function chooses the one to apply
        """

        ignore_bits = {}
        ignore_bits_optimized = {}

        # this is give me the actual parameter as a node, or a combine?
        # node should be a tree manipulation
        # returns a partial with a parent, node, aux_node which can either
        # add node to parent, remove node from parent, or swap node and aux
        # under parent

        # self._optimize_type_iteration(
        #     ignore_bits=ignore_bits,
        #     split_strategy="gradient",
        #     use_gradients=True,
        # )
        # # node, grad_split = self._optimize_type_iteration(
        # #     ignore_bits=ignore_bits,
        # #     split_strategy="gradient",
        # #     use_gradients=True,
        # # )

        # if node is None:
        #     print("No new parameter split, done!")
        #     break

        # # evaluate the change to the FF
        # jobtype = "OPTIMIZE" if optimize_during_typing else "GRADIENT"
        # self._po.apply(jobtype=jobtype)
        # new_obj, new_grad = self._po.X, self._po.G

        # # decide to keep the new node
        # grad_scale_factor = 1.0
        # if (obj > ref and optimize_during_typing) or (
        #     grad_split > ref_grad * grad_scale_factor
        #     and not optimize_during_typing
        # ):
        #     # reject steps where the objective goes up

        #     self._po.new_ff = current_ff

        #     self.to_smirnoff_xml(newff_name, verbose=False)
        #     self._po._setup.ff_fname = newff_name
        #     self._po.ff_fname = newff_name
        #     self._po._init = False
        #     print(
        #         "Rejecting iteration", i, "objective reference still", ref
        #     )
        #     # self._plot_gradients(fname_prefix=str(i) + ".reject")
        #     # self.to_smirnoff_xml("newFF" + str(i) + ".reject.offxml")
        #     parent = self[node.parent]
        #     bit = (
        #         self.db[parent.payload]["data"]["group"]
        #         - self.db[node.payload]["data"]["group"]
        #     )
        #     try:
        #         # patch; assume everything is correct and we can
        #         # just add symmetric cases without issue
        #         if bit not in ignore_bits:
        #             ignore_bits[bit] = []
        #         ignore_bits_optimized[bit] = ignore_bits[bit]
        #     except KeyError as e:
        #         # probably a torsion or something that caused
        #         # the bit calculation to fail
        #         print(e)

        #     print("Keeping ignore bits for next iteration:")
        #     for bit, bit_grad in ignore_bits_optimized.items():
        #         print(bit_grad, bit)

        #     # ignore_bits will keep track of things that didn't work
        #     # do this after keeping the ignore bits since we access
        #     # the info
        #     self[node.parent].children.remove(node.index)
        #     self.node_index.pop(node.index)
        #     self.db.pop(node.payload)

        #     # reset the physical terms after resetting the tree
        #     self.load_new_parameters(current_ff)

        #     if optimize_during_typing:
        #         ignore_bits = ignore_bits_optimized.copy()
        # else:
        #     # reset the ignore bits since we found a good move

        #     ignore_bits_optimized = {}
        #     ignore_bits = {}
        #     print("Keeping iteration", i, "objective:", obj)
        #     print("Split kept is")
        #     print(node)
        #     ref = obj
        #     ref_grad = grad_split
        #     # self._plot_gradients(fname_prefix=str(i) + ".accept")
        #     self.to_smirnoff_xml("newFF" + str(i) + ".accept.offxml")
        #     self.to_pickle()

        # return report

    class OptimizationStepReport:

        __slots__ = ["success", "X", "G", "H", "log_str"]

        def __init__(self):
            self.success: bool = None
            self.X: float = None
            self.G: float = None
            self.H: float = None
            self.log_str: str = None

        def print_result(self):
            print(self.X, self.G, self.H)
            print(self.log_str)

    def optimize(
        self,
        optimize_types=True,
        optimize_parameters=False,
        optimize_during_typing=True,
        optimize_during_scoring=False,
        optimize_initial=True,
        join=True,
        join_protect_existing=True,
        maxiter=None,
    ):

        # self._to.apply()

        self.clear_caches()
        self._ignore_parameters = []

        newff_name = "tmp.offxml"
        self.to_smirnoff_xml(newff_name, verbose=False)
        self._po._setup.ff_fname = newff_name
        self._po.ff_fname = newff_name
        self._po._init = False

        # if self.trust0 is None:
        #     self.trust0 = 0.25
        # self.finite_difference_h = None

        jobtype = "OPTIMIZE"
        if not optimize_initial:
            jobtype = "GRADIENT"

        initial = None
        # fill in the parameters with bits so we don't we waste time splitting
        # terms

        # how to do this?
        # we turn off every bit possible that keeps the same coverage. This is
        # already done for non-leaf parameters
        # we go to each leaf, then split all the way to primitives?
        #

        print("Performing initial objective calculation...")
        success = self._run_optimizer(jobtype, keep_state=False)
        if not success:
            print("Failed. Cannot proceed.")
            return
        # self.finite_difference_h = self._po._options["finite_difference_h"]

        print("Initial objective :", self._po.X)
        obj = self._po.X
        ref = obj
        initial = obj
        ref_grad = self._po.G

        rejects = 0

        self._labeler = None
        newff_name = "tmp.offxml"
        self.load_new_parameters(self._po.new_ff)
        self.to_smirnoff_xml(newff_name, verbose=False)
        self._po._setup.ff_fname = newff_name
        self._po._init = False

        ignore_bits_ref = {}
        ignore_bits_optimized_ref = {}

        if join and join_protect_existing and not self._ignore_parameters:

            print("\nProtecting existing parameters from deletion\n")
            for ph in self.root().children:
                ph = self[ph]
                for param in self.node_iter_depth_first(ph):
                    ignore_bits_ref[(param.payload, None, "delete")] = [None, None]
                    ignore_bits_optimized_ref[(param.payload, None, "delete")] = [
                        None,
                        None,
                    ]
                    self._ignore_parameters.append(param.payload)

        n_ignore = len(self._ignore_parameters)
        print(f"Ignoring {n_ignore} parameters")
        # print("Protecting from deletion:")
        # print(ignore_bits_ref.keys())

        ignore_bits = ignore_bits_ref.copy()
        ignore_bits_optimized = ignore_bits_optimized_ref.copy()

        # self._plot_gradients(fname_prefix="0")

        ops = ["new"]
        if join:
            ops.append("delete")

        current_ff = copy.deepcopy(self._po.new_ff)
        podb = copy.deepcopy(self._po.db)
        current_db = self.db.copy()
        poff = copy.deepcopy(self._po._forcefield)

        no_operation = False
        if optimize_types:
            i = 0
            moresteps = True if maxiter is None else i < (maxiter)
            if False:
                pass
            else:
                while no_operation is False and moresteps:

                    if self.reject_limit and self.reject_limit == rejects:
                        print(
                            f"\nMaximum number of rejects reached ({rejects}). Optimization complete.\n"
                        )
                        break

                    # while True:
                    i += 1
                    # This objective should be exactly the same as the fit obj,
                    # but no regularization, therefore we ignore it mostly
                    no_operation = True
                    for opt_operation in ops:
                        print("\n####################################################")
                        print(
                            "# ----- {} Parameter operation: {}".format(
                                i, opt_operation
                            )
                        )
                        print("####################################################\n")

                        self._param_data_cache = None
                        self._ref_data_cache = None
                        try:
                            (
                                node,
                                grad_split,
                                score,
                                operation,
                            ) = self._optimize_type_iteration(
                                optimize_during_typing=optimize_during_typing,
                                optimize_during_scoring=optimize_during_scoring,
                                ignore_bits=ignore_bits,
                                split_strategy="gradient",
                                use_gradients=True,
                                step=i,
                                operation=opt_operation,
                            )

                            if node is not None:
                                no_operation = False
                                rejects = 0
                            if node is None:
                                continue

                            obj = score

                        except RuntimeError as e:
                            self.logger.error(str(e))
                            self.logger.error(
                                "Optimization failed; assuming bogus operation"
                            )
                            obj = np.inf
                            grad_split = np.inf
                            try:
                                parent = self[node.parent]
                            except KeyError as e:
                                # breakpoint()
                                pass
                            # bit = (
                            #     self.db[parent.payload]["data"]["group"]
                            #     - self.db[node.payload]["data"]["group"]
                            # )
                            bit = self.db[node.payload]["data"]["group"]

                            ignore_bits[(None, bit, opt_operation)] = [None, None]
                            # the db is in a null state, and causes failures
                            # TODO allow skipping an optimization

                        print(
                            "New objective:",
                            obj,
                            "Delta is",
                            obj - ref,
                            "Reference is",
                            ref,
                            "({:8.2f}%)".format(100.0 * ((obj - ref) / ref)),
                        )
                        print(
                            "New objective gradient from split:",
                            grad_split,
                            "Previous",
                            ref_grad,
                            "Delta is",
                            grad_split - ref_grad,
                            "({:8.2f}%)".format(
                                100.0 * ((grad_split - ref_grad) / ref_grad)
                            ),
                        )
                        print("Parameter score is ", score)
                        grad_scale_factor = 1.0

                        if (obj > ref and optimize_during_typing) or (
                            grad_split < ref_grad * grad_scale_factor
                            and not optimize_during_typing
                        ):
                            # reject steps where the objective goes up

                            self._po.new_ff = current_ff
                            self._po._forcefield = current_ff
                            self._po.db = podb

                            self.to_smirnoff_xml(newff_name, verbose=False)
                            self._po._setup.ff_fname = newff_name
                            self._po.ff_fname = newff_name
                            self._po._init = False
                            rejects += 1
                            print(
                                "Rejecting iteration",
                                i,
                                "for operation",
                                opt_operation,
                                "objective reference still",
                                ref,
                            )
                            print(f"Consecutive rejects: {rejects}/{self.reject_limit}")
                            # self._plot_gradients(fname_prefix=str(i) + ".reject")
                            # self.to_smirnoff_xml("newFF" + str(i) + ".reject.offxml")
                            parent = self[node.parent]
                            # bit = (
                            #     self.db[parent.payload]["data"]["group"]
                            #     - self.db[node.payload]["data"]["group"]
                            # )
                            bit = self.db[node.payload]["data"]["group"]

                            try:
                                ignore_bits[(None, bit, opt_operation)] = [None, None]

                                bits = [
                                    x
                                    for x in ignore_bits
                                    if x[1] == bit and x[2] == opt_operation
                                ]
                                for bit_key in bits:
                                    key = (bit_key[0], bit_key[1], opt_operation)
                                    ignore_bits_optimized[key] = ignore_bits[
                                        bit_key
                                    ].copy()
                            except KeyError as e:
                                # probably a torsion or something that caused
                                # the bit calculation to fail
                                breakpoint()
                                print("Key error during ignore add:")
                                print(e)
                                ignore_bits_optimized[(None, bit, opt_operation)] = [
                                    None,
                                    None,
                                ]

                            print("Keeping ignore bits for next iteration:")
                            for (
                                lbl,
                                bit,
                                operation,
                            ), bit_grad in ignore_bits_optimized.items():
                                print(lbl, bit_grad, operation, bit.compact_str())

                            # ignore_bits will keep track of things that didn't work
                            # do this after keeping the ignore bits since we access
                            # the info
                            if opt_operation == "new":
                                self[node.parent].children.remove(node.index)
                                self.node_index.pop(node.index)

                                # self.db.pop(node.payload)

                                self.db = current_db
                                self._po._forcefield = poff

                            # reset the physical terms after resetting the tree
                            # self.load_new_parameters(current_ff)

                            if optimize_during_typing:
                                ignore_bits = ignore_bits_optimized.copy()

                        else:
                            # reset the ignore bits since we found a good move
                            # assumes that the operation was already applied

                            ignore_bits_optimized = ignore_bits_optimized_ref.copy()
                            ignore_bits = ignore_bits_ref.copy()
                            print("Keeping iteration", i, "objective:", obj)
                            print("Operation {} on node".format(opt_operation))
                            print(node)

                            ref = obj
                            ref_grad = grad_split
                            # self._plot_gradients(fname_prefix=str(i) + ".accept")
                            newff_name = "newFF" + str(i) + ".accept.offxml"
                            self.to_smirnoff_xml(newff_name, renumber=True, verbose=False)

                            current_ff = copy.deepcopy(self._po.new_ff)
                            podb = copy.deepcopy(self._po.db)
                            current_db = self.db.copy()
                            poff = copy.deepcopy(self._po._forcefield)

                        self.labels(force=True)
                        # self.to_smirnoff_xml("tmp.offxml", verbose=False)
                        # self._labeler = offsb.ui.qcasb.QCArchiveSpellBook(
                        #     QCA=self._po.source.source
                        # ).assign_labels_from_openff("tmp.offxml", "tmp.offxml")

            self.to_smirnoff_xml(newff_name, verbose=True)
            self.print_label_assignments()
            print("Final objective is", obj, "initial was", initial)
            print(
                "Total drop is",
                obj - initial,
                "{:8.2f}%".format(100.0 * (obj - initial) / initial),
            )
            print("Splitting done")

            # while True:
            #     try:
            #         self._po.load_options(
            #             options_override={
            #                 "trust0": self.trust0,
            #                 "finite_difference_h": self.finite_difference_h,
            #             }
            #         )
            #         self._po.apply(jobtype="OPTIMIZE")
            #         break
            #     except RuntimeError:
            #         self._bump_zero_parameters(1e-1, names="epsilon")
            #         self.to_smirnoff_xml(newff_name, verbose=False)
            #         self._po._setup.ff_fname = newff_name
            #         self._po.ff_fname = newff_name
            #         self._po._init = False
            #         self.trust0 = self._po._options["trust0"] / 2.0
            #         print(
            #             "Initial optimization failed; reducing trust radius to",
            #             self.trust0,
            #         )
            #         self._po._options["trust0"] = self.trust0
            #         mintrust = self._po._options["mintrust"]
            #         if self.trust0 < mintrust:
            #             print(
            #                 "Trust radius below minimum trust of {}; cannot proceed.".format(
            #                     mintrust
            #                 )
            #             )
            #             return

        # since we presumably just finished an optimization from a split, I think
        # we can skip the final optimization
        if optimize_parameters and not (optimize_types and optimize_during_typing):
            if not optimize_parameters:
                while True:
                    try:
                        self._po.apply(jobtype="GRADIENT")
                        initial = self._po.X
                        self.trust0 = self._po._options["trust0"]
                        break
                    except RuntimeError as e:
                        self.trust0 = self._po._options["trust0"] / 2.0
                        print(
                            "Initial gradient failed; reducing trust radius to",
                            self.trust0,
                        )
                        self._po._options["trust0"] = self.trust0
                        mintrust = self._po._options["mintrust"]
                        if self.trust0 < mintrust:
                            print(
                                "Trust radius below minimum trust of {}; cannot proceed.".format(
                                    mintrust
                                )
                            )
                            raise

            else:
                print("Performing final optimization")
                while True:
                    try:
                        self._po.apply(jobtype="OPTIMIZE")
                        obj = self._po.X
                        self.trust0 = self._po._options["trust0"]
                        break
                    except RuntimeError as e:
                        self.trust0 = self._po._options["trust0"] / 2.0
                        print(
                            "Optimization failed; reducing trust radius to", self.trust0
                        )
                        self._po._options["trust0"] = self.trust0
                        mintrust = self._po._options["mintrust"]
                        if self.trust0 < mintrust:
                            print(
                                "Trust radius below minimum trust of {}; cannot proceed.".format(
                                    mintrust
                                )
                            )
                            raise
            # Some target failed... just use the current best

            # FB likes to change directories
            # if we fail, it appears we are in some iter_xxxx directory,
            # and there should be an offxml there... keep it
            new_ff = self._po._setup.prefix + ".offxml"
            # new_ff_path = os.path.join("result", self._po._setup.prefix, new_ff)
            if os.path.exists(new_ff):
                self._po.new_ff = ForceField(new_ff, allow_cosmetic_attributes=True)
            else:
                print("Could not find FF from best iteration!")

            self.load_new_parameters(self._po.new_ff)
            newff_name = "tmp.offxml"
            # self._plot_gradients(fname_prefix="optimized.final")
            self.to_smirnoff_xml(newff_name, verbose=True, renumber=False)
            if initial is None:
                print("Optimized objective is", obj)
            else:
                print("Optimized objective is", obj, "initial was", initial)
                print(
                    "Total drop is",
                    obj - initial,
                    "{:8.2f}%".format(100.0 * (obj - initial) / initial),
                )

    @classmethod
    def from_smirnoff(self, input):
        pass

    def set_physical_optimizer(self, obj):
        self._po = obj

        #
        # need the optimizer that should just try until it works
        #
        self._po.raise_on_error = False

    def clear_caches(self):

        # clear the caches
        self._ic = None
        self._fc = None
        self._prim = None
        self._ic_prim = None

        self._ref_data_cache = None
        self._param_data_cache = None

    def set_smarts_generator(self, obj):
        self._to = obj
        self._to.chembit = False
        # self._to.processes = 1
        # self._to.apply()
        self._to.apply()

        self.clear_caches()
