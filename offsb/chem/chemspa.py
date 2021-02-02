#!/usr/bin/env python

import copy
import functools
import io
import itertools
import logging
import os
import pprint
import re
import sys
import tempfile

import numpy as np
import simtk.unit
import tqdm
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.typing.engines.smirnoff.parameters import (
    AngleHandler, BondHandler, ImproperDict, ImproperTorsionHandler,
    ParameterList, ProperTorsionHandler, ValenceDict, vdWHandler)

import offsb.chem.types
import offsb.op.chemper
import offsb.op.forcebalance
import offsb.op.internal_coordinates
import offsb.tools.const
import offsb.treedi.node
import offsb.treedi.tree
import offsb.ui.qcasb
from offsb.treedi.tree import DEFAULT_DB

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


class ChemicalSpace(offsb.treedi.tree.Tree):
    def __init__(
        self,
        ff_fname,
        obj,
        root_payload=None,
        node_index=None,
        db=None,
        payload=None,
    ):
        print("Building ChemicalSpace")
        if isinstance(obj, str):
            super().__init__(
                obj,
                root_payload=root_payload,
                node_index=node_index,
                db=db,
                payload=payload,
            )

            self.ffname = ff_fname

            self._prim_clusters = dict()

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

            self.trust0 = None
            self.eig_lowerbound = None
            self.finite_difference_h = None

    def to_pickle(self, db=True, index=True, name=None):
        po, self._po = self._po, None
        to, self._to = self._to, None
        super().to_pickle(db=db, index=index, name=name)
        self._po = po
        self._to = to

    def to_pickle_str(self):
        pass

    def to_smirnoff(self, verbose=True, renumber=False):

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
                param = copy.deepcopy(self.db[param_node.payload]["data"]["parameter"])

                if visible_node_depth(param_node) > 1:
                    print("->", end="")
                print(
                    "    " * visible_node_depth(param_node)
                    + "{:8d}".format(self.node_depth(param_node))
                    + "{}".format(param_node)
                )
                ff_param = copy.deepcopy(
                    self.db[param_node.payload]["data"]["parameter"]
                )
                if renumber:
                    ff_param.id = num_map[param.id]
                ff_param.smirks = ""
                print(
                    "  " + "      " * visible_node_depth(param_node),
                    self.db[param_node.payload]["data"]["group"],
                )
                smarts = self.db[param_node.payload]["data"]["group"].to_smarts()
                print(
                    "  " + "      " * visible_node_depth(param_node),
                    "{:12s} : {}".format("smarts", smarts),
                )
                print(
                    "  " + "      " * visible_node_depth(param_node),
                    "{:12s} : {}".format("id", ff_param.id),
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

    def to_smirnoff_xml(self, output, verbose=True, renumber=False):

        ff = self.to_smirnoff(verbose=verbose, renumber=renumber)
        if output:
            ff.to_file(output)

    @classmethod
    def from_smirnoff_xml(cls, input, name=None, add_generics=False):

        """
        add parameter ids to db
        each db entry has a ff param (smirks ignored) and a group
        """

        ff = ForceField(input, allow_cosmetic_attributes=True)
        cls = cls(input, name, root_payload=ff)

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

        cls.to_smirnoff_xml("ohmy.offxml")
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

    def split_parameter(self, label, bit):

        node = list(
            [x for x in self.node_iter_depth_first(self.root()) if x.payload == label]
        )[0]
        data = self.db[node.payload]["data"]
        group = data["group"]
        param = data["parameter"]
        child = group - bit

        if child == group:
            return None
        if not child.is_valid():
            return None

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
        param.smirks = group.to_smarts(tag=True)
        try:
            new_param.smirks = child.to_smarts(tag=True)
        except Exception as e:
            breakpoint()
        new_param.id = param_name

        self.db[param_name] = DEFAULT_DB(
            {"data": {"parameter": new_param, "group": child}}
        )
        return pnode

    def _scan_param_with_bit(
        self,
        param_data,
        lbl,
        group,
        bit,
        key=None,
        eps=1.0,
        mode="sum_difference",
        bit_gradients=None,
        ignore_bits=None,
    ):

        ys_bit = []
        no_bit = []

        denoms = {
            "n": VDW_DENOM,
            "b": self._po._setup._bond_denom / offsb.tools.const.bohr2angstrom,
            "a": self._po._setup._angle_denom,
            "i": self._po._setup._improper_denom,
            "t": self._po._setup._dihedral_denom,
        }

        denoms = {
            "n": VDW_DENOM,
            "b": 0.1 * offsb.tools.const.angstrom2bohr,
            "a": 10,
            "i": 10,
            "t": 10,
        }

        verbose = False

        if type(mode) is str:
            mode = [mode]

        if bit_gradients is None:
            bit_gradients = []

        if ignore_bits is None:
            ignore_bits = {}

        groups = self._prim_clusters.get(lbl, None)

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
            self._prim_clusters[lbl] = groups

        chosen_split = [[], []]
        for i, (prim, dat) in enumerate(param_data[lbl].items()):
            if key is not None:
                dat = dat[key]
            # pdb.set_trace()
            if verbose:
                try:
                    smarts = prim.to_smarts()
                    print("Considering prim", prim, "with smarts", smarts, end=" ")
                except Exception as e:
                    breakpoint()
            # if bit in prim:
            if prim not in (group - bit):
                if verbose:
                    print(
                        "same (N=",
                        len(dat),
                        ") mean: {} var: {} sum: {}".format(
                            np.mean(dat, axis=0),
                            np.var(dat, axis=0),
                            np.sum(dat, axis=0),
                        ),
                    )
                chosen_split[0].append(i)
                ys_bit.extend(dat)
            else:
                if verbose:
                    print(
                        "change (N=",
                        len(dat),
                        ") mean: {} var: {} sum: {}".format(
                            np.mean(dat, axis=0),
                            np.var(dat, axis=0),
                            np.sum(dat, axis=0),
                        ),
                    )
                chosen_split[1].append(i)
                no_bit.extend(dat)
        if verbose:
            print("    Parent group (N=", len(ys_bit), ") :", end=" ")
        if len(ys_bit) > 0:
            pass
            if verbose:
                print(
                    "mean:",
                    np.mean(ys_bit, axis=0),
                    "var:",
                    np.var(ys_bit, axis=0),
                    "sum:",
                    np.sum(ys_bit, axis=0),
                )
        else:
            pass
            if verbose:
                print("None")
        # print(ys_bit)
        if verbose:
            print("    New group (N=", len(no_bit), ") :", end=" ")
        if len(no_bit) > 0 and len(ys_bit) > 0:
            if verbose:
                print(
                    "mean:",
                    np.mean(no_bit, axis=0),
                    "var:",
                    np.var(no_bit, axis=0),
                    "sum:",
                    np.sum(no_bit, axis=0),
                )

            lhs = no_bit
            # lhs = np.sum(lhs, axis=0)
            rhs = ys_bit
            # rhs = np.sum(rhs, axis=0)

            vals = {}
            for mode_i in mode:
                val = 0.0
                if mode_i == "sum_difference":
                    val = np.sum(rhs, axis=0) - np.sum(lhs, axis=0)
                if mode_i == "mag_difference":
                    val = np.linalg.norm(np.sum(rhs, axis=0) - np.sum(lhs, axis=0))
                elif mode_i == "mean_difference":
                    if lbl[0] in "ait":
                        lhs = list(map(lambda x: x + 180 if x < 180 else x, lhs))
                        rhs = list(map(lambda x: x + 180 if x < 180 else x, rhs))
                    val = np.mean(rhs, axis=0) - np.mean(lhs, axis=0)
                if key == "measure":
                    if denoms[lbl[0]] == 0.0:
                        val = 0.0
                    else:
                        val = val / denoms[lbl[0]]
                vals[mode_i] = val

            # default is to score by the first mode supplied
            val = vals[mode[0]]
            success = np.abs(val) > eps

            try:
                success = bool(success)
            except ValueError:
                # we could use all or any here. Using any allows, for a bond for
                # example, either the length or the force to accepted. If we use
                # all, both terms must be above eps. This is only important for
                # sum_difference, which could provide multiple values.
                success = any(success)
            if verbose:
                print("    Delta: {} eps: {} split? {}".format(val, eps, success))
            if success:
                if verbose:
                    print("Sorting groups...")
                rank = None
                groups = self._prim_clusters
                if len(groups):
                    group_a = sorted(chosen_split[0])
                    group_b = sorted(chosen_split[1])
                    chosen_split = sorted(map(tuple, [group_a, group_b]))
                    chosen_split = tuple(chosen_split)
                    groups_sorted = sorted(
                        groups,
                        key=lambda x: np.max(np.abs(x[2][mode[0]])),
                        reverse=True,
                    )
                    rank_pos = [
                        i
                        for i, v in enumerate(groups_sorted)
                        if v[2][mode[0]] == vals[mode[0]]
                    ][0]
                    score_0_to_1 = (
                        (vals[mode[0]] - groups_sorted[-1][2][mode[0]])
                        / (groups_sorted[0][2][mode[0]] - groups_sorted[-1][2][mode[0]])
                        if (
                            groups_sorted[0][2][mode[0]] - groups_sorted[-1][2][mode[0]]
                        )
                        else 1.0
                    )
                    rank = {
                        "rank": rank_pos + 1,
                        "rank_of": len(groups),
                        "rank_1": groups_sorted[0][2][mode[0]],
                        "score_0_to_1": score_0_to_1,
                    }

                new_val = [lbl, bit.copy(), vals, chosen_split, rank]
                duplicate = False
                for i, x in enumerate(bit_gradients):
                    if (
                        x[0] == lbl
                        # and (any([x[1] == y for y in bit]) or bit not in x[1])
                        and chosen_split == x[3]
                    ):
                        if bit.bits() < x[1].bits():
                            if verbose:
                                print(
                                    "Swapping existing result (bits=",
                                    x[1].bits(),
                                    ")",
                                    x,
                                    "with (bits=",
                                    bit.bits(),
                                    ")",
                                    new_val,
                                )
                            bit_gradients[i] = new_val

                            if verbose:
                                print("Adding ", lbl, bit, "to ignore")
                            if (lbl, bit) in ignore_bits:
                                ignore_bits[(lbl, bit)].append((vals, None))
                            else:
                                ignore_bits[(lbl, bit)] = [(vals, None)]

                            if (lbl, x[0]) in ignore_bits:
                                ignore_bits[(x[0], x[1])].append((x[2], None))
                            else:
                                ignore_bits[(x[0], x[1])] = [(x[2], None)]

                        duplicate = True
                        break

                if not duplicate:
                    if verbose:
                        print("Appending result (bits=", bit.bits(), ")", new_val)
                    bit_gradients.append(new_val)
                else:
                    if verbose:
                        print(
                            "Not appending result since a lower bit split produces the same result"
                        )
                        print("Adding ", lbl, bit, "to ignore")
                    if (lbl, bit) in ignore_bits:
                        ignore_bits[(lbl, bit)].append((vals, None))
                    else:
                        ignore_bits[(lbl, bit)] = [(vals, None)]
                if verbose:
                    print("\n")
        else:
            if verbose:
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

        verbose = False

        if mode is None:
            mode = ["sum_difference"]

        if bit_gradients is None:
            bit_gradients = []
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

        labeler = self._labeler

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

        for node in tqdm.tqdm(nodes, total=len(nodes), desc="bit scanning", ncols=80):
            lbl = node.payload

            if lbl not in labels:
                continue
            # if lbl[0] != 't':
            #     continue

            # This check essentially means this param was not applied
            # Most likely for outofplanes
            if lbl not in param_data:
                continue

            if lbl in bit_cache:
                continue

            print("\nConsidering for bit scans", node)

            ff_param = self.db[lbl]["data"]["group"]
            # print("FF param:", ff_param)

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

            try:
                # Checking the sum does not work for torsions, so check each individually
                # if group not in self.db[lbl]["data"]["group"]:
                if any(
                    [x not in self.db[lbl]["data"]["group"] for x in param_data[lbl]]
                ):
                    print("ERROR: data is not covered by param!")
                    print("Group is", group)
                    print("FF Group is", self.db[lbl]["data"]["group"])
                    print("marginal is ", group - self.db[lbl]["data"]["group"])
                    # breakpoint()
            except Exception as e:
                print(e)
                # breakpoint()

            # Now that we know all params are covered by this FF param,
            # sanitize the group by making it match the FF group, so things
            # stay tidy
            # group = group & self.db[lbl]["data"]["group"]

            # assert (group - self.db[lbl]['data']['group']).reduce() == 0

            # this indicates what this smirks covers, but we don't represent in the
            # current data
            try:
                uncovered = self.db[lbl]["data"]["group"] - group
            except Exception as e:
                pass
                # breakpoint()

            if verbose:
                if uncovered.reduce() == 0:
                    print("This dataset completely covers", lbl, ". Nice!")
                else:
                    print("This dataset does not cover this information:")
                    print(uncovered)

            param_group = self.db[lbl]["data"]["group"]

            # iterate bits that we cover (AND them just to be careful)
            # group = group & self.db[lbl]["data"]["group"]
            if verbose:
                print("\nContinuing with param ", lbl, "this information:")
                for data in param_data[lbl]:
                    print(data)
                print(group)
                print("\nFF param for ", lbl, "is:")
                print(param_group)

            n_bits[lbl] = sum([1 for x in group])
            bit_visited = {}

            # manipulations = set([bit for bit in param_group if bit in group])
            manipulations = set([bit for bit in group])

            todo = len(manipulations) + 1
            completed = 0
            hits = 0

            self._prim_clusters.clear()
            maxbits = 1
            total = len(manipulations)
            pbar = tqdm.tqdm(ncols=80, total=total, desc="bit splitting")
            while len(manipulations) > 0:
                bit = manipulations.pop()
                todo -= 1
                if verbose:
                    print(
                        "{:8d}/{:8d}".format(todo, completed),
                        "Scanning for bit ({:3d})".format(bit.bits()),
                        bit,
                    )

                # if lbl == 't3' and todo == 15 and completed == 2:
                #     breakpoint()
                completed += 1
                pbar.update(completed)

                # Already visited; skip
                if any([x == bit for x in bit_visited]):
                    continue

                if any([x[0] == lbl and x[1] == bit for x in ignore_bits]):
                    if verbose:
                        print("Ignoring since it is in the ignore list")
                    continue

                # we shouldn't try to split if it makes the new parameter invalid
                # aa = param_group & bit
                # ab = param_group - bit
                # if lbl == 'b83':
                #     breakpoint()

                # here we want to make sure that the bit is anchored to the FF
                # param, and that removing it will still produce a valid param
                # no reason to continue if our split will produce a dud

                if (param_group & bit) != bit or not (param_group - bit).is_valid():
                    bit_visited[bit] = None
                    if verbose:
                        print("This bit is required, cannot split")
                    continue
                # elif verbose:
                #     print("This bit is valid:", param_group - bit)

                # an important check is if any of the prims would actually move
                # over to the new param; skip if not since this would produce
                # a useless parameter
                in_new = [x in (param_group - bit) for x in param_data[lbl]]
                occluding_split = all(in_new)
                valid_split = occluding_split or any(in_new)

                if not valid_split:
                    bit_visited[bit] = None
                    if verbose:
                        print("This bit would not separate the data; refusing to use")
                    continue

                # if the bit is still there, then it is there by symmetry
                # this means we will be double counting, since we will try the
                # bit twice
                # if bit in (param_group - bit) or occluding_split:
                if occluding_split:
                    bit_visited[bit] = None
                    # Print this out to give an idea
                    print("This bit occludes the parent parameter", end=" ")
                    if (
                        self.bit_search_limit is None
                        or bit.bits() < self.bit_search_limit
                    ):
                        new_manips = list([bit + x for x in group])
                        n_manips = len(manipulations)
                        manipulations = manipulations.union(new_manips)
                        todo += len(manipulations) - n_manips
                        total += len(manipulations) - n_manips
                        pbar.total = total
                        print(
                            "adding another bit (",
                            bit.bits() + 1,
                            "), producing {:d} more splits to check (total= {:d})".format(
                                len(manipulations), total
                            ),
                        )
                    elif verbose:
                        print()
                    continue

                # for ignore in ignore_bits:
                #     if type(ignore) == type(bit) and :
                #         if verbose:
                #             print("Ignoring since it is in the ignore list. Matches this ignore:")
                #             print(ignore)
                #         continue
                hits += 1
                maxbits = max(maxbits, bit.bits())

                self._scan_param_with_bit(
                    param_data,
                    lbl,
                    param_group,
                    bit,
                    key=key,
                    mode=mode,
                    eps=eps,
                    bit_gradients=bit_gradients,
                    ignore_bits=ignore_bits,
                )

            bit_cache[lbl] = True

            print(
                "Evaluated {:d} bit splits, producing {:d} hits of max bit depth of {:d}".format(
                    completed, hits, maxbits
                )
            )
            pbar.close()

        if len(bit_gradients) == 0:
            return None, None

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
        for bit_gradient in bit_gradients:
            keep = True
            for ignore in ignore_bits:
                if bit_gradient[0] == ignore[0] and bit_gradient[1] == ignore[1]:
                    keep = False
                    break
            if keep:
                bg.append(bit_gradient)
        bit_gradients = bg
        # sort by largest difference, then by fewest bits, so two splits with
        # the same difference will cause the most general split to win
        bit_gradients = sorted(
            bit_gradients,
            key=lambda x: (np.max(np.abs(x[2][mode[0]])), -x[1].bits(), x[1]),
            reverse=True,
        )

        if len(bit_gradients):
            print(
                "\n\n### Here are the candidates to split, ordered by priority (mode: {}) ###".format(
                    mode[0]
                )
            )
            for bit_gradient in bit_gradients:
                print("bits={}".format(bit_gradient[1].bits()), bit_gradient)

            if self.score_candidate_limit is not None:
                print(
                    "Limiting candidates to the top {}".format(
                        self.score_candidate_limit
                    )
                )
                bit_gradients = bit_gradients[: self.score_candidate_limit]

        QCA = self._po.source.source
        # self.to_smirnoff_xml("tmp.offxml", verbose=False)
        # coverage_pre = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA).assign_labels_from_openff("tmp.offxml", "tmp.offxml")
        # need this for measuring geometry
        # should only need to do it once

        if verbose:
            print("\n\nHere is ignore:")
            for ignore in ignore_bits:
                print(ignore)
            print("\n")

        for bit_gradient in bit_gradients:
            # split_bit = bit_gradients[0][1]
            lbl = bit_gradient[0]
            split_bit = bit_gradient[1]
            split_combination = bit_gradient
            if all(
                [
                    not (
                        (x[0] == lbl and x[1] == split_bit)
                        and (
                            x[0] == lbl
                            and any([split_combination == y[1] for y in ignore_bits[x]])
                        )
                    )
                    for x in ignore_bits
                ]
            ):

                # child = group - split_bit
                node = self.split_parameter(lbl, split_bit)
                if node is None:
                    continue
                # self.to_smirnoff_xml("tmp.offxml", verbose=False)
                # coverage_post = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA).assign_labels_from_openff("tmp.offxml", "tmp.offxml")
                # overlapped = self._check_overlapped_parameter(coverage_pre, coverage_post, node)
                print("Adding ", lbl, split_bit, "to ignore")
                if (lbl, split_bit) in ignore_bits:
                    ignore_bits[(lbl, split_bit)].append(
                        (bit_gradient[2], bit_gradient[3])
                    )
                else:
                    ignore_bits[(lbl, split_bit)] = [(bit_gradient[2], bit_gradient[3])]
                print(
                    "\n=====\nSplitting",
                    lbl,
                    "to",
                    node.payload,
                    "\n\n",
                    self.db[lbl]["data"]["group"],
                    "using\n",
                    split_bit,
                    "\nvals (key=",
                    key,
                    ") eps=",
                    eps,
                    bit_gradient[2],
                    "\nresult is\n",
                    self.db[lbl]["data"]["group"] - split_bit,
                )
                print(
                    "\n\nParent SMARTS:",
                    self.db[lbl]["data"]["group"].to_smarts(),
                    "\nNew SMARTS   :",
                    (self.db[lbl]["data"]["group"] - split_bit).to_smarts(),
                    "\n\n",
                )

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

                return node, bit_gradient[2]
            else:
                print("This parameter is in the ignore list:", bit_gradient)

        return None, None

    def _calculate_ic_force_constants(self):

        """
        TODO: Make an operation out of this rather than do it here
        """

        if hasattr(self, "_ic"):
            return

        self._ic = {}
        self._fc = {}
        self._prim = {}

        import offsb.ui.qcasb
        import geometric.internal
        import geometric.molecule

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

        # should return a dict of the kind
        #
        # well, the measure already collects what is needed
        # need a way to transform indices to primitives st we have prim: measure
        # so we could just keep a indices -> prim, which is what _to is

        self._to.processes = None
        self._to.chembit = False
        self._to.apply()

        ic_op = offsb.op.internal_coordinates.InteralCoordinateGeometricOperation(
            QCA, "ic", verbose=True
        )
        ic_op.processes = None
        ic_op.apply()

        # need this for measuring geometry
        # should only need to do it once
        qcasb = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA)
        qcasb.verbose = False

        self.to_smirnoff_xml("tmp.offxml", verbose=False)
        labeler = qcasb.assign_labels_from_openff("tmp.offxml", "tmp.offxml")
        self._labeler = labeler

        self._ic = qcasb.measure_internal_coordinates()
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

            for grad_node in QCA.node_iter_depth_first(entry, select="Gradient"):
                for mol in QCA.node_iter_depth_first(grad_node, select="Molecule"):
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
                                param_name = "i"
                            else:
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
                                else:
                                    prim = prim_to_graph[
                                        param_name[0]
                                    ].from_string_list(
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

    def print_label_assignments(self):

        """
        print the entry, atoms, prim, label, label smarts
        """

        handlers = [
            self[x]
            for x in self.root().children
            if self[x].payload in self.parameterize_handlers
        ]

        QCA = self._po.source.source

        print("\nPRINT OUT OF MOLECULE ASSIGNMENTS\n")

        self.to_smirnoff_xml("tmp.offxml", verbose=False)
        self._labeler = offsb.ui.qcasb.QCArchiveSpellBook(
            QCA=QCA
        ).assign_labels_from_openff("tmp.offxml", "tmp.offxml")

        params = {
            lbl: param
            for ph in handlers
            for lbl, param in self._labeler.db["ROOT"]["data"][ph.payload].items()
        }

        for entry in QCA.iter_entry():

            labels = {
                aidx: lbl
                for ph in handlers
                for aidx, lbl in self._labeler.db[entry.payload]["data"][
                    ph.payload
                ].items()
            }

            # prims = self._prim[entry.payload]

            for aidx in labels:
                lbl = labels[aidx]
                if lbl is None:
                    breakpoint()
                    continue
                if self._to.chembit:
                    smarts = self._to.db[entry.payload]["data"][aidx].to_smarts()
                else:
                    smarts = "".join(self._to.db[entry.payload]["data"][aidx])

                print(
                    "    ",
                    entry.payload,
                    aidx,
                    lbl,
                    params[lbl]["smirks"],
                    smarts,
                    # prims[aidx],
                )
            print("---------------------------------")

        print("#################################")

    ###

    def _combine_reference_data(self, QCA=None, prims=True):
        """
        Collect the expected parameter values directly from the reference QM
        molecules
        """
        import offsb.ui.qcasb
        import geometric.internal

        # import geometric.molecule

        if QCA is None:
            QCA = self._po.source.source

        if hasattr(self, "_qca"):
            if self._qca is not QCA and hasattr(self, "_ic"):
                del self._ic

        self._qca = QCA

        self._calculate_ic_force_constants()

        # need this for measuring geometry
        # should only need to do it once

        qcasb = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA)
        vtable = {
            "Bonds": qcasb.measure_bonds,
            "Angles": qcasb.measure_angles,
            "ImproperTorsions": qcasb.measure_outofplanes,
            "ProperTorsions": qcasb.measure_dihedrals,
        }

        if self._labeler is None:
            self.to_smirnoff_xml("tmp.offxml", verbose=False)
            labeler = qcasb.assign_labels_from_openff("tmp.offxml", "tmp.offxml")
            self._labeler = labeler

        labeler = self._labeler

        # should return a dict of the kind
        #
        # well, the measure already collects what is needed
        # need a way to transform indices to primitives st we have prim: measure
        # so we could just keep a indices -> prim, which is what _to is

        # so to make this work, we iterate the bits
        # then

        param_data = {}
        n_entries = len(list(QCA.iter_entry()))
        for entry in tqdm.tqdm(
            QCA.iter_entry(),
            total=n_entries,
            desc="label assignment",
            ncols=80,
            disable=False,
        ):
            # need to unmap... sigh
            # smi = QCA.db[entry.payload]['data'].attributes['canonical_isomeric_explicit_hydrogen_mapped_smiles']
            # rdmol = offsb.rdutil.mol.build_from_smiles(smi)
            # atom_map = offsb.rdutil.mol.atom_map(rdmol)
            # map_inv = offsb.rdutil.mol.atom_map_invert(atom_map)
            labels = {
                aidx: val
                for ic_type in vtable
                for aidx, val in labeler.db[entry.payload]["data"][ic_type].items()
            }

            for ic_type, ic in self._ic.items():

                params = labeler.db["ROOT"]["data"][ic_type]

                for param_name, param in params.items():
                    # if select is not None and param_name not in select:
                    #     continue
                    if param_name not in param_data:
                        param_data[param_name] = {}
                    for mol in QCA.node_iter_depth_first(entry, select="Molecule"):
                        ic_data = ic.db[mol.payload]
                        for aidx, vals in ic_data.items():
                            # for aidx in labels:
                            # if aidx not in labels:
                            #     breakpoint()
                            #     print("HI!")
                            if labels[aidx] == param_name:
                                # param_vals.extend(vals)
                                if aidx not in self._prim[entry.payload]:
                                    breakpoint()
                                    print("HI!")

                                key = self._prim[entry.payload][aidx]
                                # if aidx not in self._fc[mol.payload]:
                                #     breakpoint()
                                #     print("HI!")

                                if key not in param_data[param_name]:
                                    param_data[param_name][key] = {
                                        "measure": [],
                                        "force": [],
                                    }

                                mol_fc = self._fc.get(mol.payload)
                                param_data[param_name][key]["measure"].extend(vals)

                                if mol_fc is not None:
                                    force_vals = mol_fc[aidx]

                                    param_data[param_name][key]["force"].append(
                                        force_vals
                                    )

        return param_data

    ###

    def _combine_optimization_data(self):

        QCA = self._po.source.source

        # smi_to_label = self._po._setup.labeler.db["ROOT"]["data"]
        # smi_to_label = {
        #     k: v["smirks"] for keys in smi_to_label.values() for k, v in keys.items()
        # }
        # smi_to_label = {v: k for k, v in smi_to_label.items()}

        param_names = self._po._forcefield.plist
        param_labels = [param.split("/")[-1] for param in param_names]
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

        n_entries = len(list(QCA.iter_entry()))
        for i, entry in enumerate(QCA.iter_entry()):

            print("    {:8d}/{:8d} : {}".format(i + 1, n_entries, entry))

            if entry.payload not in self._po._setup.labeler.db:
                continue
            labels = self._po._setup.labeler.db[entry.payload]["data"]
            labels = {k: v for keys in labels.values() for k, v in keys.items()}

            if entry.payload not in self._to.db:
                continue
            primitives = self._to.db[entry.payload]["data"]

            for molecule in QCA.node_iter_depth_first(entry, select="Molecule"):
                mol_id = molecule.payload

                obj = self._po.db.get(mol_id)
                if obj is None:
                    continue

                obj = obj["data"]

                # IC keys (not vdW)
                for key in [k for k in obj if type(k) == tuple and k in labels]:
                    matched_params = []
                    for j, val in enumerate(obj[key]["dV"]):
                        if labels[key] == param_labels[j]:
                            matched_params.append(val)

                    lbl = labels[key]

                    if len(matched_params) == 0:
                        # print("This label didn't match:", key, "actual label", labels[key])
                        # print("Debug: param_labels is", param_labels )
                        # if we have no matches, then likely we are not trying to
                        # fit to it, so we can safely skip
                        continue

                    # if lbl == 't3':
                    #     breakpoint()
                    prim = prim_to_graph[lbl[0]].from_string_list(primitives[key])
                    if lbl not in param_data:
                        param_data[lbl] = {}
                        all_data[lbl] = {}
                    if prim not in param_data[lbl]:
                        param_data[lbl][prim] = [matched_params]
                        all_data[lbl][prim] = [obj[key]["dV"]]

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

                    # this is the sum over all IC, for FF param i (the vdW terms)
                    for i in matched_params_idx:
                        ff_grad.append(
                            sum([obj[k]["dV"][i] for k in obj if type(k) == tuple])
                        )

                    # since we are applying the entire FF gradient to this single
                    # IC, we need to divide by the number of ICs which also match
                    # this param. This causes the gradient to be spread out over
                    # all matched ICs
                    # Also, we need to spread out over the entire parameter list
                    # so divide by number of FF params as well
                    # Finally, we need a denom-like scalar to make the magnitudes
                    # similar to existing parameters; it is currently chosen to
                    # be on the scale of bonds (very sensitive)

                    for i, _ in enumerate(ff_grad):
                        ff_grad[i] /= (
                            len([x for x in vdw_keys if labels[x] == labels[key]])
                            * VDW_DENOM
                            # * len(param_labels)
                        )

                    matched_params = ff_grad
                    all_params = list([np.mean(ff_grad)] * len(param_labels))
                    lbl = labels[key]
                    prim = prim_to_graph[lbl[0]].from_string(primitives[key][0])
                    if lbl not in param_data:
                        param_data[lbl] = {}
                        all_data[lbl] = {}
                    if prim not in param_data[lbl]:
                        param_data[lbl][prim] = [matched_params]
                        all_data[lbl][prim] = [all_params]

                    param_data[lbl][prim].append(matched_params)
                    all_data[lbl][prim].append(all_params)
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

    def _run_optimizer(self, jobtype):
        newff_name = "tmp.offxml"

        options_override = {}
        if self.trust0 is not None:
            options_override["trust0"] = self.trust0
            print("Setting trust0 to", self.trust0)
        else:
            self.trust0 = self._po._options.get("trust0")

        if self.finite_difference_h is not None:
            options_override["finite_difference_h"] = self.finite_difference_h
            print("Setting finite_difference_h to", self.finite_difference_h)
        else:
            self.finite_difference_h = self._po._options.get("finite_difference_h")

        if self.eig_lowerbound:
            options_override["eig_lowerbound"] = self.eig_lowerbound
            print("Setting eig_lowerbound to", self.eig_lowerbound)
        else:
            self.eig_lowerbound = self._po._options.get("eig_lowerbound")

        while True:
            try:
                self._po.load_options(options_override=options_override)
                self.to_smirnoff_xml(newff_name, verbose=False)
                self._po._setup.ff_fname = newff_name
                self._po.ff_fname = newff_name
                self._po._init = False

                self._po.apply(jobtype=jobtype)
                self.load_new_parameters(self._po.new_ff)

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

        self.trust0 = self._po._options.get("trust0")
        self.finite_difference_h = self._po._options.get("finite_difference_h")
        self.eig_lowerbound = self._po._options.get("eig_lowerbound")

        return True

    def _optimize_type_iteration(
        self,
        optimize_during_typing=False,
        optimize_during_scoring=False,
        ignore_bits=None,
        use_gradients=True,
        split_strategy="spatial_reference",
        ignore_parameters=None,
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

        bit_gradients = []

        if ignore_bits is None:
            ignore_bits = {}

        if ignore_parameters is None:
            ignore_parameters = []

        # BREAK
        # breakpoint()

        if use_gradients:
            print("Running reference gradient calculation...")

            success = self._run_optimizer(jobtype)

            if not success:
                print("Reference gradient calculation failed; cannot continue!")
                return None, np.inf, -np.inf

            obj = self._po.X

            grad_scale = 1.0

            grad = self._po.G
            best = [None, grad * grad_scale, obj, None, None, -1, None, None, self.db]
        else:
            best = [None, np.inf, np.inf, None, None, -1, None, None, self.db]

        param_data = self._combine_reference_data()

        print("Assignments from initial FF:")
        self.print_label_assignments()

        current_ff = "tmp.offxml"
        self.to_smirnoff_xml(current_ff, verbose=False)

        bit_gradients = []
        bit_cache = {}

        olddb = copy.deepcopy(self._po.db)

        eps = 1.0

        if use_gradients:
            print("Running reference gradient calculation...")

            success = self._run_optimizer(jobtype)
            if not success:
                print("Reference gradient calculation failed; cannot continue!")
                return None, np.inf

            obj = self._po.X

            grad_scale = 1.0

            grad = self._po.G

        # This generates the tier 1 scoring data
        if split_strategy == "spatial_reference":
            key = "measure"
            mode = ["mean_difference"]
            eps = 1.0
            # param_data = self._combine_reference_data()
        elif split_strategy == "force_reference":
            key = "force"
            mode = ["mean_difference"]
            eps = 10.0
            # param_data = self._combine_reference_data()
        elif use_gradients:

            eps = 1e-14
            key = None
            # mode = "sum_difference"
            mode = ["sum_difference", "mag_difference"][::-1]
            param_data, all_data = self._combine_optimization_data()

        if not self.gradient_assigned_only:
            param_data = all_data

        if ignore_parameters is not None and len(ignore_parameters):
            print("Stripping parameters:", ignore_parameters)
            param_data = {
                k: v for k, v in param_data.items() if k[0] not in ignore_parameters
            }

        candidate_mode_choices = ["split_gradient_max"]
        candidate_mode_choices.extend(mode)

        candidate_mode = candidate_mode_choices[0]

        while len(candidates) < candidate_limit:
            if use_gradients:
                print("\n\nMicroiter", i)
            i += 1

            # print("Finding new split...")
            # print("Ignore bits are")
            # for ignore, grads in ignore_bits.items():
            #     print(grads, ignore)

            # This is tier 1 scoring
            # Examines the entire set and tries to find the next split
            # This means the bits split are valid until a split is kept

            node, score = self._find_next_split(
                param_data,
                key=key,
                ignore_bits=ignore_bits,
                mode=mode,
                eps=eps,
                bit_gradients=bit_gradients,
                bit_cache=bit_cache,
            )
            print("Split is", node)
            print("Score is", score)

            if node is None:
                break

            print("Assignments post split:")
            self.print_label_assignments()

            if use_gradients:

                # This starts the tier 2 scoring

                # self._po._options["forcefield"] = [newff_name]
                print("Calculating new gradient with split param")

                # self._po.logger.setLevel(logging.ERROR)
                success = self._run_optimizer(jobtype)
                # self._po.logger.setLevel(self.logger.getEffectiveLevel())

                if not success:
                    print("Gradient failed for this split; skipping")
                    # if there is an exception, the po will have no data
                    self._po.db = olddb
                else:
                    ref_obj = self._po.X
                    grad_new = self._po.G
                    print(
                        "\ngrad_new",
                        grad_new,
                        "grad",
                        grad,
                        "grad_new > abs(grad*scale)?",
                        np.abs(grad_new - grad * grad_scale) > eps,
                        grad_new - grad * grad_scale,
                        eps,
                    )

                    grad_new_opt = np.inf
                    if optimize_during_scoring:

                        print("Performing micro optimization for candidate")

                        success = self._run_optimizer("OPTIMIZE")

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

                    # easy mode: take the best looking split
                    # best = [node, grad_new, node.parent, self.db[node.payload]]
                    # break

                    # hard core mode: calc all and take the best best
                    candidate = [
                        node.copy(),
                        grad_new,
                        node.parent,
                        self.db[node.payload].copy(),
                        score,
                        len(candidates),
                        ref_obj,
                        obj,
                        copy.deepcopy(self.db),
                        grad,
                        grad_new_opt,
                    ]
                    candidates.append(candidate)
                    # np.abs(x[7]), reverse=False  is the best objective drop (brute force)
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
                            self[c[2]].payload,
                            "->",
                            c[0].payload,
                            self.db[self[c[2]].payload]["data"]["group"].to_smarts(),
                            "->",
                            c[3]["data"]["group"].to_smarts(),
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

                # remove the previous term if it exists
                print("Remove parameter", node)

                self[node.parent].children.remove(node.index)
                self.node_index.pop(node.index)

                self.load_new_parameters(current_ff)

                newff_name = current_ff
                self.to_smirnoff_xml(newff_name, verbose=False)
                # self._po._options["forcefield"] = [newff_name]

                self._po._setup.ff_fname = newff_name
                self._po.ff_fname = newff_name
                self._po._init = False

            else:
                best = [
                    node.copy(),
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
                ]
                candidates = [best]
                # hack so that we add it using the common path below
                self[node.parent].children.remove(node.index)
                self.node_index.pop(node.index)
                self.db.pop(node.payload)
                break

        print("First scoring pass complete; assessing candidates")

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
                    return None, np.inf, np.inf

            elif candidate_mode == "split_gradient_max":
                candidates = sorted(candidates, key=lambda x: x[1] - x[9], reverse=True)
            elif candidate_mode in [
                "sum_difference",
                "mag_difference",
                "mean_difference",
            ]:
                candidates = sorted(
                    candidates, key=lambda x: np.abs(x[4]), reverse=True
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
                for ii, candidate in enumerate(candidates[:total_candidates], 1):
                    olddb = self.db
                    node = candidate[0]

                    print(
                        "Add candidate parameter", node, "to parent", self[candidate[2]]
                    )
                    candidate[0] = self.add(candidate[2], candidate[0], index=0)
                    self.db = candidate[8]

                    print(
                        "Performing micro optimization for the best candidate score number {}/{}".format(
                            ii, total_candidates
                        )
                    )
                    self.to_smirnoff_xml(
                        "newFF_" + str(i) + "." + str(ii) + ".offxml", verbose=True
                    )
                    success = self._run_optimizer("OPTIMIZE")

                    if success:
                        obj = self._po.X
                        grad_new_opt = self._po.G
                        print("Objective after minimization:", self._po.X)
                        self.load_new_parameters(self._po.new_ff)
                        candidate[7] = obj
                        candidate[10] = grad_new_opt

                        # since we are optimizing, overwrite the SP gradient
                        candidate[1] = grad_new_opt
                        n_success += 1
                    else:
                        self.logger.info("Optimization failed; assuming bogus split")

                    # print("Remove candidate parameter", node, "from parent", node.parent, candidate[2], node.parent == candidate[2])
                    self[node.parent].children.remove(node.index)
                    self.node_index.pop(node.index)
                    self.db = olddb

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
                    self[c[2]].payload,
                    "->",
                    c[0].payload,
                    self.db[self[c[2]].payload]["data"]["group"].to_smarts(),
                    "->",
                    c[3]["data"]["group"].to_smarts(),
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
                return None, np.inf, np.inf
            best = candidates[0]
            # only re-add if we did a complete scan, since we terminate that case
            # with no new node, and the best has to be re-added
            # if we break early, the node is already there

            # I think nodes need to be prepended to conserve hierarchy
            # for example, if we split a param, do we want it to override
            # all children? no, since we were only focused on the parent, so
            # we only care that the split node comes after *only* the parent,
            # which is true since it is a child
            best[0] = self.add(best[2], best[0], index=0)
            self.db = best[8]

            print("Best split parameter")
            print(best[0])
            print("Best split gradient", best[1])
            print("Best split score", best[4])
            print(
                "Best split objective drop {}%".format(
                    100.0 * (best[7] - best[6]) / best[6]
                )
            )

        # newff_name = "newFF.offxml"
        # self.db = best[8]
        # self.to_smirnoff_xml(newff_name, verbose=False)

        # self._po._setup.ff_fname = newff_name
        # self._po._init = False

        return best[0], best[1], best[7]

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

        elif ptype in [
            ImproperTorsionHandler.ImproperTorsionType,
            ProperTorsionHandler.ProperTorsionType,
        ]:
            current = param.phase
            n_params = len(param.phase)
            if issubclass(type(value), list):
                new = list(
                    [x * simtk.unit.degree for x, _ in zip(value, range(n_params))]
                )
            else:
                new = list([value * simtk.unit.degree for _ in range(n_params)])
            if modify:
                param.phase = new
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

    def _set_parameter_force(self, param_name, value, report_only=False):

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
        spatial=True,
        force=True,
    ):
        """
        get the ops from the data
        get the labels from the data
        combine
        use the qcasb module?!
        """
        # import offsb.ui.qcasb
        # import geometric.internal
        # import geometric.molecule

        print("\n\nInitializing parameters from data")

        if ignore_parameters is None:
            ignore_parameters = []

        fn_map = {}

        if spatial:
            fn_map["measure"] = self._set_parameter_spatial

        if force:
            fn_map["force"] = self._set_parameter_force

        param_data = self._combine_reference_data(QCA=QCA, prims=False)

        for param_name, param_types in param_data.items():
            if param_name[0] in ignore_parameters:
                continue
            if only_parameters is None or param_name in only_parameters:
                param_types = list(param_types.values())
                for p_type, fn in fn_map.items():
                    new_vals = []
                    for param_dict in param_types:
                        for p, vals in param_dict.items():
                            if p == p_type:
                                new_vals.extend(vals)
                    # new_vals = param_data[param_name](p_type, None)
                    if len(new_vals) > 0:
                        val = np.mean(new_vals)
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

    def split_from_data(self, ignore_parameters=None, modify_parameters=False):

        if ignore_parameters is None:
            ignore_parameters = []
        ignore_bits = {}
        ignore_bits_optimized = {}
        ref_grad = np.inf
        ref = np.inf
        newff_name = "tmp.offxml"

        i = -1
        # self._to.apply()

        splits = -1

        print("Splitting parameters from data")
        while splits != 0:
            splits = 0
            print("Splitting based on spatial difference")
            while True:
                i += 1
                try:
                    node, grad_split, score = self._optimize_type_iteration(
                        optimize_during_typing=False,
                        ignore_bits=ignore_bits,
                        split_strategy="spatial_reference",
                        use_gradients=False,
                        ignore_parameters=ignore_parameters,
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
                    bit = (
                        self.db[parent.payload]["data"]["group"]
                        - self.db[node.payload]["data"]["group"]
                    )
                    ignore_bits[(None, bit)] = [None, None]
                    # the db is in a null state, and causes failures
                    # TODO allow skipping an optimization
                    # reset the ignore bits since we found a good move

                # ignore_bits_optimized = {}
                # ignore_bits = {}
                print("Keeping iteration", i)
                print("Split kept is")
                print(node)
                splits += 1
                # ref = obj
                # ref_grad = grad_split
                # self._plot_gradients(fname_prefix=str(i) + ".accept")

                # reset values of the new split
                self.initialize_parameters_from_data(
                    QCA=self._to.source.source,
                    only_parameters=[node.payload, self[node.parent].payload],
                    report_only=not modify_parameters,
                )

                self.to_smirnoff_xml("newFF" + str(i) + ".accept.offxml", verbose=False)
                self.to_pickle()

            print("Splitting based on force difference")
            ignore_bits = {}
            while True:
                i += 1
                try:
                    node, grad_split, score = self._optimize_type_iteration(
                        optimize_during_typing=False,
                        ignore_bits=ignore_bits,
                        split_strategy="force_reference",
                        use_gradients=False,
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
                    bit = (
                        self.db[parent.payload]["data"]["group"]
                        - self.db[node.payload]["data"]["group"]
                    )
                    print("Adding ", bit, "to ignore")
                    ignore_bits[(None, bit)] = [None, None]
                    # the db is in a null state, and causes failures
                    # TODO allow skipping an optimization
                    # reset the ignore bits since we found a good move

                # ignore_bits_optimized = {}
                # ignore_bits = {}
                print("Keeping iteration", i)
                print("Split kept is")
                print(node)
                splits += 1
                # ref = obj
                # ref_grad = grad_split
                # self._plot_gradients(fname_prefix=str(i) + ".accept")

                # reset values of the new split
                if False:
                    self.initialize_parameters_from_data(
                        QCA=self._to.source.source,
                        only_parameters=[node.payload, self[node.parent].payload],
                        report_only=not modify_parameters,
                    )

                self.to_smirnoff_xml("newFF" + str(i) + ".accept.offxml", verbose=False)
                self.to_pickle()
                # self.load_new_parameters("newFF" + str(i) + ".accept.offxml")
            break
        # print("Final objective is", obj, "initial was", initial)
        # print("Total drop is", obj - initial)
        self.to_smirnoff_xml(newff_name, verbose=False)
        self.to_pickle()

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

    def optimize_v2(
        self,
        optimize_types=True,
        optimize_parameters=False,
        optimize_during_typing=True,
        optimize_initial=True,
    ):

        self._po.apply(jobtype=jobtype)
        obj, ref_grad = self._po.X, self._po.G
        self.load_new_parameters(self._po.new_ff)

        # this should force a reinit the forcefield but not the targets
        self._po.reset()

        success = True
        i = 0
        while success:
            print("Macroiteration", i)
            report = self._optimize_step()
            success = report.success
            report.print_result()
            i += 1

    def optimize(
        self,
        optimize_types=True,
        optimize_parameters=False,
        optimize_during_typing=True,
        optimize_during_scoring=False,
        optimize_initial=True,
    ):

        # self._to.apply()

        newff_name = "input.offxml"
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
        success = self._run_optimizer(jobtype)
        if not success:
            print("Failed. Cannot proceed.")
            return
        # self.finite_difference_h = self._po._options["finite_difference_h"]

        print("Initial objective :", self._po.X)
        obj = self._po.X
        ref = obj
        initial = obj
        ref_grad = self._po.G

        self._labeler = None
        newff_name = "tmp.offxml"
        self.load_new_parameters(self._po.new_ff)
        self.to_pickle()
        self.to_smirnoff_xml(newff_name, verbose=False)
        self._po._setup.ff_fname = newff_name
        self._po._init = False

        ignore_bits = {}
        ignore_bits_optimized = {}

        # self._plot_gradients(fname_prefix="0")

        if optimize_types:
            i = 0
            if False:
                pass
            else:
                while True:
                    # while True:
                    i += 1
                    # This objective should be exactly the same as the fit obj,
                    # but no regularization, therefore we ignore it mostly
                    current_ff = self._po.new_ff

                    try:
                        node, grad_split, score = self._optimize_type_iteration(
                            optimize_during_typing=optimize_during_typing,
                            optimize_during_scoring=optimize_during_scoring,
                            ignore_bits=ignore_bits,
                            split_strategy="gradient",
                            use_gradients=True,
                        )

                        if node is None:
                            print("No new parameter split, done!")
                            break

                        obj = score

                    except RuntimeError as e:
                        self.logger.error(str(e))
                        self.logger.error("Optimization failed; assuming bogus split")
                        obj = np.inf
                        grad_split = np.inf
                        try:
                            parent = self[node.parent]
                        except KeyError as e:
                            # breakpoint()
                            pass
                        bit = (
                            self.db[parent.payload]["data"]["group"]
                            - self.db[node.payload]["data"]["group"]
                        )
                        ignore_bits[(None, bit)] = []
                        # the db is in a null state, and causes failures
                        # TODO allow skipping an optimization

                    print(
                        "New objective:",
                        obj,
                        "Delta is",
                        obj - ref,
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

                        self.to_smirnoff_xml(newff_name, verbose=False)
                        self._po._setup.ff_fname = newff_name
                        self._po.ff_fname = newff_name
                        self._po._init = False
                        print(
                            "Rejecting iteration", i, "objective reference still", ref
                        )
                        # self._plot_gradients(fname_prefix=str(i) + ".reject")
                        # self.to_smirnoff_xml("newFF" + str(i) + ".reject.offxml")
                        parent = self[node.parent]
                        bit = (
                            self.db[parent.payload]["data"]["group"]
                            - self.db[node.payload]["data"]["group"]
                        )
                        try:
                            # patch; assume everything is correct and we can
                            # just add symmetric cases without issue
                            if bit not in [x[1] for x in ignore_bits]:
                                print("Adding ", bit, "to ignore")
                                ignore_bits[(None, bit)] = [None, None]
                            ignore_bits_optimized[(None, bit)] = ignore_bits[
                                (None, bit)
                            ].copy()
                        except KeyError as e:
                            # probably a torsion or something that caused
                            # the bit calculation to fail
                            print("Key error during ignore add:")
                            print(e)
                            ignore_bits_optimized[(None, bit)] = [None, None]

                        print("Keeping ignore bits for next iteration:")
                        for (lbl, bit), bit_grad in ignore_bits_optimized.items():
                            print(lbl, bit_grad, bit)

                        # ignore_bits will keep track of things that didn't work
                        # do this after keeping the ignore bits since we access
                        # the info
                        self[node.parent].children.remove(node.index)
                        self.node_index.pop(node.index)
                        self.db.pop(node.payload)

                        # reset the physical terms after resetting the tree
                        self.load_new_parameters(current_ff)

                        if optimize_during_typing:
                            ignore_bits = ignore_bits_optimized.copy()
                    else:
                        # reset the ignore bits since we found a good move

                        ignore_bits_optimized = {}
                        ignore_bits = {}
                        print("Keeping iteration", i, "objective:", obj)
                        print("Split kept is")
                        print(node)
                        ref = obj
                        ref_grad = grad_split
                        # self._plot_gradients(fname_prefix=str(i) + ".accept")
                        self._labeler = None
                        self.to_smirnoff_xml("newFF" + str(i) + ".accept.offxml")
                        self.to_pickle()
            self.to_smirnoff_xml(newff_name, verbose=True)
            self.to_pickle()
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
            newff_name = "newFF.offxml"
            # self._plot_gradients(fname_prefix="optimized.final")
            self.to_smirnoff_xml(newff_name, verbose=True, renumber=False)
            self.to_pickle()
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

    def set_smarts_generator(self, obj):
        self._to = obj
