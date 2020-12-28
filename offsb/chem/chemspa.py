#!/usr/bin/env python

import copy
import functools
import io
import logging
import os
import pprint
import re
import sys
import tempfile

import numpy as np
import simtk.unit
import tqdm

import offsb.chem.types
import offsb.op.chemper
import offsb.op.forcebalance
import offsb.tools.const
import offsb.treedi.node
import offsb.treedi.tree
from offsb.treedi.tree import DEFAULT_DB
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.typing.engines.smirnoff.parameters import (
    AngleHandler, BondHandler, ImproperDict, ImproperTorsionHandler,
    ParameterList, ProperTorsionHandler, ValenceDict, vdWHandler)

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
    "ImproperTorsions": None,
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

            # default try to make new types for these handlers
            self.parameterize_handlers = [
                "vdW",
                "Bonds",
                "Angles",
                "ImproperTorsions",
                "ProperTorsions",
            ]

            # root = self.root()

            # # # This sets up a "blank" ChemSpace, one term that covers everything
            # for pl_name in ["vdW"]:

            #     node = offsb.treedi.node.Node(name=pl_name, payload=pl_name)
            #     node = self.add(root.index, node)
            #     ph = vdWHandler(version="0.3")
            #     self.db[self.name + "-" + pl_name] = DEFAULT_DB({"data": ph})

            #     smirks = "[*]"
            #     param = vdWHandler.vdWType(
            #         epsilon=".1 * kilocalorie/mole",
            #         rmin_half="1.6 * angstrom",
            #         smirks=smirks,
            #         id="n1",
            #     )

            #     param_name = param.id

            #     pnode = offsb.treedi.node.Node(
            #         name=str(type(param)), payload=param_name
            #     )
            #     self.add(node.index, pnode)
            #     self.db[param_name] = DEFAULT_DB(
            #         {
            #             "data": {
            #                 "parameter": param,
            #                 "group": offsb.chem.types.AtomType.from_string(
            #                     param.smirks
            #                 ),
            #             }
            #         }
            #     )
            # for pl_name in ["Bonds"]:

            #     node = offsb.treedi.node.Node(name=pl_name, payload=pl_name)
            #     node = self.add(root.index, node)
            #     ph = BondHandler(version="0.3")
            #     self.db[self.name + "-" + pl_name] = DEFAULT_DB({"data": ph})

            #     smirks = "[*:1]~[*:2]"
            #     param = BondHandler.BondType(
            #         k="20.0 * kilocalorie/(angstrom**2*mole)",
            #         length="1.4 * angstrom",
            #         smirks=smirks,
            #         id="b1",
            #     )
            #     # ph.add_parameter(param)

            #     param_name = param.id

            #     pnode = offsb.treedi.node.Node(
            #         name=str(type(param)), payload=param_name
            #     )
            #     self.add(node.index, pnode)
            #     group = offsb.chem.types.BondGroup.from_string(param.smirks)
            #     self.db[param_name] = DEFAULT_DB(
            #         {
            #             "data": {
            #                 "parameter": param,
            #                 "group": group,
            #             }
            #         }
            #     )
            # for pl_name in ["Angles"]:

            #     node = offsb.treedi.node.Node(name=pl_name, payload=pl_name)
            #     node = self.add(root.index, node)
            #     ph = AngleHandler(version="0.3")
            #     self.db[self.name + "-" + pl_name] = DEFAULT_DB({"data": ph})

            #     smirks = "[*:1]~[*:2]~[*:3]"
            #     param = AngleHandler.AngleType(
            #         k="10.0 * kilocalorie/(degree**2*mole)",
            #         angle="109.5 * degree",
            #         smirks=smirks,
            #         id="a1",
            #     )
            #     # ph.add_parameter(param)

            #     param_name = param.id

            #     pnode = offsb.treedi.node.Node(
            #         name=str(type(param)), payload=param_name
            #     )
            #     self.add(node.index, pnode)
            #     self.db[param_name] = DEFAULT_DB(
            #         {
            #             "data": {
            #                 "parameter": param,
            #                 "group": offsb.chem.types.AngleGroup.from_string(
            #                     param.smirks
            #                 ),
            #             }
            #         }
            #     )

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
            for i, param_node in enumerate(self.node_iter_dive(ph_node), 1):
                if param_node.payload not in self.db:
                    print("This param not in the db???!", param_node.payload)
                    continue
                if param_node == ph_node:
                    continue
                param = self.db[param_node.payload]["data"]["parameter"]
                if param.id in visited:
                    continue
                visited.add(param.id)
                param = copy.deepcopy(param)
                if renumber:
                    num_map[param.id] = ph_to_letter[ph_node.payload] + str(i)
                    param.id = num_map[param.id]
                params.append(param)
            ff_ph._parameters = ParameterList(params)

            # this is for printing
            if not verbose:
                continue
            for i, param_node in enumerate(self.node_iter_dive(ph_node), 1):
                if param_node.payload not in self.db:
                    continue
                if param_node == ph_node:
                    continue
                param = copy.deepcopy(self.db[param_node.payload]["data"]["parameter"])

                if renumber:
                    param.id = num_map[param.id]
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
                    "{:12s} : {}".format("id", param_node.payload),
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
                    nodes[group].append((pnode, param))
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
                    cls.add(parent.index, pnode)
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
                        cls.add(prev_node.index, bit_node, index=n_idx)
                        prev_node = bit_node
                        n_idx = None

                    # if we have a string of bit nodes, then proceed
                    # "pop" the child from the parent; we need to be the child
                    # of the last bit node we enter
                    if bit_node:
                        del parent_node.children[idx]
                        cls.add(bit_node.index, child)
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
                cls.add(node.index, pnode)
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
                cls.add(node.index, pnode)
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
                cls.add(node.index, pnode)
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
                cls.add(node.index, pnode)
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
                cls.add(node.index, pnode)
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

    def split(self, label, bit):

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
        self.add(node.index, pnode, index=0)

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
        self, param_data, lbl, bit, key=None, eps=1.0, mode="sum_difference"
    ):
        ys_bit = []
        no_bit = []

        denoms = {
            "n": 0.0,
            "b": 0.05 / offsb.tools.const.bohr2angstrom,
            "a": 8.0,
            "i": 20.0,
            "t": 0.0,
        }

        verbose = True

        bit_gradients = []

        for prim, dat in param_data[lbl].items():
            if key is not None:
                dat = dat[key]
            # pdb.set_trace()
            if verbose:
                print(
                    "Considering prim", prim, "with smarts", prim.to_smarts(), end=" "
                )
            if bit in prim:
                if verbose:
                    print("yes (N=", len(dat), ")")
                ys_bit.extend(dat)
            else:
                if verbose:
                    print("no  (N=", len(dat), ")")
                no_bit.extend(dat)
        if verbose:
            print("    Has bit (N=", len(ys_bit), ") :", end=" ")
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
            print("    No bit (N=", len(no_bit), ") :", end=" ")
        if len(no_bit) > 0:
            if verbose:
                print(
                    "mean:",
                    np.mean(no_bit, axis=0),
                    "var:",
                    np.var(no_bit, axis=0),
                    "sum:",
                    np.sum(no_bit, axis=0),
                )

            val = 0.0
            if mode == "sum_difference":
                val = np.sum(no_bit, axis=0) - np.sum(ys_bit, axis=0)
            if mode == "mag_difference":
                val = np.linalg.norm(np.sum(no_bit, axis=0) - np.sum(ys_bit, axis=0))
            elif mode == "mean_difference":
                if lbl[0] in "ait":
                    no_bit = list(map(lambda x: x + 180 if x < 180 else x, no_bit))
                    ys_bit = list(map(lambda x: x + 180 if x < 180 else x, ys_bit))
                val = np.mean(no_bit, axis=0) - np.mean(ys_bit, axis=0)
            if key == "measure":
                if denoms[lbl[0]] == 0.0:
                    val = 0.0
                else:
                    val = val / denoms[lbl[0]]
            success = np.abs(val) > eps
            if key == "measure":
                if denoms[lbl[0]] == 0.0:
                    val = 0.0
                else:
                    val = val * denoms[lbl[0]]
            try:
                success = bool(success)
            except ValueError:
                # we could use all or any here. Using any allows, for a bond for
                # example, either the length or the force to accepted. If we use
                # all, both terms must be above eps. This is only important for
                # sum_difference, which could provide multiple values.
                success = any(success)
            if success:
                new_val = [
                    lbl,
                    bit.copy(),
                    val,
                ]
                if verbose:
                    print("Appending result", new_val)
                bit_gradients.append(new_val)
        else:
            if verbose:
                print("None")

        return bit_gradients

    def _find_next_split(
        self, param_data, key=None, ignore_bits=None, mode="sum_difference", eps=1.0
    ):

        verbose = True

        bit_gradients = []
        if ignore_bits is None:
            ignore_bits = {}

        handlers = [
            self[x]
            for x in self.root().children
            if self[x].payload in self.parameterize_handlers
        ]
        nodes = list(self.node_iter_breadth_first(handlers))
        n_bits = {}
        for node in tqdm.tqdm(nodes, total=len(nodes), desc="bit scanning", ncols=80):
            lbl = node.payload
            # if lbl[0] != 't':
            #     continue

            # This check essentially means this param was not applied
            # Most likely for outofplanes
            if lbl not in param_data:
                continue
            group = functools.reduce(lambda x, y: x + y, param_data[lbl])

            # only iterate on bits that matched from the smirks from the data
            # we are considering

            # The data should be a subset of the parameter that covers it

            # this does not cover degeneracies; use the in keyword

            # if (group - self.db[lbl]["data"]["group"]).reduce() != 0:

            # TODO: why does this true so often???
            # try:
            #     # Checking the sum does not work for torsions, so check each individually
            #     # if group not in self.db[lbl]["data"]["group"]:
            #     if any(
            #         [x not in self.db[lbl]["data"]["group"] for x in param_data[lbl]]
            #     ):
            #         print("ERROR: data is not covered by param!")
            #         print("Group is", group)
            #         print("FF Group is", self.db[lbl]["data"]["group"])
            #         print("marginal is ", group - self.db[lbl]["data"]["group"])
            #         # breakpoint()
            # except Exception as e:
            #     print(e)
            #     breakpoint()

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

            # iterate bits that we cover (AND them just to be careful)
            # group = group & self.db[lbl]["data"]["group"]
            if verbose:
                print("\nContinuing with param ", lbl, "this information:")
                print(group)

            n_bits[lbl] = sum([1 for x in group])

            for bit in group:
                if verbose:
                    print("Scanning for bit", bit)
                if any([x == bit for x in ignore_bits]):
                    if verbose:
                        print("Ignoring since it is in the ignore list")
                    continue
                elif verbose:
                    print("This is a new bit, continuing")

                # for ignore in ignore_bits:
                #     if type(ignore) == type(bit) and :
                #         if verbose:
                #             print("Ignoring since it is in the ignore list. Matches this ignore:")
                #             print(ignore)
                #         continue
                bit_grads = self._scan_param_with_bit(
                    param_data, lbl, bit, key=key, mode=mode, eps=eps
                )
                bit_gradients.extend(bit_grads)

        if len(bit_gradients) == 0:
            return None

        print("\n\nBits per parameter:")
        for i, (b, v) in enumerate(n_bits.items(), 1):
            print("| {:5} {:4d}".format(b, v), end="")
            if i % 7 == 0:
                print(" |\n", end="")
        print("\nTotal parameters:", len(n_bits))

        bit_gradients = sorted(
            bit_gradients, key=lambda x: np.max(np.abs(x[2])), reverse=True
        )
        for bit_gradient in bit_gradients:
            # split_bit = bit_gradients[0][1]
            split_bit = bit_gradient[1]
            if all([x != split_bit for x in ignore_bits]):
                # child = group - split_bit
                lbl = bit_gradient[0]
                node = self.split(lbl, split_bit)
                if node is None:
                    continue
                ignore_bits[split_bit] = bit_gradient[2]
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
                # print("The parent is")
                # print(group.drop(child))
                # print("Smarts is")
                # print(group.drop(child).to_smarts())
                # print("The child is")
                # print(child)
                # print("Smarts is")
                # print(child.to_smarts())

                return node

        return None

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

        # so to make this work, we iterate the bits
        # then

        self._to.apply()

        # need this for measuring geometry
        # should only need to do it once
        qcasb = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA)
        qcasb.verbose = False

        vtable = {
            "Bonds": qcasb.measure_bonds,
            "Angles": qcasb.measure_angles,
            "ImproperTorsions": qcasb.measure_outofplanes,
            "ProperTorsions": qcasb.measure_dihedrals,
        }

        self.to_smirnoff_xml("tmp.offxml", verbose=False)
        labeler = qcasb.assign_labels_from_openff("tmp.offxml", "tmp.offxml")

        n_entries = len(list(QCA.iter_entry()))
        for entry in tqdm.tqdm(
            QCA.iter_entry(), total=n_entries, desc="IC generation", ncols=80
        ):

            self._prim[entry.payload] = {}
            for ic_type, measure_function in vtable.items():
                # the geometry measurements
                ic = measure_function(ic_type)
                ic.source.source = QCA
                ic.verbose = False
                ic.apply()

                # if ic_type == "Bonds":
                #     for mol in ic.db:
                #         ic.db[mol] = {
                #             idx: [x * offsb.tools.const.bohr2angstrom for x in vals]
                #             for idx, vals in ic.db[mol].items()
                #         }
                self._ic[ic_type] = ic

                # for mol in QCA.node_iter_depth_first(entry, select="Molecule"):
                #     ic_data = ic.db[mol.payload]
                #     for aidx, vals in ic_data.items():
                #             # prim = prim_to_graph[param_name[0]].from_string_list(primitives[aidx])
                #         if ic_type == "Bonds":
                #             vals = [x*offsb.tools.const.bohr2angstrom for x in vals]
                #         # param_data[param_name][prim]["measure"].extend(vals)
                #         self.db[mol.payload]['data']['measure'][aidx] = vals

            # need to unmap... sigh
            smi = QCA.db[entry.payload]["data"].attributes[
                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
            ]
            rdmol = offsb.rdutil.mol.build_from_smiles(smi)
            atom_map = offsb.rdutil.mol.atom_map(rdmol)
            map_inv = offsb.rdutil.mol.atom_map_invert(atom_map)

            primitives = self._to.db[entry.payload]["data"]

            for hessian_node in QCA.node_iter_depth_first(entry, select="Hessian"):
                mol = QCA[hessian_node.parent]

                with tempfile.NamedTemporaryFile(mode="wt") as f:
                    offsb.qcarchive.qcmol_to_xyz(
                        QCA.db[mol.payload]["data"], fnm=f.name
                    )
                    gmol = geometric.molecule.Molecule(f.name, ftype="xyz")
                with open("out.xyz", mode="wt") as f:
                    offsb.qcarchive.qcmol_to_xyz(QCA.db[mol.payload]["data"], fd=f)

                xyz = QCA.db[mol.payload]["data"].geometry
                hess = QCA.db[hessian_node.payload]["data"].return_result
                grad = np.array(
                    QCA.db[hessian_node.payload]["data"].extras["qcvars"][
                        "CURRENT GRADIENT"
                    ]
                )

                ic_prims = geometric.internal.PrimitiveInternalCoordinates(
                    gmol,
                    build=True,
                    connect=True,
                    addcart=False,
                    constraints=None,
                    cvals=None,
                )

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
                            prim = prim_to_graph[param_name[0]].from_string_list(
                                primitives[unmapped]
                            )
                            self._prim[entry.payload][unmapped] = prim
                        except Exception as e:
                            breakpoint()

                ic_hess = ic_prims.calcHess(xyz, grad, hess)

                ic_data = {}
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
                    # ic_data[key] = val

                # for aidx, vals in ic_data.items():
                #     if ic_type == "Bonds":
                #         vals = vals / (offsb.tools.const.bohr2angstrom**2)
                #     vals *= offsb.tools.const.hartree2kcalmol

                # param_vals.append(vals)

                # if len(param_vals) > 0:
                #     new_val = np.mean(param_vals)
                #     # until we resolve generating an IC for each FF match,
                #     # we skip setting params which the ICs don't make
                #     print("average val is", new_val)
                #     self._set_parameter_force(param_name, new_val)

        # self.to_smirnoff_xml("tmp.offxml", verbose=True)
        # return param_data

    ###

    def _combine_optimization_data_v2(self, QCA=None):
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

        self.to_smirnoff_xml("tmp.offxml", verbose=False)
        labeler = qcasb.assign_labels_from_openff("tmp.offxml", "tmp.offxml")

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
                                # if aidx not in self._prim[entry.payload]:
                                #     breakpoint()
                                #     print("HI!")

                                key = self._prim[entry.payload][aidx]
                                # if aidx not in self._fc[mol.payload]:
                                #     breakpoint()
                                #     print("HI!")

                                force_vals = self._fc[mol.payload][aidx]
                                if key not in param_data[param_name]:
                                    param_data[param_name][key] = {
                                        "measure": [],
                                        "force": [],
                                    }

                                param_data[param_name][key]["measure"].extend(vals)
                                param_data[param_name][key]["force"].append(force_vals)

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

        self.to_smirnoff_xml("tmp.offxml", verbose=False)
        labeler = qcasb.assign_labels_from_openff("tmp.offxml", "tmp.offxml")

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
                                # if aidx not in self._prim[entry.payload]:
                                #     breakpoint()
                                #     print("HI!")

                                key = self._prim[entry.payload][aidx]
                                # if aidx not in self._fc[mol.payload]:
                                #     breakpoint()
                                #     print("HI!")

                                force_vals = self._fc[mol.payload][aidx]
                                if key not in param_data[param_name]:
                                    param_data[param_name][key] = {
                                        "measure": [],
                                        "force": [],
                                    }

                                param_data[param_name][key]["measure"].extend(vals)
                                param_data[param_name][key]["force"].append(force_vals)

                    # for hessian_node in QCA.node_iter_depth_first(entry, select="Hessian"):
                    #     mol = QCA[hessian_node.parent]
                    #         self.db[mol.payload]['data']['force'][key] = vals

                    #     with tempfile.NamedTemporaryFile(mode='wt') as f:
                    #         offsb.qcarchive.qcmol_to_xyz(QCA.db[mol.payload]['data'], fnm=f.name)
                    #         gmol = geometric.molecule.Molecule(f.name, ftype='xyz')
                    #     with open("out.xyz", mode='wt') as f:
                    #         offsb.qcarchive.qcmol_to_xyz(QCA.db[mol.payload]['data'], fd=f)

                    #     xyz = QCA.db[mol.payload]['data'].geometry
                    #     hess = QCA.db[hessian_node.payload]['data'].return_result
                    #     grad = np.array(QCA.db[hessian_node.payload]['data'].extras["qcvars"]["CURRENT GRADIENT"])

        # # IC = CoordClass(geometric_mol(mol_xyz_fname), build=True,
        # #                         connect=connect, addcart=addcart, constraints=Cons,
        # #                         cvals=CVals[0] if CVals is not None else None )
        #     ic_prims = geometric.internal.PrimitiveInternalCoordinates(gmol, build=True, connect=True, addcart=False, constraints=None, cvals=None)

        #     ic_hess = ic_prims.calcHess(xyz, grad, hess)

        #     ic_data = {}
        #     # eigs = np.linalg.eigvalsh(ic_hess)
        #     # s = np.argsort(np.diag(ic_hess))
        #     # force_vals = eigs
        #     force_vals = np.diag(ic_hess)
        #     # ic_vals = [ic_prims.Internals[i] for i in s]
        #     ic_vals = ic_prims.Internals
        #     for aidx, val in zip(ic_vals, force_vals):
        #         key = tuple(map(lambda x: map_inv[int(x) - 1], str(aidx).split()[1].split("-")))

        #         if ic_type == "ImproperTorsions" and len(key) == 4:
        #             key = ImproperDict.key_transform(key)
        #         else:
        #             key = ValenceDict.key_transform(key)

        #         ic_data[key] = val

        #     # ic_data = ic.db[mol.payload]
        #     # labels = {transform(tuple([atom_map[x]-1 for x in i])): v for i, v in labels.items()}
        #     for aidx, vals in ic_data.items():
        #         if aidx in labels and labels[aidx] == param_name:
        #             # print("fc for key", aidx, "is", vals)
        #             prim = prim_to_graph[param_name[0]].from_string_list(primitives[aidx])
        #             if prim not in param_data[param_name]:
        #                 param_data[param_name][prim] = {"measure": [], "force": []}
        #             if ic_type == "Bonds":
        #                 vals = vals / (offsb.tools.const.bohr2angstrom**2)
        #             vals *= offsb.tools.const.hartree2kcalmol
        #             param_data[param_name][prim]["force"].append(vals)
        #             # param_data[param_name][prim].append(vals)

        #             # param_vals.append(vals)

        # # if len(param_vals) > 0:
        # #     new_val = np.mean(param_vals)
        # #     # until we resolve generating an IC for each FF match,
        # #     # we skip setting params which the ICs don't make
        # #     print("average val is", new_val)
        # #     self._set_parameter_force(param_name, new_val)

        # self.to_smirnoff_xml("tmp.offxml", verbose=True)
        return param_data

    ###

    def _combine_optimization_data(self):

        QCA = self._po.source.source
        param_names = self._po._forcefield.plist

        # smi_to_label = self._po._setup.labeler.db["ROOT"]["data"]
        # smi_to_label = {
        #     k: v["smirks"] for keys in smi_to_label.values() for k, v in keys.items()
        # }
        # smi_to_label = {v: k for k, v in smi_to_label.items()}

        param_labels = [param.split("/")[-1] for param in param_names]

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

            if entry.payload not in self._po._setup.labeler.db:
                continue
            labels = self._po._setup.labeler.db[entry.payload]["data"]
            labels = {k: v for keys in labels.values() for k, v in keys.items()}

            if entry.payload not in self._to.db:
                continue
            primitives = self._to.db[entry.payload]["data"]

            print("    {:8d}/{:8d} : {}".format(i + 1, n_entries, entry))

            for molecule in QCA.node_iter_depth_first(entry, select="Molecule"):
                mol_id = molecule.payload
                obj = self._po.db[mol_id]["data"]

                # IC keys (not vdW)
                for key in [k for k in obj if type(k) == tuple and k in labels]:
                    matched_params = [
                        val
                        for i, val in enumerate(obj[key]["dV"])
                        if labels[key] == param_labels[i]
                    ]
                    if len(matched_params) == 0:
                        # if we have no matches, then likely we are not trying to
                        # fit to it, so we can safely skip
                        continue

                    lbl = labels[key]
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

    def _optimize_type_iteration(
        self,
        ignore_bits=None,
        use_gradients=True,
        split_strategy="spatial_reference",
        ignore_parameters=None,
    ):

        jobtype = "GRADIENT"

        grad_new = 0
        grad = 0
        i = 0
        node = None
        if ignore_bits is None:
            ignore_bits = {}

        if ignore_parameters is None:
            ignore_parameters = []

        if use_gradients:
            print("Running reference gradient calculation...")

            # skip this since there was an optimization done immediately prior
            newff_name = "newFF.offxml"
            self.to_smirnoff_xml(newff_name, verbose=False)
            # self._po._options["forcefield"] = [newff_name]

            self._po._setup.ff_fname = newff_name
            self._po.ff_fname = newff_name

            self._po._init = False
            if self.trust0 is None:
                self.trust0 = self._po._options.get("trust0", 0.1)
            if self.finite_difference_h is None:
                self.finite_difference_h = self._po._options.get(
                    "finite_difference_h", 0.01
                )
            print("Setting trust0 to", self.trust0)
            print("Setting finite_difference_h to", self.finite_difference_h)
            while True:
                try:
                    self._po.load_options(
                        options_override={
                            "trust0": self.trust0,
                            "finite_difference_h": self.finite_difference_h,
                        }
                    )
                    self._po.apply(jobtype=jobtype)

                    # make sure to reset here so that the changes are picked
                    # up in the single points when splitting via gradients
                    self.to_smirnoff_xml(newff_name, verbose=False)
                    self._po._setup.ff_fname = newff_name
                    self._po.ff_fname = newff_name
                    self._po._init = False
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
                        "Reference gradient failed; reducing trust radius to",
                        self.trust0,
                        "finite_difference_h",
                        self.finite_difference_h,
                    )
                    self._po._options["trust0"] = self.trust0
                    self._po._options["finite_difference_h"] = self.finite_difference_h
                    mintrust = self._po._options["mintrust"]
                    if self.trust0 < mintrust:
                        print("Reference gradient calculation failed; cannot continue!")
                        return None, np.inf

            obj = self._po.X

            grad_scale = 1.0

            grad = self._po.G
            best = [None, grad * grad_scale, None]
        else:
            best = [None, np.inf, None]

        self._combine_reference_data()

        while True:
            if use_gradients:
                print("\n\nMicroiter", i)
            i += 1

            success = True
            olddb = self._po.db.copy()

            eps = 1.0

            if split_strategy == "spatial_reference":
                key = "measure"
                mode = "mean_difference"
                eps = 3.0
                param_data = self._combine_reference_data()
            elif split_strategy == "force_reference":
                key = "force"
                mode = "mean_difference"
                eps = 3.0
                param_data = self._combine_reference_data()
            elif use_gradients:
                # this number is highly dependent on other parameters
                # allow everything since we only accept if it lowers the grad
                eps = 0.0
                key = None
                mode = "sum_difference"
                # mode = "mag_difference"
                param_data, all_data = self._combine_optimization_data()

            if ignore_parameters is not None:
                param_data = {
                    k: v for k, v in param_data.items() if k[0] not in ignore_parameters
                }
            # print("Finding new split...")
            # print("Ignore bits are")
            # for ignore, grads in ignore_bits.items():
            #     print(grads, ignore)
            node = self._find_next_split(
                param_data, key=key, ignore_bits=ignore_bits, mode=mode, eps=eps
            )
            print("Split is", node)

            if node is None:
                break

            if use_gradients:
                # self._po._options["forcefield"] = [newff_name]
                print("Calculating new gradient with split param")

                # would be nice to get the previous settings

                self._po.logger.setLevel(logging.ERROR)
                try:
                    self._po.load_options(
                        options_override={
                            "trust0": self.trust0,
                            "finite_difference_h": self.finite_difference_h,
                        }
                    )
                    newff_name = "newFF.offxml"
                    self.to_smirnoff_xml(newff_name, verbose=False)
                    # self._po._options["forcefield"] = [newff_name]

                    self._po._setup.ff_fname = newff_name
                    self._po.ff_fname = newff_name

                    self._po._init = False
                    self._po.apply(jobtype=jobtype)
                except RuntimeError:
                    print("Gradient failed for this split; skipping")

                    success = False

                self._po.logger.setLevel(self.logger.getEffectiveLevel())

                if success:
                    grad_new = self._po.G
                    print(
                        "grad_new",
                        grad_new,
                        "grad",
                        grad,
                        "grad_new < grad*scale?",
                        grad_new < grad * grad_scale,
                    )

                    # current mode: take the best looking split
                    # best = [node, grad_new, node.parent, self.db[node.payload]]
                    # break

                    # hard core mode: only take the one with the smaller grad
                    if grad_new < best[1]:
                        best = [
                            node.copy(),
                            grad_new,
                            node.parent,
                            self.db[node.payload],
                        ]

                # remove the previous term if it exists
                print("Remove parameter", node)
                self[node.parent].children.remove(node.index)
                self.node_index.pop(node.index)
                self.db.pop(node.payload)

                newff_name = "newFF.offxml"
                self.to_smirnoff_xml(newff_name, verbose=False)
                # self._po._options["forcefield"] = [newff_name]

                self._po._setup.ff_fname = newff_name
                self._po.ff_fname = newff_name
                self._po._init = False

                # if there is an exception, the po will have no data

                self._po.db = olddb
            else:
                best = [node, np.inf, node.parent, self.db[node.payload]]
                break

        if best[0] is not None:
            # only readd if we did a complete scan, since we terminate that case
            # with no new node, and the best has to be read
            # if we break early, the node is already there
            # self.add(best[2], best[0])
            # self.db[best[0].payload] = best[3]

            print("Best split parameter")
            print(best[0])
            print("Best split gradient", best[1])

        newff_name = "newFF.offxml"
        self.to_smirnoff_xml(newff_name, verbose=False)

        self._po._setup.ff_fname = newff_name
        self._po._init = False
        return best[0], best[1]

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
        self, QCA, ignore_parameters=None, only_parameters=None, report_only=False
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

        param_data = self._combine_reference_data(QCA=QCA, prims=False)

        for param_name, param_types in param_data.items():
            if param_name[0] in ignore_parameters:
                continue
            if only_parameters is not None and param_name in only_parameters:
                param_types = list(param_types.values())
                for p_type, fn in {
                    "measure": self._set_parameter_spatial,
                    "force": self._set_parameter_force,
                }.items():
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
                self.db[param.payload]["data"]["parameter"] = ff_params[param.payload]

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
                    node, grad_split = self._optimize_type_iteration(
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
                    ignore_bits[bit] = []
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
                    node, grad_split = self._optimize_type_iteration(
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
                    ignore_bits[bit] = []
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
        self.to_smirnoff_xml(newff_name, verbose=True)
        self.to_pickle()

    def optimize(
        self,
        optimize_types=True,
        optimize_parameters=False,
        optimize_during_typing=True,
        optimize_initial=True,
    ):

        # self._to.apply()

        newff_name = "input.offxml"
        self.to_smirnoff_xml(newff_name, verbose=False)
        self._po._setup.ff_fname = newff_name
        self._po.ff_fname = newff_name
        self._po._init = False

        self.trust0 = 0.25
        self.finite_difference_h = 0.01

        jobtype = "OPTIMIZE"
        if not optimize_initial:
            jobtype = "GRADIENT"

        if optimize_types:
            print("Performing initial FF fit...")
            while True:
                try:
                    self._po.load_options(
                        options_override={
                            "trust0": self.trust0,
                            "finite_difference_h": self.finite_difference_h,
                        }
                    )
                    self._po.apply(jobtype=jobtype)
                    break
                except RuntimeError:
                    self._bump_zero_parameters(1e-3, names="epsilon")
                    self.to_smirnoff_xml(newff_name, verbose=False)
                    self._po._setup.ff_fname = newff_name
                    self._po.ff_fname = newff_name
                    self._po._init = False
                    self.trust0 = self._po._options["trust0"] / 2.0
                    print(
                        "Initial optimization failed; reducing trust radius to",
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
                        return
            self.finite_difference_h = self._po._options["finite_difference_h"]

            print("Initial objective after first fit:", self._po.X)
            obj = self._po.X
            ref = obj
            initial = obj
            ref_grad = self._po.G

            newff_name = "newFF.offxml"
            self.load_new_parameters(self._po.new_ff)
            self.to_pickle()
            self.to_smirnoff_xml(newff_name, verbose=False)
            self._po._setup.ff_fname = newff_name
            self._po._init = False

            ignore_bits = {}
            ignore_bits_optimized = {}

            # self._plot_gradients(fname_prefix="0")

            # this ref is prior to splitting
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
                        node, grad_split = self._optimize_type_iteration(
                            ignore_bits=ignore_bits,
                            split_strategy="gradient",
                            use_gradients=True,
                        )

                        if node is None:
                            print("No new parameter split, done!")
                            break

                        if optimize_during_typing:
                            print("Performing micro optimization for new split")
                            newff_name = "optimize.offxml"
                            self.to_smirnoff_xml(newff_name, verbose=False)
                            self._po._setup.ff_fname = newff_name
                            self._po.ff_fname = newff_name
                            self._po._init = False
                            self._po.apply(jobtype="OPTIMIZE")
                            obj = self._po.X
                            print("Objective after minimization:", self._po.X)
                            self.load_new_parameters(self._po.new_ff)
                            self.to_smirnoff_xml(newff_name, verbose=False)
                            self._po._setup.ff_fname = newff_name
                            self._po.ff_fname = newff_name
                            self._po._init = False
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
                        ignore_bits[bit] = []
                        # the db is in a null state, and causes failures
                        # TODO allow skipping an optimization

                    print("New objective:", obj, "Delta is", obj - ref)
                    print(
                        "New objective gradient from split:",
                        grad_split,
                        "Previous",
                        ref_grad,
                        "Delta is",
                        grad_split - ref_grad,
                    )
                    grad_scale_factor = 1.0
                    if (obj > ref and optimize_during_typing) or (
                        grad_split > ref_grad * grad_scale_factor
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
                            if bit not in ignore_bits:
                                ignore_bits[bit] = []
                            ignore_bits_optimized[bit] = ignore_bits[bit]
                        except KeyError as e:
                            # probably a torsion or something that caused
                            # the bit calculation to fail
                            print(e)

                        print("Keeping ignore bits for next iteration:")
                        for bit, bit_grad in ignore_bits_optimized.items():
                            print(bit_grad, bit)

                        # ignore_bits will keep track of things that didn't work
                        # do this after keeping the ignore bits since we access
                        # the info
                        self[node.parent].children.remove(node.index)
                        self.node_index.pop(node.index)
                        self.db.pop(node.payload)

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
                        self.to_smirnoff_xml("newFF" + str(i) + ".accept.offxml")
                        self.to_pickle()
            print("Final objective is", obj, "initial was", initial)
            print("Total drop is", obj - initial)
            self.to_smirnoff_xml(newff_name, verbose=True)
            self.to_pickle()
            print("Splitting done; performing final optimization")

            while True:
                try:
                    self._po.load_options(
                        options_override={
                            "trust0": self.trust0,
                            "finite_difference_h": self.finite_difference_h,
                        }
                    )
                    self._po.apply(jobtype="OPTIMIZE")
                    break
                except RuntimeError:
                    self._bump_zero_parameters(1e-1, names="epsilon")
                    self.to_smirnoff_xml(newff_name, verbose=False)
                    self._po._setup.ff_fname = newff_name
                    self._po.ff_fname = newff_name
                    self._po._init = False
                    self.trust0 = self._po._options["trust0"] / 2.0
                    print(
                        "Initial optimization failed; reducing trust radius to",
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
                        return

        if optimize_parameters:
            if not optimize_types:
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

            while True:
                try:
                    self._po.apply(jobtype="OPTIMIZE")
                    obj = self._po.X
                    self.trust0 = self._po._options["trust0"]
                    break
                except RuntimeError as e:
                    self.trust0 = self._po._options["trust0"] / 2.0
                    print("Optimization failed; reducing trust radius to", self.trust0)
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
            print("Optimized objective is", obj, "initial was", initial)
            print("Total drop is", obj - initial)
            self.to_smirnoff_xml(newff_name, verbose=True, renumber=False)
            self.to_pickle()

    @classmethod
    def from_smirnoff(self, input):
        pass

    def set_physical_optimizer(self, obj):
        self._po = obj

    def set_smarts_generator(self, obj):
        self._to = obj

    # def _combine_reference_data(self):
    #     """
    #     Collect the expected parameter values directly from the reference QM
    #     molecules
    #     """

    #     self._calculate_ic_force_constants()

    #     import geometric.internal
    #     import geometric.molecule
    #     QCA = self._po.source.source
    #     n_entries = len(list(QCA.iter_entry()))

    #     param_data = {}
    #     hessian_data = {}

    #     QCA = self._po.source.source
    #     # need this for measuring geometry
    #     # should only need to do it once
    #     qcasb = offsb.ui.qcasb.QCArchiveSpellBook(QCA=QCA)

    #     vtable = {
    #         "Bonds": qcasb.measure_bonds,
    #         "Angles": qcasb.measure_angles,
    #         "ImproperTorsions": qcasb.measure_outofplanes,
    #         "ProperTorsions": qcasb.measure_dihedrals,
    #     }

    #     self.to_smirnoff_xml("tmp.offxml", verbose=False)
    #     labeler = qcasb.assign_labels_from_openff("tmp.offxml", self.ffname)

    #     n_entries = len(list(QCA.iter_entry()))
    #     for entry in tqdm.tqdm(QCA.iter_entry(), total=n_entries, desc="IC generation", ncols=80):
    #         # need to unmap... sigh
    #         smi = QCA.db[entry.payload]['data'].attributes['canonical_isomeric_explicit_hydrogen_mapped_smiles']
    #         rdmol = offsb.rdutil.mol.build_from_smiles(smi)
    #         atom_map = offsb.rdutil.mol.atom_map(rdmol)
    #         map_inv = offsb.rdutil.mol.atom_map_invert(atom_map)
    #         labels = {aidx: val for ic_type in vtable for aidx, val in  labeler.db[entry.payload]["data"][ic_type].items()}
    #         primitives = self._to.db[entry.payload]["data"]

    #         for ic_type, measure_function in vtable.items():

    #             # the geometry measurements
    #             ic = measure_function(ic_type)
    #             ic.source.source = QCA
    #             ic.verbose = False
    #             ic.apply()
    #     for entry in tqdm.tqdm(QCA.iter_entry(), total=n_entries, desc="IC generation", ncols=80):
    #         for hessian_node in QCA.node_iter_depth_first(entry, select="Hessian"):
    #             mol = QCA[hessian_node.parent]

    #             with tempfile.NamedTemporaryFile(mode='wt') as f:
    #                 offsb.qcarchive.qcmol_to_xyz(QCA.db[mol.payload]['data'], fnm=f.name)
    #                 gmol = geometric.molecule.Molecule(f.name, ftype='xyz')
    #             with open("out.xyz", mode='wt') as f:
    #                 offsb.qcarchive.qcmol_to_xyz(QCA.db[mol.payload]['data'], fd=f)

    #             xyz = QCA.db[mol.payload]['data'].geometry
    #             hess = QCA.db[hessian_node.payload]['data'].return_result
    #             grad = np.array(QCA.db[hessian_node.payload]['data'].extras["qcvars"]["CURRENT GRADIENT"])


# # IC = CoordClass(geometric_mol(mol_xyz_fname), build=True,
# #                         connect=connect, addcart=addcart, constraints=Cons,
# #                         cvals=CVals[0] if CVals is not None else None )
#             ic_prims = geometric.internal.PrimitiveInternalCoordinates(gmol, build=True, connect=True, addcart=False, constraints=None, cvals=None)

#             ic_hess = ic_prims.calcHess(xyz, grad, hess)

#             ic_data = {}
#             # eigs = np.linalg.eigvalsh(ic_hess)
#             # s = np.argsort(np.diag(ic_hess))
#             # force_vals = eigs
#             force_vals = np.diag(ic_hess)
#             # ic_vals = [ic_prims.Internals[i] for i in s]
#             ic_vals = ic_prims.Internals
#             for aidx, val in zip(ic_vals, force_vals):
#                 key = tuple(map(lambda x: map_inv[int(x) - 1], str(aidx).split()[1].split("-")))

#                 if ic_type == "ImproperTorsions" and len(key) == 4:
#                     key = ImproperDict.key_transform(key)
#                 else:
#                     key = ValenceDict.key_transform(key)

#                 ic_data[key] = val


#             # ic_data = ic.db[mol.payload]
#             # labels = {transform(tuple([atom_map[x]-1 for x in i])): v for i, v in labels.items()}
#             for aidx, vals in ic_data.items():
#                 if aidx in labels and labels[aidx] == param_name:
#                     # print("fc for key", aidx, "is", vals)
#                     prim = prim_to_graph[param_name[0]].from_string_list(primitives[aidx])
#                     if prim not in param_data:
#                         param_data[prim] = {"measure": [], "force": []}
#                     if ic_type == "Bonds":
#                         vals = vals / (offsb.tools.const.bohr2angstrom**2)
#                     vals *= offsb.tools.const.hartree2kcalmol
#                     param_data[prim]["force"].append(vals)
#                     # param_data[param_name][prim].append(vals)
