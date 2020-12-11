#!/usr/bin/env python

import copy
import functools
import logging
import os
import pprint

import numpy as np
import simtk.unit
import tqdm

import offsb.chem.types
import offsb.op.chemper
import offsb.op.forcebalance
import offsb.treedi.node
import offsb.treedi.tree
from offsb.treedi.tree import DEFAULT_DB
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.typing.engines.smirnoff.parameters import (
    AngleHandler, BondHandler, ImproperTorsionHandler, ParameterList,
    ProperTorsionHandler, vdWHandler)

VDW_DENOM = 10.0


class ChemicalSpace(offsb.treedi.tree.Tree):
    def __init__(self, obj, root_payload=None, node_index=None, db=None, payload=None):
        print("Building ChemicalSpace")
        if isinstance(obj, str):
            super().__init__(
                obj,
                root_payload=root_payload,
                node_index=node_index,
                db=db,
                payload=payload,
            )

            self.ffname = obj

            # default try to make new types for these handlers
            self.parameterize_handlers = [
                "vdW",
                "Bonds",
                "Angles",
                "ImproperTorsions",
                "ProperTorsions",
            ]

            # root = self.root()

            # # This sets up a "blank" ChemSpace, one term that covers everything
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
            # this is the actual useful bit; puts them in the correct SMIRNOFF order
            for i, param_node in enumerate(self.node_iter_breadth_first(ph_node), 1):
                param = copy.deepcopy(self.db[param_node.payload]["data"]["parameter"])
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
                param = copy.deepcopy(self.db[param_node.payload]["data"]["parameter"])

                if renumber:
                    param.id = num_map[param.id]
                if self.node_depth(param_node) > 1:
                    print("->", end="")
                print("    " * self.node_depth(param_node), param_node)
                ff_param = copy.deepcopy(
                    self.db[param_node.payload]["data"]["parameter"]
                )
                ff_param.smirks = ""
                print(
                    "  " + "      " * (1 + self.node_depth(param_node)),
                    self.db[param_node.payload]["data"]["group"],
                )
                for k, v in ff_param.__dict__.items():
                    k = str(k).lstrip("_")
                    if any([k.startswith(x) for x in ["cosmetic", "smirks"]]):
                        continue

                    # list of Quantities...
                    if issubclass(type(v), list):
                        v = " ".join(["{}".format(x.__str__()) for x in v])

                    print(
                        "  " + "      " * (1 + self.node_depth(param_node)),
                        "{:12s} : {}".format(k, v),
                    )

        return ff

    def to_smirnoff_xml(self, output, verbose=True, renumber=False):

        ff = self.to_smirnoff(verbose=verbose, renumber=renumber)
        ff.to_file(output)

    @classmethod
    def from_smirnoff_xml(cls, input):

        """
        add parameter ids to db
        each db entry has a ff param (smirks ignored) and a group
        """

        ff = ForceField(input, allow_cosmetic_attributes=True)
        cls = cls(input, root_payload=ff)

        return cls

    @classmethod
    def default_from_smirnoff_xml(cls, input, name=None):

        """
        add parameter ids to db
        each db entry has a ff param (smirks ignored) and a group
        """

        ff = ForceField(input, allow_cosmetic_attributes=True)

        name = input if name is None else name
        cls = cls(name, root_payload=ff)
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
                    epsilon="0.5 * kilocalorie/mole",
                    rmin_half="1.2 * angstrom",
                    smirks=smirks,
                    id="n1",
                )

                ff.get_parameter_handler(pl_name)._parameters = ParameterList()
                ff.get_parameter_handler(pl_name).add_parameter(param_dict)

                param = vdWHandler.vdWType(**param_dict)

                param_name = param.id

                pnode = offsb.treedi.node.Node(
                    name=str(type(param)), payload=param_name
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
                    name=str(type(param)), payload=param_name
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
                    name=str(type(param)), payload=param_name
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
                param_dict = dict(
                    periodicity=[1, 2, 3],
                    k=[
                        "0 * kilocalorie/mole",
                        "0 * kilocalorie/mole",
                        "0 * kilocalorie/mole",
                    ],
                    phase=["0.0 * degree", "0 * degree", "0 * degree"],
                    smirks=smirks,
                    id="t1",
                    idivf=list([1.0] * 3),
                )
                param = ProperTorsionHandler.ProperTorsionType(**param_dict)
                ff.get_parameter_handler(pl_name)._parameters = ParameterList()
                ff.get_parameter_handler(pl_name).add_parameter(param_dict)

                param_name = param.id

                pnode = offsb.treedi.node.Node(
                    name=str(type(param)), payload=param_name
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
                    k=["-0.0 * kilocalorie/mole"],
                    phase=["-0.000 * degree"],
                    smirks=smirks,
                    id="i1",
                    idivf=[1.0],
                )
                param = ImproperTorsionHandler.ImproperTorsionType(**param_dict)
                ff.get_parameter_handler(pl_name)._parameters = ParameterList()
                ff.get_parameter_handler(pl_name).add_parameter(param_dict)

                param_name = param.id

                pnode = offsb.treedi.node.Node(
                    name=str(type(param)), payload=param_name
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

        new_param = copy.deepcopy(param)

        param_name = label[0] + str(np.random.randint(1000, 9999))

        pnode = offsb.treedi.node.Node(name=str(type(param)), payload=param_name)

        self.add(node.index, pnode)

        # param.smirks = group.drop(child).to_smarts()
        param.smirks = group.to_smarts(tag=True)
        new_param.smirks = child.to_smarts(tag=True)
        new_param.id = param_name

        self.db[param_name] = DEFAULT_DB(
            {"data": {"parameter": new_param, "group": child}}
        )
        return pnode

    def _find_next_split(self, param_data, ignore_bits=None, procedure="norm"):

        bit_gradients = []
        if ignore_bits is None:
            ignore_bits = {}

        handlers = [
            self[x]
            for x in self.root().children
            if self[x].payload in self.parameterize_handlers
        ]
        for node in self.node_iter_breadth_first(handlers):
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
                    breakpoint()
            except Exception as e:
                print(e)
                breakpoint()

            # Now that we know all params are covered by this FF param,
            # sanitize the group by making it match the FF group, so things
            # stay tidy
            # group = group & self.db[lbl]["data"]["group"]

            # assert (group - self.db[lbl]['data']['group']).reduce() == 0

            # this indicates what this smirks covers, but we don't represent in the
            # current data
            uncovered = self.db[lbl]["data"]["group"] - group

            verbose = False

            if verbose:
                if uncovered.reduce() == 0:
                    print("This dataset completely covers", lbl, ". Nice!")
                else:
                    print("This dataset does not cover this information:")
                    print(uncovered)

            # iterate bits that we cover (AND them just to be careful)
            # group = group & self.db[lbl]["data"]["group"]
            if verbose:
                print("\nContinuing with this information:")
                print(group)

            for bit in group:
                if verbose:
                    print("Scanning for bit", bit)
                if bit in ignore_bits:
                    if verbose:
                        print("Ignoring since it is in the ignore list")
                    continue

                # for ignore in ignore_bits:
                #     if type(ignore) == type(bit) and :
                #         if verbose:
                #             print("Ignoring since it is in the ignore list. Matches this ignore:")
                #             print(ignore)
                #         continue
                ys_bit = []
                no_bit = []
                for prim, dat in param_data[lbl].items():
                    # pdb.set_trace()
                    if verbose:
                        print("Considering prim", prim, end=" ")
                    if bit in prim:
                        if verbose:
                            print("yes")
                        ys_bit.extend(dat)
                    else:
                        if verbose:
                            print("no")
                        no_bit.extend(dat)
                if verbose:
                    print("    Has bit:", end=" ")
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
                    print("    No  bit:", end=" ")
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
                    bit_gradients.append(
                        [
                            lbl,
                            bit.copy(),
                            np.sum(no_bit, axis=0) - np.sum(ys_bit, axis=0),
                        ]
                    )
                else:
                    pass
                    if verbose:
                        print("None")

        if len(bit_gradients) == 0:
            return None
        bit_gradients = sorted(
            bit_gradients, key=lambda x: np.max(np.abs(x[2])), reverse=True
        )
        split_bit = bit_gradients[0][1]
        if split_bit not in ignore_bits:
            ignore_bits[split_bit] = bit_gradients[0][2]
            # child = group - split_bit
            lbl = bit_gradients[0][0]
            print(
                "Splitting",
                lbl,
                "\n",
                self.db[lbl]["data"]["group"],
                "using\n",
                split_bit,
                "vals",
                bit_gradients[0][2],
            )
            # print("The parent is")
            # print(group.drop(child))
            # print("Smarts is")
            # print(group.drop(child).to_smarts())
            # print("The child is")
            # print(child)
            # print("Smarts is")
            # print(child.to_smarts())

            node = self.split(lbl, split_bit)
            return node
        else:
            return None

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
        #     param_labels = [smi_to_label[param.split("/")[-1]] for param in param_names]
        # except KeyError as e:
        #     print("KeyError! Dropping to a debugger")
        #     breakpoint()
        #     print("KeyError: these keys were from FB")
        #     print([param.split("/")[-1] for param in param_names])
        #     print("The FF keys are:")
        #     print(smi_to_label)

        prim_to_group = {
            "n": offsb.chem.types.AtomType,
            "b": offsb.chem.types.BondGroup,
            "a": offsb.chem.types.AngleGroup,
            "i": offsb.chem.types.OutOfPlaneGroup,
            "t": offsb.chem.types.TorsionGroup,
        }
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
                    prim = prim_to_group[lbl[0]].from_string_list(primitives[key])
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
                    prim = prim_to_group[lbl[0]].from_string(primitives[key][0])
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
                            breakpoint()
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

    def _optimize_type_iteration(self, ignore_bits=None):

        jobtype = "GRADIENT"

        grad_new = 0
        grad = 0
        i = 0
        node = None
        if ignore_bits is None:
            ignore_bits = {}

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
            self.finite_difference_h = self._po._options.get("finite_difference_h", .01)
        print("Setting trust0 to", self.trust0)
        print("Setting finite_difference_h to", self.finite_difference_h)
        self._po.load_options(options_override={"trust0": self.trust0, "finite_difference_h": self.finite_difference_h})
        while True:
            try:
                self._po.apply(jobtype=jobtype)
                break
            except RuntimeError:
                # self._bump_zero_parameters(1e-3, names="epsilon")
                self.trust0 = self._po._options["trust0"] / 2.0
                self.finite_difference_h = self._po._options["finite_difference_h"] / 2.0
                print(
                    "Reference gradient failed; reducing trust radius to", self.trust0, "finite_difference_h", self.finite_difference_h
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

        while True:
            print("Microiter", i)
            i += 1

            # remove the previous term if it exists
            if node is not None:
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

            # to run again with a new ff, just do:
            # self._po._options['forcefield'] = newff
            # where newff is a filename

            param_data, all_data = self._combine_optimization_data()

            print("Finding new split...")
            print("Ignore bits are")
            for ignore, grads in ignore_bits.items():
                print(grads, ignore)
            node = self._find_next_split(param_data, ignore_bits=ignore_bits)
            print("Split is", node)

            if node is None:
                break

            newff_name = "newFF.offxml"
            self.to_smirnoff_xml(newff_name, verbose=False)
            # self._po._options["forcefield"] = [newff_name]

            self._po._setup.ff_fname = newff_name
            self._po.ff_fname = newff_name
            self._po._init = False

            # self._po._options["forcefield"] = [newff_name]
            print("Calculating new gradient with split param")

            # would be nice to get the previous settings

            self._po.logger.setLevel(logging.ERROR)
            try:
                self._po.apply(jobtype=jobtype)
            except RuntimeError:
                print("Gradient failed for this split; skipping")
                continue
            self._po.logger.setLevel(self.logger.getEffectiveLevel())
            grad_new = self._po.G
            # np.linalg.norm(
            #     np.vstack(
            #         [
            #             v["dV"].sum(axis=1)
            #             for k, v in self._po._objective.ObjDict.items()
            #             if hasattr(v, "__iter__") and "dV" in v and v["dV"] is not None
            #         ]
            #     ).sum(axis=0)
            # )
            print(
                "grad_new",
                grad_new,
                "grad",
                grad,
                "grad_new < grad*scale?",
                grad_new < grad * grad_scale,
            )
            # if grad_new < best[1]:
            #     best = [node.copy(), grad_new, node.parent, self.db[node.payload]]
            best = [node, grad_new, node.parent, self.db[node.payload]]
            break

        if best[0] is not None:
            # only readd if we did a complete scan, since we terminate that case
            # with no new node, and the best has to be readded
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

    def load_new_parameters(self, new_ff):

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
                        for i,v in enumerate(vals):
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


    def optimize(
        self,
        optimize_types=True,
        optimize_parameters=False,
        optimize_during_typing=True,
    ):

        self._to.apply()

        newff_name = "input.offxml"
        self.to_smirnoff_xml(newff_name, verbose=True)
        self._po._setup.ff_fname = newff_name
        self._po.ff_fname = newff_name
        self._po._init = False

        self.trust0 = None
        self.finite_difference_h = None
        if optimize_types:
            print("Performing initial FF fit...")
            while True:
                try:
                    self._po.apply(jobtype="OPTIMIZE")
                    break
                except RuntimeError:
                    self._bump_zero_parameters(1e-3, names="epsilon")
                    self.trust0 = self._po._options["trust0"] / 2.0
                    print(
                        "Initial optimization failed; reducing trust radius to", self.trust0
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
            self.to_smirnoff_xml(newff_name, verbose=True)
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
                            ignore_bits=ignore_bits
                        )

                        if node is None:
                            print("No new parameter split, done!")
                            break

                        if optimize_during_typing:
                            print("Performing micro optimization for new split")
                            self._po.apply(jobtype="OPTIMIZE")
                            obj = self._po.X
                            print("Objective after minimization:", self._po.X)
                            self.load_new_parameters(self._po.new_ff)
                            newff_name = "newFF.offxml"
                            self.to_smirnoff_xml(newff_name, verbose=True)
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
                            breakpoint()
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

        if optimize_parameters:
            try:
                if not optimize_types:
                    self._po.apply(jobtype="GRADIENT")
                    initial = self._po.X

                self._po.apply(jobtype="OPTIMIZE")
                obj = self._po.X
            except RuntimeError as e:
                raise e
                breakpoint()
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
                obj = np.inf
            finally:
                self.load_new_parameters(self._po.new_ff)
                newff_name = "newFF.offxml"
                # self._plot_gradients(fname_prefix="optimized.final")
                print("Optimized objective is", obj, "initial was", initial)
                print("Total drop is", obj - initial)
                self.to_smirnoff_xml(newff_name, verbose=True, renumber=True)
                self.to_pickle()

    @classmethod
    def from_smirnoff(self, input):
        pass

    def set_physical_optimizer(self, obj):
        self._po = obj

    def set_smarts_generator(self, obj):
        self._to = obj
