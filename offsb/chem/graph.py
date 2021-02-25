import collections

import graph_tool
import networkx
import networkx.algorithms.isomorphism
import numpy as np
import offsb.chem.types
import offsb.treedi.node
import offsb.treedi.tree
import rdkit.Chem


def node_eq(a, b):
    return a["primitive"] == b["primitive"]


def node_contains(a, b):
    return a["primitive"] in b["primitive"]


def edge_eq(a, b):
    return a["primitive"] == b["primitive"]


def edge_contains(a, b):
    return a["primitive"] in b["primitive"]


class MoleculeGraph(object):
    def __init__(
        self,
        IC_primitives,
        distinguish_hydrogen=True,
        depth_limit=None,
        explicit_hydrogen=False,
    ):

        self._ic = IC_primitives
        self._G = self._build_graph_mol_from_primitives_nx()

        self._hydrogen = distinguish_hydrogen
        self._depth_limit = depth_limit
        self._extend_include_hydrogen = explicit_hydrogen

        self.atom = self._perceive_nx()

    def _build_graph_mol_from_primitives_nx(self):

        bond_primitives = {k: v for k, v in self._ic.items() if len(k) == 2}
        g = networkx.Graph()

        for (i, j), (atomi, bond, atomj) in bond_primitives.items():

            prim_i = offsb.chem.types.AtomType.from_smarts(atomi)
            prim_j = offsb.chem.types.AtomType.from_smarts(atomj)
            prim_b = offsb.chem.types.BondType.from_smarts(bond)
            g.add_node(i, primitive=prim_i)
            g.add_node(j, primitive=prim_j)
            g.add_edge(i, j, primitive=prim_b)

        return g

    def _perceive_nx(self):

        graphmol = self._G
        hydrogen = self._hydrogen

        subgraphs = {}
        h_graphs = {}

        # generate a subgraph with each atom as the root
        for i, n in enumerate(graphmol.nodes):
            primitive = graphmol.nodes[n]["primitive"]

            tree = AtomEnvironment(n, primitive)
            if (not hydrogen) and primitive._symbol[1]:
                h_graphs[i] = tree
            else:
                subgraphs[i] = tree

        self._distinguish_nx(subgraphs)

        subgraphs.update(h_graphs)
        subgraphs = list(subgraphs.values())
        return {
            i: x
            for i, x in enumerate(sorted(subgraphs, key=lambda x: list(x.nodes)[0]))
        }

    def _extend_subgraph_nx(self, subgraph):

        """
        just add some children
        """

        graphmol = self._G
        limit = self._depth_limit
        hydrogen = self._extend_include_hydrogen

        nodes = list(subgraph.nodes)
        root = list(subgraph.nodes)[0]
        modified = False

        for n in nodes:
            if len(list(subgraph.successors(n))) > 0:
                continue
            if (
                limit is not None
                and networkx.shortest_path_length(subgraph, root, n) > limit
            ):
                continue

            neighbors = [i for i in graphmol.adj[n]]
            if all(
                [i in subgraph.nodes and (n, i) in subgraph.edges for i in neighbors]
            ):
                continue
            for nbr in neighbors:
                pb = graphmol.edges[(n, nbr)]["primitive"]
                pc = graphmol.nodes[nbr]["primitive"]
                if not hydrogen and pc._symbol[1]:
                    continue
                modified = True
                # print(n, nbr, "".join(map(lambda x: x.to_smarts(), [pa, pb, pc])))
                subgraph.add_node(nbr, primitive=pc)
                subgraph.add_edge(n, nbr, primitive=pb)

        return modified

    def _distinguish_nx(self, subgraphs):

        keys = list(subgraphs)

        groups = []
        while len(keys) > 0:
            n = keys.pop()
            new_group = [n]
            sg_n = subgraphs[n]
            for m in keys:
                iso = networkx.algorithms.isomorphism.is_isomorphic(
                    sg_n._G,
                    subgraphs[m]._G,
                    node_match=node_eq,
                    edge_match=edge_eq,
                )
                if iso:
                    new_group.append(m)
            groups.append(new_group)
            for m in new_group[1:]:
                keys.remove(m)

        modified = False
        for group in groups:
            if len(group) == 1:
                continue
            for n in group:
                sg = subgraphs[n]
                modified |= self._extend_subgraph_nx(sg)

        if modified:
            self._distinguish_nx(subgraphs)

    def __eq__(self, o):

        return networkx.is_isomorphic(
            self._G.to_undirected(),
            o._G.to_undirected(),
            node_match=node_eq,
            edge_match=edge_eq,
        )

    def __contains__(self, o):

        return len(self.find(o)) > 0

    def find(self, o):

        GM = networkx.algorithms.isomorphism.GraphMatcher(
            self._G.to_undirected(),
            o._G.to_undirected(),
            node_match=node_contains,
            edge_match=edge_contains,
        )
        GM.subgraph_is_isomorphic()

        return GM.mapping

    def _ic_generator(self, n, builder, raise_on_error=True):
        ret = {}
        for ic in self._ic:
            if len(ic) != n:
                continue
            try:
                ret[ic] = builder(*(self.atom[i] for i in ic), self)
            except Exception:
                if raise_on_error:
                    raise
        return ret

    def bonds(self):
        return self._ic_generator(2, BondEnvironment)

    def angles(self):
        return self._ic_generator(3, AngleEnvironment)

    def torsions(self):
        return self._ic_generator(4, TorsionEnvironment, raise_on_error=False)

    def outofplanes(self):
        return self._ic_generator(4, OutOfPlaneEnvironment, raise_on_error=False)


class ChemicalEnvironment(object):
    def __init__(self):
        self._G = networkx.DiGraph()
        self._primary = tuple()

    def __getattr__(self, name):
        return getattr(self._G, name)

    def _dag_descend_nx(
        self, G, visited, tag, source, target, branch, encloser=("", "")
    ):

        lhs, rhs = encloser

        smarts = ""
        g = set(
            x
            for x in networkx.algorithms.dag.descendants(G, branch)
            if x not in visited
        )
        # g = networkx.algorithms.dag.descendants(G, nbr)
        g.add(branch)
        bond_edge = G.adj[source][target]
        bond = bond_edge["primitive"].to_smarts()
        ret = self._dag_to_smarts_nx(G.subgraph(g), visited=visited, source=branch)
        if len(ret) > 0:
            smarts += lhs + bond + ret + rhs
        return smarts

    def _max_path_nx(self, G, visited, source=None):
        # mst = networkx.minimum_spanning_tree(G)
        paths = networkx.shortest_path(G)

        pair = [None, []]

        for i in paths:
            for j, path in paths[i].items():
                A = len(path) > len(pair[1])
                B = all(x not in visited for x in path)
                C = source is None or (source is not None and i == source)
                if A and B and C:
                    pair = [(i, j), path]

        return pair[1]

    def align_to(self, o):



    def _dag_to_smarts_nx(self, G, visited=None, source=None):

        tag = {k: i for i, k in enumerate(self._primary, 1)}

        smarts = ""

        if visited is None:
            visited = []
        if tag is None:
            tag = {}

        # path = networkx.dag_longest_path(G)
        path = self._max_path_nx(G.to_undirected(), visited=visited, source=source)
        # path = [i for i in path if i not in visited]
        if len(path) == 0:
            return ""

        src = path[0]
        path = path[1:]

        tag_idx = tag.get(src, None)
        node = G.nodes[src]
        if False and node["primitive"]._symbol[1]:
            smarts = "[#1]"
            if tag_idx is not None:
                smarts = "[#1:{}]".format(tag_idx)
        else:
            smarts = node["primitive"].to_smarts(tag=tag_idx)
        visited.append(src)

        for i, node_i in enumerate(path):

            if node_i in visited:
                continue

            neighbors = G.adj[src]
            next_node = None
            for nbr in neighbors:

                if nbr not in path:
                    smarts += self._dag_descend_nx(
                        G, visited, tag, src, node_i, nbr, encloser=("(", ")")
                    )
                else:
                    next_node = nbr

            if next_node is not None:
                smarts += self._dag_descend_nx(
                    G, visited, tag, src, node_i, next_node, encloser=("", "")
                )

            src = node_i

        return smarts

    def to_smarts(self):
        return self._dag_to_smarts_nx(self._G)

    def _build(self, atoms, molecule, connect):
        self._G = networkx.compose_all([a._G for a in atoms]).to_undirected()
        nodes = [a.root() for a in atoms]
        self._primary = tuple(nodes)

        # Should throw an error if not present, hopefully, so try to fail early
        for c in connect:
            a, b = nodes[c[0]], nodes[c[1]]
            ab = molecule._G.edges[a, b]

        for c in connect:
            a, b = nodes[c[0]], nodes[c[1]]
            ab = molecule._G.edges[a, b]
            if a not in self._G.adj[b]:
                self.add_edge(b, a, **ab)


class AtomEnvironment(ChemicalEnvironment):
    def __init__(self, n, primitive):
        super().__init__()
        self.add_node(n, primitive=primitive)
        self._primary = (n,)

    def root(self):
        return list(self.nodes)[0]


class BondEnvironment(ChemicalEnvironment):
    def __init__(self, A: AtomEnvironment, B: AtomEnvironment, M: MoleculeGraph):
        super().__init__()

        self._build([A, B], M, connect=((0, 1),))
        self._G = networkx.compose_all([A._G, B._G]).to_undirected()
        a = A.root()
        b = B.root()
        self._primary = (a, b)

        # Should throw an error if not present, hopefully
        ab = M._G.edges[a, b]

        if a not in self._G.adj[b]:
            self.add_edge(b, a, **ab)


class AngleEnvironment(ChemicalEnvironment):
    def __init__(
        self,
        A: AtomEnvironment,
        B: AtomEnvironment,
        C: AtomEnvironment,
        M: MoleculeGraph,
    ):
        super().__init__()

        self._build([A, B, C], M, connect=((0, 1), (1, 2)))


class TorsionEnvironment(ChemicalEnvironment):
    def __init__(
        self,
        A: AtomEnvironment,
        B: AtomEnvironment,
        C: AtomEnvironment,
        D: AtomEnvironment,
        M: MoleculeGraph,
    ):
        super().__init__()

        self._build([A, B, C, D], M, connect=((0, 1), (1, 2), (2, 3)))


class OutOfPlaneEnvironment(ChemicalEnvironment):
    def __init__(
        self,
        A: AtomEnvironment,
        B: AtomEnvironment,
        C: AtomEnvironment,
        D: AtomEnvironment,
        M: MoleculeGraph,
    ):
        super().__init__()

        self._build([A, B, C, D], M, connect=((0, 1), (1, 2), (1, 3)))


#     @classmethod
#     def from_networkx(cls, graph):
#         pass

#     @classmethod
#     def from_string_list(cls, string_list, sorted=False):
#         atom1 = AtomType.from_string(string_list[0])
#         bond1 = BondType.from_string(string_list[1])
#         atom2 = AtomType.from_string(string_list[2])
#         if len(string_list) > 3:
#             raise NotImplementedError(
#                 "The SMARTS pattern supplied has not been implemented"
#             )
#         return cls(atom1, bond1, atom2, sorted=sorted)

#     @classmethod
#     def _split_string(cls, string, sorted=False):
#         tokens = cls._smirks_splitter(string, atoms=2)
#         return cls.from_string_list(tokens, sorted=sorted)

#     def to_smarts(self, tag=True):

#         if tag:
#             return (
#                 self._atom1.to_smarts(1)
#                 + self._bond.to_smarts()
#                 + self._atom2.to_smarts(2)
#             )
#         else:
#             return (
#                 self._atom1.to_smarts()
#                 + self._bond.to_smarts()
#                 + self._atom2.to_smarts()
#             )

#     def drop(self, other):

#         graph = self.copy()
#         graph._atom1 = graph._atom1.drop(other._atom1)
#         graph._bond = graph._bond.drop(other._bond)

#         if type(other) == type(self):
#             graph._atom2 = graph._atom2.drop(other._atom2)

#         return graph

#     def _is_valid(self) -> bool:
#         return not self.is_null()

#     def is_primitive(self):

#         if super().is_null():
#             return False

#         if not super().is_primitive():
#             return False

#         return True

#         # Skipping these for now; do we want unbonded associations?
#         if self._atom1._X._v[0]:
#             return False

#         if self._atom2._X._v[0]:
#             return False

#         if self._atom1._H._v[0] and self._atom2._symbol._v[1]:
#             return False

#         if self._atom2._H._v[0] and self._atom1._symbol._v[1]:
#             return False

#         return True

#     def cluster(self, primitives=None):
#         if primitives is None:
#             primitives = self.to_primitives()

#         groups = {}
#         for prim in primitives:
#             a1, b, a2 = prim._atom1, prim._bond, prim._atom2
#             if a1 < a2:
#                 a2, a1 = a1, a2
#             if a1 not in groups:
#                 groups[a1] = []
#             groups[a1].extend([b, a2])

#         return groups

#     def to_primitives(self):
#         import tqdm

#         prims = []
#         for field in tqdm.tqdm(self._fields, total=len(self._fields), desc="types"):
#             obj = getattr(self, field)
#             prims.append(set(obj.to_primitives()))

#         ret = set()
#         for a1 in tqdm.tqdm(prims[0], total=len(prims[0]), desc="prims"):
#             for bnd in prims[1]:
#                 for a2 in prims[2]:

#                     found = False
#                     for existing in ret:
#                         if existing._atom1 == a1:
#                             graph = BondGraph(a1, bnd, a2)
#                             found = True
#                             break
#                         elif existing._atom1 == a2:
#                             graph = BondGraph(a2, bnd, a1)
#                             found = True
#                             break
#                     if not found:
#                         if a2 > a1:
#                             graph = BondGraph(a2, bnd, a1)
#                         else:
#                             graph = BondGraph(a1, bnd, a2)

#                     if graph.is_primitive():
#                         ret.add(graph)

#         return list(ret)

#     def __repr__(self):
#         return (
#             "("
#             + self._atom1.__repr__()
#             + ") ["
#             + self._bond.__repr__()
#             + "] ("
#             + self._atom2.__repr__()
#             + ")"
#         )
