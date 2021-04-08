import copy
import itertools
import sys

import networkx
import networkx.algorithms.graph_hashing
import networkx.algorithms.isomorphism
import numpy as np
import offsb.chem.types
import offsb.rdutil.mol
import openforcefield.typing.engines.smirnoff.parameters
import rdkit.Chem


class ValenceDict(openforcefield.typing.engines.smirnoff.parameters.ValenceDict):
    def __repr__(self):
        return dict(self).__repr__()


class ImproperDict(openforcefield.typing.engines.smirnoff.parameters.ImproperDict):
    def __repr__(self):
        return dict(self).__repr__()


def ordered_align_score(i, ref, o):
    return i, ref.align_score(o)


def ordered_contains(i, ref, o):
    return i, o in ref


def node_eq(a, b):
    return a["primitive"] == b["primitive"]


def node_contains(a, b):
    return a["primitive"] in b["primitive"]


def edge_eq(a, b):
    return a["primitive"] == b["primitive"]


def edge_contains(a, b):
    return a["primitive"] in b["primitive"]


class SMARTSUnconnectedTaggedAtomsException(Exception):
    pass


class SMARTSTagsIndexError(Exception):
    pass


def primitive_add(a, b):
    return a + b


def primitive_iadd(a, b):
    a += b
    return a


def primitive_subtract(a, b):
    return a - b


def primitive_isubtract(a, b):
    a -= b
    return a


def primitive_xor(a, b):
    return a ^ b


def primitive_ixor(a, b):
    a ^= b
    return a


def primitive_and(a, b):
    return a & b


def primitive_iand(a, b):
    a &= b
    return a


def primitive_or(a, b):
    return a | b


def primitive_ior(a, b):
    a |= b
    return a


def primitive_neg(a):
    return ~a


def primitive_to_smarts(primitive, **kwargs):

    tag = kwargs.get("tag")

    if tag is None and "tag" in kwargs:
        kwargs.pop("tag")

    return primitive.to_smarts(**kwargs)


def primitive_compact_str(primitive, **kwargs):
    return primitive.compact_str()


class ChemicalGraph:
    def __init__(self):
        self._G = None
        self._primary = tuple()
        self._n_atoms = 0
        self._connect = tuple()

        self._permutations = None

        self._score_cache = {}

        self._depth_cache = {}
        self._smarts_hash = None
        self._cache_smarts = None

    def __getattr__(self, name):

        # needed for serialization in e.g. multiprocessing
        if name in {"__getstate__", "__setstate__"}:
            return object.__getattr__(self, name)

        return getattr(self._G, name)

    def __iter__(self):
        yield from self.iter_bits()

    def bits(self, maxbits=False):

        bits = 0
        for atom in self.nodes:
            bits += self.nodes[atom]["primitive"].bits(maxbits=maxbits)
        for bond in self.edges:
            bits += self.edges[bond]["primitive"].bits(maxbits=maxbits)
        return bits

    def atom_enable(self, field):

        status = False
        for atom in self.nodes:
            status |= self.nodes[atom]["primitive"].enable(field)
        return status

    def bond_enable(self, field):
        status = False
        for bond in self.edges:
            status |= self.edges[bond]["primitive"].enable(field)
        return status

    def atom_disable(self, field):

        status = False
        for atom in self.nodes:
            status |= self.nodes[atom]["primitive"].disable(field)
        return status

    def bond_disable(self, field):
        status = False
        for bond in self.edges:
            status |= self.edges[bond]["primitive"].disable(field)
        return status

    def clear_caches(self):

        self._depth_cache.clear()
        self._score_cache.clear()

    def clear_primitives(self):

        for atom in self.nodes:
            self.nodes[atom]["primitive"].clear()

        for bond in self.edges:
            self.edges[bond]["primitive"].clear()

    def compact_str(self, maxbits=False, atom_universe=None, bond_universe=None):

        return self._dag_to_smarts_nx(
            self._G,
            primitive_compact_str,
            allow_empty=True,
            atom_universe=atom_universe,
            bond_universe=bond_universe,
        )

    def any(self):

        for atom in self.nodes:
            if self.nodes[atom]["primitive"].any():
                return True

        for bond in self.edges:
            if self.edges[bond]["primitive"].any():
                return True

        return False

    def __hash__(self):

        for atom in self.nodes:
            self.nodes[atom]["smarts"] = str(hash(self.nodes[atom]["primitive"]))

        for bond in self.edges:
            self.edges[bond]["smarts"] = str(hash(self.edges[bond]["primitive"]))

        h = networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(
            self._G, edge_attr="smarts", node_attr="smarts"
        )

        for atom in self.nodes:
            del self.nodes[atom]["smarts"]

        for bond in self.edges:
            del self.edges[bond]["smarts"]

        return hash(h)

    def all(self):

        for atom in self.nodes:
            if not self.nodes[atom]["primitive"].all():
                return False

        for bond in self.edges:
            if not self.edges[bond]["primitive"].all():
                return False

        return True

    def is_null(self):

        for atom in self.nodes:
            if self.nodes[atom]["primitive"].is_null():
                return True

        for bond in self.edges:
            if self.edges[bond]["primitive"].is_null():
                return True

        return False

    def is_valid(self):

        for atom in self.nodes:
            if not self.nodes[atom]["primitive"].is_valid():
                return False

        for bond in self.edges:
            if not self.edges[bond]["primitive"].is_valid():
                return False

        return True

    @classmethod
    def _from_smarts(
        cls,
        smarts,
        allow_unconnected=False,
        smiles=False,
        protonate=False,
        openff_compat=True,
    ):
        """
        This means we don't have a complete molecule, or at least can't assume
        """

        cls = cls()

        cls._G = networkx.Graph()
        cls.graph["tags"] = {}

        if smiles:
            mol = offsb.rdutil.mol.build_from_smiles(smarts, protonate=protonate)
        else:
            mol = rdkit.Chem.MolFromSmarts(smarts)

        primaries = []

        for i, atom in enumerate(mol.GetAtoms()):
            idx = atom.GetIdx()
            if smiles:
                h = atom.GetTotalNumHs(includeNeighbors=True)
            else:
                h = atom.GetNumExplicitHs()
            ring = 0
            if atom.IsInRing():
                # argmax gives the first index that is True
                ring = np.argmax([atom.IsInRingSize(i) for i in range(3, 10000)]) + 1

            # This doesn't quite work because RD says if it isn't present, it doesn't
            # exist, which is opposite of my approach. This should work with
            # from_smiles, though.
            if smiles:
                data = {
                    "S": int(atom.GetAtomicNum()),
                    "H": int(h),
                    "X": sum([0] + [1 for b in atom.GetBonds()]),
                    "x": sum([0] + [1 for b in atom.GetBonds() if b.IsInRing()]),
                    "r": int(ring),
                    "aA": int(atom.GetIsAromatic()),
                    "q": int(atom.GetFormalCharge()),
                }
                primitive = offsb.chem.types.AtomType.from_dict(data)
            else:

                atom_smarts = atom.GetSmarts()
                primitive = offsb.chem.types.AtomType.from_smarts(atom_smarts)

            if openff_compat:
                primitive.disable("aA")

            cls.add_node(idx, primitive=primitive)
            tag = atom.GetAtomMapNum()
            if tag > 0:
                cls.graph["tags"][tag] = idx
                primaries.append(idx)

        if cls._n_atoms > 0:
            assert len(primaries) == cls._n_atoms
            for i in range(1, cls._n_atoms + 1):
                if i not in cls.graph["tags"]:
                    raise SMARTSTagsIndexError()

        order_map = {1: 1, 2: 2, 3: 3, 1.5: 4}
        for b, bond in enumerate(mol.GetBonds()):

            if smiles:
                order = order_map[bond.GetBondTypeAsDouble()]
                ring = int(bond.IsInRing())
                data = {"Order": order, "aA": ring}
                primitive = offsb.chem.types.BondType.from_dict(data)
            else:
                bond_smarts = bond.GetSmarts()
                primitive = offsb.chem.types.BondType.from_smarts(bond_smarts)

            idx_i, idx_j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            cls.add_edge(idx_i, idx_j, primitive=primitive)

        if not allow_unconnected:
            for connect in cls._connect:
                i = cls.graph["tags"][connect[0]]
                j = cls.graph["tags"][connect[1]]
                if j not in cls.adj[i]:
                    raise SMARTSUnconnectedTaggedAtomsException()

        cls._primary = tuple(cls.graph["tags"][i] for i in sorted(cls.graph["tags"]))

        if cls._permutations is None:
            cls._permutations = tuple(i for i in range(len(cls.nodes)))

        if cls._n_atoms == 0:
            cls._n_atoms = len(cls._primary)

        return cls

    @classmethod
    def from_smiles(cls, smarts, allow_unconnected=False):
        return cls._from_smarts(
            smarts, allow_unconnected=allow_unconnected, smiles=True
        )

    @classmethod
    def from_smarts(cls, smarts, allow_unconnected=False):
        return cls._from_smarts(
            smarts, allow_unconnected=allow_unconnected, smiles=False
        )

    def _vertex_overlap(self, i, j):

        prim_i = i["primitive"]
        prim_j = j["primitive"]

        # prim_i.align_to(prim_j)

        return (prim_i & prim_j).bits(maxbits=True)

    def pairwise_overlap(self, A, o, B):

        H = {}

        for i in A:
            prim_i = self.nodes[i]["primitive"]
            bonds_i = self.adj[i].values()
            for j in B:
                prim_j = o.nodes[j]["primitive"]
                bonds_j = o.adj[j].values()
                best_score = 0
                for bond_i in itertools.permutations(bonds_i):
                    for bond_j in itertools.permutations(bonds_j):
                        score = 0
                        for b_i, b_j in zip(bond_i, bond_j):
                            b_i = b_i["primitive"]
                            b_j = b_j["primitive"]
                            score += (b_i & b_j).bits(maxbits=True)
                        best_score = max(best_score, score)

                # prim_i.align_to(prim_j)
                H[(i, j)] = (prim_i & prim_j).bits(maxbits=True) + best_score

        return H

    def _guess_center_atom(self):
        if type(self._G) is networkx.DiGraph:
            top = [
                self._G.nodes[i] for i in self._G.nodes if self._G.predecessors(i) == 0
            ]
            if len(top) == 0:
                return list(self._G.nodes)[0]
            else:
                return top[0]
        else:
            path = dict(networkx.all_pairs_shortest_path(self._G))
            max_len = -1
            center = None
            for i in self._G.nodes:
                for j in self._G.nodes:
                    p_len = len(path[i][j])
                    if p_len > max_len:
                        center = path[i][j][p_len // 2]
                        max_len = p_len
            assert center is not None
            return center

    def _iter_bits(self, g, nodes, visited, skip_ones=False):

        children = set()
        for v in nodes:
            for connected in g.adj[v]:
                bond = (v, connected)
                if bond not in visited:
                    for bit in self._G.edges[bond]["primitive"].iter(
                        skip_ones=skip_ones
                    ):
                        g._G.edges[bond]["primitive"] += bit
                        yield g.copy()
                        g._G.edges[bond]["primitive"] ^= bit
                    visited.add(bond)
                atom = connected
                if atom not in visited:
                    for bit in self._G.nodes[atom]["primitive"].iter(
                        skip_ones=skip_ones
                    ):
                        g._G.nodes[atom]["primitive"] += bit
                        yield g.copy()
                        g._G.nodes[atom]["primitive"] ^= bit
                    visited.add(atom)
                    children.add(atom)

        if len(children) == 0:
            return
        else:
            yield from self._iter_bits(g, children, visited, skip_ones=skip_ones)

    def iter_bits(self, skip_ones=False):

        visited = set()
        g = self.copy()
        for n in g.nodes:
            g.nodes[n]["primitive"].clear()
        for e in g.edges:
            g.edges[e]["primitive"].clear()

        connect_lst = g._connect
        center = [self._guess_center_atom()]

        if len(connect_lst) == 0:
            if len(self._primary) > 0:
                center = self._primary[0]
            connect_lst = ((center, center),)
            center = [center]

        for connect in connect_lst:
            atom = self.graph["tags"].get(connect[0])

            # hope this works; assumes there are no tags so try to treat
            # the center atom as the root, and expand outward
            if atom is None:
                atom = center[0]

            if atom not in visited:
                for bit in self.nodes[atom]["primitive"].iter(skip_ones=skip_ones):
                    g.nodes[atom]["primitive"] += bit
                    yield g.copy()
                    g.nodes[atom]["primitive"] ^= bit
            if connect[0] != connect[1]:

                bond = (atom, self.graph["tags"][connect[1]])
                assert bond in self._G.edges

                if bond not in visited:
                    for bit in self._G.edges[bond]["primitive"].iter(
                        skip_ones=skip_ones
                    ):
                        g._G.edges[bond]["primitive"] += bit
                        yield g.copy()
                        g._G.edges[bond]["primitive"] ^= bit

                atom = self.graph["tags"][connect[1]]
                if atom not in visited:
                    for bit in self.nodes[atom]["primitive"].iter(skip_ones=skip_ones):
                        g.nodes[atom]["primitive"] += bit
                        yield g.copy()
                        g.nodes[atom]["primitive"] ^= bit

        yield from self._iter_bits(g, center, visited=visited, skip_ones=skip_ones)

    def _overlap_scores(self, o):

        scores = {}

        lvl = 0

        while True:
            A = self.vertices_at_depth(lvl)
            B = o.vertices_at_depth(lvl)
            if len(A) == 0 or len(B) == 0:
                break

            H = self.pairwise_overlap(A, o, B)
            scores.update(H)
            lvl += 1
        return scores

    def _map_vertices(self, o, a, b, scores, lvl, strict=False, equality=False):

        group_mappings = {}

        # if lvl >= len(scores):
        #     # print("Case a")
        #     return {0: [{}]}

        sucA = [i for i in self.vertices_at_depth(lvl) if i in self.adj[a]]
        sucB = [i for i in o.vertices_at_depth(lvl) if i in o.adj[b]]

        if len(sucB) == 0:
            # print("case A")
            # if strict and not equality:
            #     valid = True
            #     for i in sucA:
            #         if not self.nodes[i]['primitive'].all():
            #             valid = False
            #             break
            #         for nbr_i, edge in self.adj[i].items():
            #             if not edge['primitive'].all():
            #                 valid = False
            #                 break
            #         if not valid:
            #             break
            #     if valid:
            #         return {0: [{a: None for a in sucA}]}
            #     else:
            #         return {-1: [{a: None for a in sucA}]}
            return {0: [{a: None for a in sucA}]}

        if len(sucA) == 0:
            if strict and not equality:
                valid = True
                for i in sucB:
                    if not o.nodes[i]["primitive"].all():
                        valid = False
                        break
                    for nbr_i, edge in o.adj[i].items():
                        if not edge["primitive"].all():
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return {0: [{}]}
                else:
                    return {-1: [{}]}

        # shortcut if there is a clear 1:1 map
        # if len(sucA) == 1 and len(sucB) == 1:
        #     return {sucA[0]: sucB[0]}

        H = scores

        n_cached = 0
        n_calc = 0

        for Ai, permA in enumerate(itertools.permutations(sucA)):
            for Bi, permB in enumerate(itertools.permutations(sucB)):
                S = 0
                mapping = {}

                cached = self._score_cache.get((permA, permB), None)
                valid = True
                if cached is None:
                    n_calc += 1
                    # print(Ai, Bi, a, b, permA, permB)
                    for i, j in itertools.zip_longest(permA, permB):

                        # this makes sure that if A has no node, we need to ensure
                        # that B is ~[*] since that is the only way a None node
                        # will be "in" B
                        if i is None and j is not None and strict and not equality:
                            if not o.nodes[j]["primitive"].all():
                                valid = False
                                break
                            for nbr_j, edge in o.adj[j].items():
                                if not edge["primitive"].all():
                                    valid = False
                                    break
                            if not valid:
                                valid = False
                                S = -1
                                break
                        if i is None or j is None:
                            self._score_cache[(permA, permB)] = (-1, None)
                            continue
                        if strict:
                            if equality:
                                if not (
                                    node_eq(self.nodes[i], o.nodes[j])
                                    and edge_eq(self.adj[a][i], o.adj[b][j])
                                ):
                                    self._score_cache[(permA, permB)] = (-1, None)
                                    valid = False
                                    S = -1
                                    break
                            else:
                                if not (
                                    node_contains(self.nodes[i], o.nodes[j])
                                    and edge_contains(self.adj[a][i], o.adj[b][j])
                                ):
                                    self._score_cache[(permA, permB)] = (-1, None)
                                    valid = False
                                    S = -1
                                    break

                        mapping[i] = j
                        S += H[(i, j)] + (
                            self.adj[a][i]["primitive"] + o.adj[b][j]["primitive"]
                        ).bits(maxbits=True)
                    self._score_cache[(permA, permB)] = (S, mapping)
                else:
                    n_cached += 1
                    S, mapping = cached

                if valid:
                    if S not in group_mappings:
                        group_mappings[S] = [mapping]
                    else:
                        group_mappings[S].append(mapping)

        # print("    ", "perms done. calculated", n_calc, "n_cached", n_cached)

        # now we can map the children of a and b, since we know a maps b
        if len(group_mappings) > 0:
            best_score = max(group_mappings)
        else:
            return {-1: {}}
        best_map = group_mappings[best_score]

        best_idx = 0
        best_s = 0
        if len(best_map) > 1 and (lvl + 1) < len(scores):
            for idx, mapping in enumerate(best_map):
                s = 0
                for x, y in mapping.items():
                    new_mappings = self._map_vertices(
                        o, x, y, scores, lvl + 1, strict=strict, equality=equality
                    )
                    s += max(new_mappings)
                if s > best_s:
                    best_s = s
                    best_idx = idx

        # TODO egh, break ties
        # assert len(best_map) == 1

        return {best_score: [group_mappings[best_score][best_idx]]}

    def _map_to(self, o, mappings, scores, lvl, strict=False, equality=False):

        best_s = -1
        best_map = {}

        for idx, mapping in enumerate(mappings):
            s = 0
            total_map = mapping.copy()

            for a, b in mapping.items():

                if b is None:
                    continue

                mapped_scores = self._map_vertices(
                    o, a, b, scores, lvl, strict=strict, equality=equality
                )
                this_s = max(mapped_scores)

                # print(a, b, s, this_s)
                lower_s, new_map = self._map_to(
                    o,
                    mapped_scores[this_s],
                    scores,
                    lvl + 1,
                    strict=strict,
                    equality=equality,
                )
                total_map.update(new_map)
                s += this_s
            if s > best_s:
                best_map = total_map.copy()
                best_s = s

        return best_s, best_map

    def strip_hydrogen(self):

        nodes = list(self.nodes)
        for n in nodes:
            prim = self.nodes[n]["primitive"]
            if prim._symbol[1] and prim._symbol.bits() == 1:
                self._G.remove_node(n)
        nodes = tuple([n for n in self._primary if n in self.nodes])
        if len(nodes) == 0:
            self._primary = tuple([self._guess_center_atom()])

    def vertices_at_depth(self, depth):
        """
        get the vertices at some some depth from the primary set
        """

        ret = self._depth_cache.get(depth)

        if ret is not None:
            return ret

        ret = set()
        for v in self.nodes:
            lens = []
            for root_v in self._primary:
                try:

                    G = self._G
                    if networkx.is_directed(self._G):
                        G = self._G.to_undirected()
                    path_len = networkx.shortest_path_length(G, root_v, v)
                except Exception as e:
                    breakpoint()
                    print(e)
                lens.append(path_len)
            if min(lens) == depth:
                ret.add(v)
        self._depth_cache[depth] = ret
        return ret

    def find(self, o):

        if tuple(self.nodes) == (1, 0, 2, 6, 21, 9, 12, 13, 17):
            breakpoint()
            print(self.to_smarts())
        mapped = o.map_to(self, strict=True)
        if tuple(mapped.values()) == (1, 6, 9, 12):
            breakpoint()
        if any([x is None for x in mapped.values()]):
            return False
        for m, n in mapped.items():
            if not node_contains(self.nodes[n], o.nodes[m]):
                return None
        for edge in o.edges:
            ni, nj = mapped[edge[0]], mapped[edge[1]]
            if not edge_contains(self.adj[ni][nj], o.adj[edge[0]][edge[1]]):
                return None

        return {v: k for k, v in mapped.items()}

    def __contains__(self, o):

        # if len(o.nodes) < len(self.nodes):
        #     return False

        ret = o.map_to(self, strict=True)
        if ret is None:
            return False
        else:
            return True

        # return all([x is not None for x in ret.values()])

    def align_score(self, o):

        self._depth_cache.clear()
        self._score_cache.clear()

        scores = self._overlap_scores(o)

        map_scores = {}
        # manually calculate the primary order since the permutations
        # are pre-determined

        H = scores

        for permA in self._permutations:

            A = list(self._primary)
            A = [A[i] for i in permA]
            B = list(o._primary)

            mapping = {}
            S = 0
            for a, b in zip(A, B):
                mapping[a] = b
                S += H[(a, b)]

            if S not in map_scores:
                map_scores[S] = [mapping]
            else:
                map_scores[S].append(mapping)

        best_score = max(map_scores)
        best_maps = map_scores[best_score]

        total_score, total_map = self._map_to(o, best_maps, scores, 1)

        return best_score + total_score

    def map_to(self, o, strict=False, equality=False):

        self.clear_caches()
        o.clear_caches()

        scores = self._overlap_scores(o)

        map_scores = {}
        # manually calculate the primary order since the permutations
        # are pre-determined

        H = scores

        tags_a = {k: v for k, v in self.graph["tags"].items()}
        tags_b = {k: v for k, v in o.graph["tags"].items()}

        for permA in self._permutations:

            A = list(self._primary)
            A = [A[i] for i in permA]
            B = list(o._primary)

            valid = True

            # need this to remap the connections to the permutation
            perm_map = {i: v + 1 for i, v in enumerate(permA, 1)}

            # eh, assume same IC type for now
            for connect_a, connect_b in zip(self._connect, o._connect):
                edge_a = tags_a[perm_map[connect_a[0]]], tags_a[perm_map[connect_a[1]]]
                edge_b = tags_b[connect_b[0]], tags_b[connect_b[1]]
                if strict:
                    if equality:
                        if not edge_eq(self.edges[edge_a], o.edges[edge_b]):
                            valid = False
                            break
                    else:
                        if not edge_contains(self.edges[edge_a], o.edges[edge_b]):
                            valid = False
                            break

            if not valid:
                continue

            valid = True
            mapping = {}
            S = 0
            for a, b in zip(A, B):
                if strict:
                    if equality:
                        if not node_eq(self.nodes[a], o.nodes[b]):
                            valid = False
                            break
                    else:
                        if not node_contains(self.nodes[a], o.nodes[b]):
                            valid = False
                            break
                mapping[a] = b
                S += H[(a, b)]

            if not valid:
                continue
            if S not in map_scores:
                map_scores[S] = [mapping]
            else:
                map_scores[S].append(mapping)

        if len(map_scores) == 0:
            return None
        best_score = max(map_scores)
        best_maps = map_scores[best_score]

        # TODO: handle ties here
        # assert len(best_map) == 1

        # best_map = best_map[0]

        # total_map = {}
        # for mapping in best_maps:
        total_score, total_map = self._map_to(
            o, best_maps, scores, 1, strict=strict, equality=equality
        )
        if strict and total_score < 0:
            return None
        # if max(new_mapping) > max(total_map):
        #     total_map = new_mapping

        # s = max(total_map)

        total_map = {k: total_map.get(k) for k in self.nodes}

        return total_map

    def _dag_descend_nx(
        self,
        G,
        visitor_fn,
        visited,
        tag,
        source,
        target,
        branch,
        seen,
        current_path,
        encloser=("", ""),
        allow_empty=False,
        atom_universe=None,
        bond_universe=None,
    ):

        lhs, rhs = encloser

        smarts = ""

        debug = False

        bond_edge = G.adj[source][branch]
        bond = visitor_fn(bond_edge["primitive"], universe=bond_universe)
        if debug and lhs:
            print(f"OPENING PAREN FOR SRC {branch}")
        ret = self._dag_to_smarts_nx(
            G,
            visitor_fn,
            visited=visited,
            source=branch,
            seen=seen,
            current_path=current_path,
            tag=tag,
            allow_empty=allow_empty,
            atom_universe=atom_universe,
            bond_universe=bond_universe,
        )
        # if len(ret) == 0 and allow_empty:
        #     ret = "."
        if ret:
            smarts = lhs + bond + ret + rhs
        if debug and rhs:
            print(f"CLOSING PAREN FOR SRC {branch}")
        return smarts

    def minimum_spanning_tree(self):

        G = self._G
        if networkx.is_directed(G):
            G = G.to_undirected()
        G2 = networkx.minimum_spanning_tree(G)

        # ring destroyer: remove bonds in the primaries that aren't specified by self._connect
        # and add those that were removed

        for i, n in G.graph["tags"].items():
            for j, m in G.graph["tags"].items():
                if m in G.adj[n] and (
                    (i, j) in self._connect or (j, i) in self._connect
                ):
                    G2.add_edge(n, m, **G.edges[n, m])
        for i, n in G.graph["tags"].items():
            for j, m in G.graph["tags"].items():
                if m in G2.adj[n] and (
                    (i, j) not in self._connect and (j, i) not in self._connect
                ):
                    G2.remove_edge(n, m)
        return G2

    def _max_path_nx(self, G, visited, source=None, avoid=None):

        if not avoid:
            avoid = []

        pair = [None, [], 0]
        tagged = tuple(G.graph["tags"].values())
        # prefer the path with all tagged atoms if it is available
        tag_path = all([x not in visited for x in tagged])

        G2 = self.minimum_spanning_tree()
        paths = networkx.shortest_path(G2)

        if tag_path:
            pair = [None, [], 0]

        # breakpoint()

        for i in paths:
            for j, path in paths[i].items():

                if (
                    (i in avoid and i != source)
                    or (j in avoid and j != source)
                    or any(x in avoid for x in path if x not in (i, j))
                ):
                    continue

                A1 = (not tag_path) and len(path) > pair[2]
                # A1 = len(path) > pair[2]

                # We are seeking the longest path that has the tagged path
                # running through it
                A21 = len(path) > pair[2]
                A23 = [
                    tagged == tuple(path[i : i + len(tagged)])
                    or tagged == tuple(path[i : i + len(tagged)][::-1])
                    for i in range(len(path) - len(tagged) + 1)
                ]
                A23 = any(A23)
                A2 = tag_path and A21 and A23

                A = A1 or A2
                B = all(x not in visited for x in path)
                C = source is None or (i == source)
                if A and B and C:
                    pair = [(i, j), path, len(path)]

        # rings are hard: we might end up with a tagged path that is longer
        # than the shortest path to that node, for example a torsion on cyclopentane,
        # with a torsion on 1,2,3,4, but the shortest path from 1 to 4 is 1,5,4
        # We inject a workaround to avoid this, since if that happens the returned
        # path will be empty due to the fact we enforced trying to match the complete
        # tagged path
        if len(pair[1]) == 0 and tag_path:
            pair[1] = tagged
        elif not pair[1] and source is not None:
            pair[1] = [source]
        return pair[1]

    def _dag_to_smarts_nx(
        self,
        G,
        visitor_fn,
        visited=None,
        source=None,
        seen=None,
        current_path=None,
        tag=None,
        allow_empty=False,
        atom_universe=None,
        bond_universe=None,
    ):

        smarts = ""

        debug = False

        if visited is None:
            visited = []
        if tag is None:
            tag = {}

        if source is None:
            source = self._primary[0]

        if seen is None:
            seen = set()

        if not current_path:
            current_path = []

        if len(current_path) < 2:
            path = self._max_path_nx(G, visited=visited, source=source, avoid=seen)
        else:
            path = current_path

        if len(path) == 0:
            return ""

        seen.update(path)

        src = path[0]
        path = path[1:]

        if tag is None:
            tag_idx = None
        else:
            tag_idx = tag.get(src, None)
        node = G.nodes[src]

        if src not in visited:
            smarts = visitor_fn(node["primitive"], tag=tag_idx, universe=atom_universe)
            if debug:
                smarts = str(src)
            if smarts == "" and allow_empty:
                smarts = "[]"
            visited.append(src)

        if debug:
            print(
                f"SOURCE {src} PATH {path} VISIT {visited} CURPAT {current_path} SEEN {seen} NBRS {list(G.adj[src])}"
            )

        def find_branches(neighbors, visited, path, current_path):
            if debug:
                for x in neighbors:
                    print(
                        f" BRANCH {x}?", x not in visited, x not in seen, f"SEEN {seen}"
                    )
            return filter(lambda x: x not in visited and x not in seen, neighbors)

        for i, node_i in enumerate(path, 0):

            neighbors = find_branches(G.adj[src], visited, path, seen)

            if debug:
                neighbors = list(neighbors)
                print(f"SRC {src} NEW BRANCHES: {neighbors}")

            seen.update(path)

            for nbr in neighbors:

                if debug:
                    print(f"   BRANCHING {src} -> {neighbors}")

                current_path = []
                ret = self._dag_descend_nx(
                    G,
                    visitor_fn,
                    visited,
                    tag,
                    src,
                    path[-1],
                    nbr,
                    seen,
                    current_path,
                    encloser=("(", ")"),
                    allow_empty=allow_empty,
                    atom_universe=atom_universe,
                    bond_universe=bond_universe,
                )
                if ret:
                    smarts += ret

            if debug:
                print(f"   FOLLOWING {src} -> {node_i}")

            current_path = path
            ret = self._dag_descend_nx(
                G,
                visitor_fn,
                visited,
                tag,
                src,
                node_i,
                node_i,
                seen,
                current_path,
                encloser=("", ""),
                allow_empty=allow_empty,
                atom_universe=atom_universe,
                bond_universe=bond_universe,
            )
            smarts += ret

            src = node_i

        return smarts

    def to_smarts(self, tag=None, atom_universe=None, bond_universe=None):

        if tag is None or tag is True:
            if len(self._primary) > 0:
                tag = {k: i for i, k in enumerate(self._primary, 1)}
            else:
                tag = None
        if tag is False:
            tag = None

        # TODO: keep ring closures included so that they are included in SMARTS
        G = self.minimum_spanning_tree()

        return self._dag_to_smarts_nx(
            G,
            primitive_to_smarts,
            tag=tag,
            atom_universe=atom_universe,
            bond_universe=bond_universe,
        )


class MoleculeGraph(ChemicalGraph):
    def __init__(self):

        super().__init__()

        self._ic = None
        self._G = None

        self._hydrogen = True
        self._depth_limit = None
        self._extend_include_hydrogen = True

        self.atom = []

        self._primary = ()

    @classmethod
    def from_ic_primitives(
        self,
        IC_primitives,
        distinguish_hydrogen=True,
        depth_limit=None,
        min_depth=0,
        explicit_hydrogen=False,
    ):

        self = self()

        self._ic = IC_primitives
        self._G = self._build_graph_mol_from_primitives_nx()
        self._G.graph["tags"] = {}

        self._hydrogen = distinguish_hydrogen
        self._depth_limit = depth_limit
        self._min_depth = min_depth
        self._extend_include_hydrogen = explicit_hydrogen

        self.atom = self._perceive_nx()

        self._primary = tuple([self._guess_center_atom()])

        return self

    def copy(self):

        obj = self.__class__()

        obj._ic = self._ic.copy()
        obj._G = copy.deepcopy(self._G.copy())

        obj._hydrogen = self._hydrogen
        obj._depth_limit = self._depth_limit
        obj._extend_include_hydrogen = self._extend_include_hydrogen

        obj.atom = {i: a.copy() for i, a in self.atom.items()}

        obj._primary = self._primary

        return obj

    def _build_graph_mol_from_primitives_nx(self):

        bond_primitives = {k: v for k, v in self._ic.items() if len(k) == 2}
        g = networkx.Graph()

        def isstr(x):
            return type(x) is str

        for (i, j), (atomi, bond, atomj) in bond_primitives.items():

            prim_i = (
                atomi
                if not isstr(atomi)
                else offsb.chem.types.AtomType.from_smarts(atomi)
            )
            prim_j = (
                atomj
                if not isstr(atomj)
                else offsb.chem.types.AtomType.from_smarts(atomj)
            )
            prim_b = (
                bond if not isstr(bond) else offsb.chem.types.BondType.from_smarts(bond)
            )
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

            tree = AtomEnvironment.from_primitive(n, primitive)
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
            depth = networkx.shortest_path_length(subgraph._G, root, n)
            if limit is not None and depth >= limit:
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
        groups = [x for x in groups if len(x) > 0]

        for group in groups:

            paths = networkx.shortest_path_length(
                subgraphs[group[0]]._G, source=subgraphs[group[0]]._primary[0]
            )
            min_path = max(paths.values())
            if len(group) == 1 and min_path >= self._min_depth:
                continue
            for n in group:
                sg = subgraphs[n]
                modified |= self._extend_subgraph_nx(sg)

        if modified:
            self._distinguish_nx(subgraphs)

    def __eq__(self, o):

        G1 = self._G
        if networkx.is_directed(G1):
            G1 = G1.to_undirected()

        G2 = o._G
        if networkx.is_directed(G2):
            G2 = G2.to_undirected()

        return networkx.is_isomorphic(
            G1,
            G2,
            node_match=node_eq,
            edge_match=edge_eq,
        )

    def __contains__(self, o):

        return len(self.find(o)) > 0

    def find_torsions(self, o):

        # n = len(o.graph['tags'])

        ret = []
        for t in self.torsions().values():
            r = t.find(o)
            if r is not None:
                ret.append(tuple(r))
        return ret

        GM = networkx.algorithms.isomorphism.GraphMatcher(
            self._G,
            o._G,
            node_match=node_contains,
            edge_match=edge_contains,
        )
        # GM.subgraph_is_monomorphic()
        # return list(GM.subgraph_isomorphisms_iter())
        matches = list()
        result = []
        for match in GM.subgraph_isomorphisms_iter():
            key = list(match)
            if key[0] > key[-1]:
                key = key[::-1]
            key = tuple(key)
            if key not in matches:
                result.append(match)
                matches.append(key)
                # print(key, matches)

        return result

    def _ic_generator(self, n, builder, raise_on_error=True):
        ret = {}
        for ic in self._ic:
            if len(ic) != n:
                continue
            try:
                ret[ic] = builder.from_environments(*(self.atom[i] for i in ic), self)
            except Exception:
                if raise_on_error:
                    raise
        return ret

    @classmethod
    def from_smiles(
        cls,
        smi,
        distinguish_hydrogen=True,
        depth_limit=None,
        min_depth=0,
        explicit_hydrogen=False,
        protonate=False,
        openff_compat=True,
    ):

        cls = super()._from_smarts(
            smi, smiles=True, protonate=protonate, openff_compat=openff_compat
        )
        mol = offsb.rdutil.mol.build_from_smiles(smi, protonate=protonate)

        IC_primitives = {}

        def get_primitives(ic):
            return tuple(map(lambda x: x["primitive"], ic))

        def atom(g, atoms):
            return get_primitives((g.nodes[atoms[0]],))

        def bond(g, atoms):
            return get_primitives(
                (g.nodes[atoms[0]], g.edges[atoms], g.nodes[atoms[1]])
            )

        def angle(g, atoms):
            return get_primitives(
                (
                    g.nodes[atoms[0]],
                    g.edges[atoms[0:2]],
                    g.nodes[atoms[1]],
                    g.edges[atoms[1:3]],
                    g.nodes[atoms[2]],
                )
            )

        def torsion(g, atoms):
            return get_primitives(
                (
                    g.nodes[atoms[0]],
                    g.edges[atoms[0:2]],
                    g.nodes[atoms[1]],
                    g.edges[atoms[1:3]],
                    g.nodes[atoms[2]],
                    g.edges[atoms[2:4]],
                    g.nodes[atoms[3]],
                )
            )
            pass

        def outofplane(g, atoms):
            return get_primitives(
                (
                    g.nodes[atoms[0]],
                    g.edges[atoms[0:2]],
                    g.nodes[atoms[1]],
                    g.edges[atoms[1], atoms[2]],
                    g.nodes[atoms[2]],
                    g.edges[atoms[1], atoms[3]],
                    g.nodes[atoms[3]],
                )
            )

        queries = {
            "[*:1]": atom,
            "[*:1]~[*:2]": bond,
            "[*:1]~[*:2]~[*:3]": angle,
            "[*:1]~[*:2]~[*:3]~[*:4]": torsion,
            "[*:1]~[*:2](~[*:3])~[*:4]": outofplane,
        }

        match_kwargs = dict()

        if openff_compat:
            match_kwargs.update(dict(uniquify=False, useChirality=True))
        for q, builder in queries.items():
            qmol = rdkit.Chem.MolFromSmarts(q)
            qmol = offsb.rdutil.mol.build_from_smarts(q)
            for match in mol.GetSubstructMatches(qmol, **match_kwargs):
                IC_primitives[match] = builder(cls, match)

        cls = cls.from_ic_primitives(
            IC_primitives,
            distinguish_hydrogen=distinguish_hydrogen,
            depth_limit=depth_limit,
            min_depth=min_depth,
            explicit_hydrogen=explicit_hydrogen,
        )

        return cls

    def atoms(self):
        return {tuple([k]): v for k, v in self.atom.items()}

    def bonds(self):
        return ValenceDict(self._ic_generator(2, BondEnvironment))

    def angles(self):
        return ValenceDict(self._ic_generator(3, AngleEnvironment))

    def torsions(self):
        return ValenceDict(
            self._ic_generator(4, TorsionEnvironment, raise_on_error=False)
        )

    def outofplanes(self):
        return ImproperDict(
            self._ic_generator(4, OutOfPlaneEnvironment, raise_on_error=False)
        )


class ChemicalEnvironment(ChemicalGraph):
    def __init__(self):
        super().__init__()
        self._G = networkx.DiGraph()
        self._primary = tuple()
        self._n_atoms = 0
        self._connect = tuple()

        self._permutations = None

    def copy(self):

        cls = self.__class__()
        cls._G = copy.deepcopy(self._G)
        cls._primary = self._primary
        cls._n_atoms = self._n_atoms
        cls._connect = self._connect

        cls._permutations = self._permutations

        cls.clear_caches()

        return cls

    def _build(self, atoms, molecule, connect):

        self._G = networkx.compose_all([a._G for a in atoms])
        if networkx.is_directed(self._G):
            self._G = self._G.to_undirected()
        nodes = [a.root() for a in atoms]
        self._primary = tuple(nodes)
        self._G.graph["tags"] = {i: k for i, k in enumerate(nodes, 1)}

        # strip down anything that is beyond the max depth from this primary

        # Should throw an error if not present, hopefully, so try to fail early
        for c in connect:
            a, b = nodes[c[0]], nodes[c[1]]
            ab = molecule._G.edges[a, b]

        for c in connect:
            a, b = nodes[c[0]], nodes[c[1]]
            ab = molecule._G.edges[a, b]
            if a not in self._G.adj[b]:
                self.add_edge(b, a, **ab)

        # maybe not needed, but just in case
        depth = molecule._depth_limit
        if depth is not None:
            lvl = depth + 1
            while True:
                to_remove = self.vertices_at_depth(lvl)
                if len(to_remove) == 0:
                    break
                self._G.remove_nodes_from(to_remove)
                lvl += 1

    def trim(self):
        nodes = list(self.nodes)
        for v in nodes:
            if self.nodes[v]["primitive"].is_null():
                if v not in self._primary:
                    self.remove_node(v)

        if len(self._primary) > 0:
            nodes = list(self.nodes)
            for v in nodes:
                if not networkx.has_path(self._G, self._primary[0], v):
                    self.remove_node(v)

    def dispatch_op(self, o, fn, add_nodes=False, map=None, fill_new_nodes=False):

        # print("Ref:")
        # for n in self.nodes:
        #     print(n, self.nodes[n]['primitive'])
        # print("o:")
        # for n in o.nodes:
        #     print(n, o.nodes[n]['primitive'])
        g = self.copy()
        go = o.copy()
        if networkx.is_directed(go._G):
            go._G = go._G.to_undirected()
        M = map
        if M is None:
            M = g.map_to(go)
        else:
            M = map.copy()
        if M is None:
            breakpoint()
            M = g.map_to(go)
            return None

        idx = max(o.nodes) + 1
        N = len(o.nodes)
        new_labels = []
        while len(new_labels) < N:
            if idx in self.nodes:
                idx += 1
                continue
            else:
                new_labels.append(idx)
                idx += 1

        new_labels_dict = {i: k for i, k in zip(o.nodes, new_labels)}
        networkx.relabel_nodes(go._G, new_labels_dict, copy=False)

        # TODO: This should be made into a relabel fn
        go._primary = tuple(new_labels_dict[i] for i in go._primary)

        go.graph["tags"] = {
            i: new_labels_dict[go.graph["tags"][i]] for i in go.graph["tags"]
        }
        go.clear_caches()

        for n in M:
            if M[n] is not None:
                M[n] = new_labels_dict[M[n]]

        for n in M:
            m = M[n]
            if m is not None:
                g.nodes[n]["primitive"] = fn(
                    g.nodes[n]["primitive"], go.nodes[m]["primitive"]
                )
            elif add_nodes:
                primitive = g.nodes[n]["primitive"]
                empty = primitive.copy()
                empty.clear()
                if fill_new_nodes:
                    empty.fill()
                g.nodes[n]["primitive"] = fn(primitive, empty)
                for i, e in g.adj[n].items():
                    primitive = e["primitive"]
                    empty = primitive.copy()
                    empty.clear()
                    if fill_new_nodes:
                        e["primitive"].fill()
                    e["primitive"] = fn(primitive, empty)
            else:
                g.remove_node(n)

        nodes_to_add = {}
        if add_nodes:
            for n in go.nodes:
                if n not in M.values():
                    primitive = go.nodes[n]["primitive"].copy()
                    empty = primitive.copy()
                    empty.clear()
                    if fill_new_nodes:
                        empty.fill()
                    # here since go is RHS, empty is actually LHS
                    primitive = fn(empty, primitive)
                    # print("adding node", n)
                    nodes_to_add[n] = primitive
                    # g.add_node(n, primitive=primitive)
        # print("nodes are now", g.nodes)
        # for n in g.nodes:
        #     print(n, g.nodes[n]['primitive'])
        # print("edges are", g.edges)

        Minv = {v: k for k, v in M.items() if v is not None}
        # print(M)

        # won't work since edges aren't added yet
        # Minv = go.map_to(g)

        for edge in g.edges:
            i, j = edge
            edge_exists = (M[i], M[j]) in go.edges()
            if M[i] is not None and M[j] is not None and edge_exists:
                try:
                    # this one is odd; this means that we have a map with
                    # and edge on one object, but not the other. trying to
                    # stick with the "somtimes add stuff" regime, we only keep
                    # the topology if add_nodes is set. Otherwise, we have
                    # to remove it as to be as general as possible
                    #
                    # if we are adding edges, then for a union, we basically
                    # do nothing. if we are intersecting, then we also do nothing
                    # if we are taking the difference, also.. nothing
                    # the only ambiguous part is whether we should be an adding
                    # empty edge...
                    # if not add_nodes:
                    #     g.remove_edge(*edge)
                    # else:
                    g.edges[i, j]["primitive"] = fn(
                        g.edges[edge]["primitive"], go.edges[M[i], M[j]]["primitive"]
                    )
                except Exception as e:
                    breakpoint()
                    print("error", e)

        while add_nodes:
            for m, v in g.map_to(go).items():
                if m not in M or (M[m] is None and v is not None):
                    M[m] = v
            Minv = {v: k for k, v in M.items() if v is not None}
            # Minv = {v: k for k, v in M.items() if v is not None}
            connected = []
            # print("M: ", M)
            # print("Minv", Minv)
            # print("len con", len(connected), len(go.edges))
            for edge in go.edges:
                i, j = edge
                if i in Minv and j in Minv:
                    connected.append(edge)
                if i in Minv and j not in Minv:
                    ref = Minv[i]
                    primitive = go.edges[edge]["primitive"].copy()
                    empty = go.edges[edge]["primitive"].copy()
                    empty.clear()
                    if fill_new_nodes:
                        empty.fill()
                    primitive = fn(empty, primitive)
                    # print("A adding edge", ref, j)
                    g.add_node(j, primitive=nodes_to_add[j])
                    g.add_edge(ref, j, primitive=primitive)
                    connected.append(edge)
                elif j in M.values() and i not in M.values():
                    ref = Minv[j]
                    primitive = go.edges[edge]["primitive"].copy()
                    empty = go.edges[edge]["primitive"].copy()
                    empty.clear()
                    if fill_new_nodes:
                        empty.fill()
                    primitive = fn(empty, primitive)
                    # print("B adding edge", ref, i)
                    g.add_node(i, primitive=nodes_to_add[i])
                    g.add_edge(ref, i, primitive=primitive)
                    connected.append(edge)
            if len(connected) == len(go.edges):
                break
                # elif j not in M.values() and i not in M.values():
                #     primitive = go.edges[edge]["primitive"].copy()
                #     empty = go.edges[edge]["primitive"].copy()
                #     empty.clear()
                #     primitive = fn(primitive, empty)
                #     print("C adding edge", i, j)
                #     g.add_edge(i, j, primitive=primitive)
        # print("edges: ", g.edges)

        # print(g.map_to(go))
        # if len(g.nodes) == 1:
        #     breakpoint()
        #     print("damn")
        # if len(g.nodes) != len(self.nodes):
        #     print("Graph reduced from", len(self.nodes), "to", len(g.nodes))
        #     print("offender is", o.to_smarts())
        #     print("before adding was", self.to_smarts())
        #     print("now it is", g.to_smarts())
        return g

    def filter_contains(self, to_check, executor=None):
        if executor is not None:
            work_list = [
                executor.submit(ordered_contains, i, self, x)
                for i, x in enumerate(to_check)
            ]
            mask = dict(iter(x.result() for x in work_list))

            to_check = [x for i, x in enumerate(to_check) if not mask[i]]

        else:
            work_list = dict(
                iter(ordered_contains(i, self, x) for i, x in enumerate(to_check))
            )
            mask = work_list

            to_check = [x for i, x in enumerate(to_check) if not mask[i]]
        return to_check

    def _union_list_binary_reduce(self, o_list, executor, add_nodes=False):
        import tqdm

        to_check = [self] + o_list
        to_check = sorted(to_check, key=lambda x: x.bits(), reverse=True)

        pbar = tqdm.tqdm(
            total=len(to_check), desc="Union", ncols=80, disable=len(to_check) < 100
        )
        total = len(to_check)
        while len(to_check) > 1:
            result = iter(
                executor.submit(to_check[i - 1].union, to_check[i], add_nodes=add_nodes)
                for i, _ in enumerate(to_check[1:])
            )
            to_check = [ret.result() for ret in result]
            pbar.update(total - len(to_check))
            total = len(to_check)

        pbar.close()
        return to_check[0]

    def union_list(self, o_list, add_nodes=False, executor=None):
        import tqdm

        ref = self.copy()
        to_check = ref.filter_contains(o_list.copy(), executor=executor)
        to_check = sorted(to_check, key=lambda x: x.bits(), reverse=True)

        # some testing needs to be done to ensure it aligns the best
        # first test says it's very slow
        # if executor:
        #     return self._union_list_binary_reduce(to_check, executor, add_nodes=add_nodes)

        pbar = tqdm.tqdm(
            total=len(to_check), desc="Union", ncols=80, disable=len(to_check) < 100
        )
        if add_nodes:
            scores = []
            i = 0
            while len(to_check) > 0:

                if not scores:
                    if executor is not None:
                        scores = [
                            executor.submit(ordered_align_score, i, ref, o)
                            for i, o in enumerate(to_check)
                        ]
                        scores = [ret.result() for ret in scores]
                    else:
                        scores = [
                            ordered_align_score(i, ref, o)
                            for i, o in enumerate(to_check)
                        ]

                s = np.argmax([x[1] for x in sorted(scores, key=lambda y: y[0])])

                result = ref.union(to_check[s], add_nodes=add_nodes)

                n_check = len(to_check)


                if result.equal(ref):
                    # if the union had no effect, then we remove the term, and
                    # keep the scores
                    scores.pop(s)
                    to_check.pop(s)
                else:
                    # if the union did have an effect, force a rescore
                    scores = []

                i += 1

                # this seems to slow things down? evaluate every once in awhile
                if i % 20 == 0:
                    to_check = ref.filter_contains(to_check, executor=executor)
                    # force a rescore
                    scores = []

                ref = result

                pbar.update(n_check - len(to_check))
        else:
            for o in to_check:

                # mapping will be none if not a subset, which means
                # o has new bits and it is worth unioning
                # The strict flag offers some shortcuts,
                # so hopefully doing two maps will be worth it
                # if there are some redundant patterns
                M = ref.map_to(o, strict=True)
                if not M:
                    ref = ref.union(o, add_nodes=add_nodes)
                pbar.update(1)

        pbar.close()

        return ref

    def __hash__(self):
        return super().__hash__()

    def subtract(self, o, add_nodes=True, map=None, fill_new_nodes=True, trim=False):
        ret = self.dispatch_op(
            o,
            primitive_isubtract,
            add_nodes=add_nodes,
            map=map,
            fill_new_nodes=fill_new_nodes,
        )
        if trim:
            ret.trim()
        return ret

    def equal(self, o, map=None):
        def check(ret):
            if ret is None:
                return False
            if any([x is None for x in ret.values()]):
                return False
            return True

        if len(self.nodes) != len(o.nodes):
            return False

        if map is not None:
            try:
                for n in map:
                    if not node_eq(self.nodes[n], o.nodes[map[n]]):
                        return False
                for edge in self.edges:
                    i, j = edge
                    o_edge = map[i], map[j]
                    if not edge_eq(self.edges[edge], o.edges[o_edge]):
                        return False
            except KeyError:
                return False

        else:
            ret = self.map_to(o, strict=True, equality=True)
            if not check(ret):
                return False
        return True

    def not_equal(self, o, map=None):
        return not self.equal(o, map=map)

    def append(self, o):

        if len(o.nodes) > len(self.nodes):

            original = o.copy()
            original = original.union(self, add_nodes=True)

        elif len(o.nodes) == len(self.nodes):
            original = self.copy()

            # get the new nodes that don't exist
            original = original.union(o, add_nodes=True)
        else:
            return self

        # reset the nodes that do exist
        mapping = None
        original = original.intersection(
            self, map=mapping, add_nodes=True, fill_new_nodes=True
        )

        return original

    def union(self, o, add_nodes=False, map=None):
        return self.dispatch_op(o, primitive_ior, add_nodes=add_nodes, map=map)

    def intersection(self, o, add_nodes=True, map=None, fill_new_nodes=True):
        return self.dispatch_op(
            o,
            primitive_iand,
            add_nodes=add_nodes,
            map=map,
            fill_new_nodes=fill_new_nodes,
        )

    def xor(self, o, add_nodes=False, map=None):
        return self.dispatch_op(o, primitive_ixor, add_nodes=add_nodes, map=map)

    def neg(self, o, add_nodes=False, map=None):
        return self.dispatch_op(o, primitive_neg, add_nodes=add_nodes, map=map)

    def __sub__(self, o):
        map = {i: j for i, j in zip(self.nodes, o.nodes)}
        ret = self.dispatch_op(
            o, primitive_isubtract, add_nodes=True, map=map, fill_new_nodes=True
        )
        # ret.trim()
        return ret
        # return self.dispatch_op(o, primitive_isubtract, add_nodes=False, map=None)

    def __add__(self, o):
        map = {i: j for i, j in zip(self.nodes, o.nodes)}
        return self.dispatch_op(o, primitive_ior, add_nodes=False, map=map)

    def __or__(self, o):
        return self + o

    def __xor__(self, o):
        map = {i: j for i, j in zip(self.nodes, o.nodes)}
        return self.dispatch_op(
            o, primitive_ixor, add_nodes=True, map=map, fill_new_nodes=False
        )

    def __and__(self, o):
        map = {i: j for i, j in zip(self.nodes, o.nodes)}
        ret = self.dispatch_op(
            o, primitive_iand, add_nodes=False, map=map, fill_new_nodes=False
        )
        # ret.trim()
        return ret
        # return self.dispatch_op(o, primitive_isubtract, add_nodes=False, map=None)

    def __eq__(self, o):

        # magnitudes faster; hope it works
        return hash(self) == hash(o)

        def check(ret):
            if ret is None:
                return False
            if any([x is None for x in ret.values()]):
                return False
            return True

        if len(self.nodes) != len(o.nodes):
            return False

        if len(self.edges) != len(o.edges):
            return False

        ret = o.map_to(self, strict=True, equality=True)
        return check(ret)

    def __ne__(self, o):
        return not (self == o)


class AtomEnvironment(ChemicalEnvironment):
    def __init__(self):
        super().__init__()
        self._n_atoms = 1
        self._connect = tuple()
        self._permutations = ((0,),)
        # self.add_node(0, primitive=offsb.chem.types.AtomType())
        # self._primary = (0,)
        # self.graph['tags'] = {1: 0}

    @classmethod
    def from_primitive(self, n, primitive):
        self = self()
        self.add_node(n, primitive=primitive)
        self._primary = (n,)
        self.graph["tags"] = {1: n}
        return self

    def __hash__(self):
        return super().__hash__()

    def root(self):
        return list(self.nodes)[0]


class BondEnvironment(ChemicalEnvironment):
    def __init__(self):
        super().__init__()
        self._n_atoms = 2
        self._connect = ((1, 2),)
        self._permutations = ((0, 1), (1, 0))

    @classmethod
    def from_environments(
        self, A: AtomEnvironment, B: AtomEnvironment, M: MoleculeGraph
    ):
        self = self()

        self._build([A, B], M, connect=((0, 1),))

        return self


class AngleEnvironment(ChemicalEnvironment):
    def __init__(self):
        super().__init__()
        self._n_atoms = 3
        self._connect = ((1, 2), (2, 3))
        self._permutations = ((0, 1, 2), (2, 1, 0))

    @classmethod
    def from_environments(
        self,
        A: AtomEnvironment,
        B: AtomEnvironment,
        C: AtomEnvironment,
        M: MoleculeGraph,
    ):
        self = self()

        self._build([A, B, C], M, connect=((0, 1), (1, 2)))

        return self


class TorsionEnvironment(ChemicalEnvironment):
    def __init__(self):
        super().__init__()
        self._n_atoms = 4
        self._connect = ((1, 2), (2, 3), (3, 4))
        self._permutations = ((0, 1, 2, 3), (3, 2, 1, 0))

    @classmethod
    def from_environments(
        self,
        A: AtomEnvironment,
        B: AtomEnvironment,
        C: AtomEnvironment,
        D: AtomEnvironment,
        M: MoleculeGraph,
    ):
        self = self()

        self._build([A, B, C, D], M, connect=((0, 1), (1, 2), (2, 3)))

        return self


class OutOfPlaneEnvironment(ChemicalEnvironment):
    def __init__(self):
        super().__init__()
        self._n_atoms = 4
        self._connect = ((1, 2), (2, 3), (2, 4))
        self._permutations = (
            (0, 1, 2, 3),
            (0, 1, 3, 2),
            (2, 1, 0, 3),
            (2, 1, 3, 0),
            (3, 1, 0, 2),
            (3, 1, 2, 0),
        )

    @classmethod
    def from_environments(
        self,
        A: AtomEnvironment,
        B: AtomEnvironment,
        C: AtomEnvironment,
        D: AtomEnvironment,
        M: MoleculeGraph,
    ):
        self = self()

        self._build([A, B, C, D], M, connect=((0, 1), (1, 2), (1, 3)))

        return self
