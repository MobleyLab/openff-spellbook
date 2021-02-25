class box:

    def extend_subgraph_nx(subgraph, graphmol):

        """
        just add some children
        """

        nodes = list(subgraph.nodes)
        modified = False

        for n in nodes:
            if len(list(subgraph.successors(n))) > 0:
                continue

            neighbors = [i for i in graphmol.adj[n]]
            if all([i in subgraph.nodes and (n, i) in subgraph.edges for i in neighbors]):
                continue
            modified = True
            for nbr in neighbors:
                pb = graphmol.edges[(n, nbr)]["primitive"]
                pc = graphmol.nodes[nbr]["primitive"]
                # print(n, nbr, "".join(map(lambda x: x.to_smarts(), [pa, pb, pc])))
                subgraph.add_node(nbr, primitive=pc)
                subgraph.add_edge(n, nbr, primitive=pb)

        return modified


    def extend_subgraph_gt(subgraph, graphmol):

        """
        just add some children
        """

        nodes = list(subgraph.vertices())
        modified = False

        for n in nodes:
            if len(list(n.out_edges())) > 0:
                continue

            parent_i = subgraph.vp.parent_idx[n]
            parent_v = graphmol.vertex(parent_i)

            modified = False

            neighbors = [i for i in parent_v.out_neighbors()]
            edges = parent_v.out_edges()

            for e, nbr in zip(edges, neighbors):

                # This means that the subgraph already has this atom from the parent
                # molecule
                if nbr in subgraph.vp.parent_idx:
                    continue

                modified = True

                pb = graphmol.ep.primitives[e]
                pc = graphmol.vp.primitives[nbr]

                vnew = subgraph.add_vertex()
                # print(n, vnew, "".join(map(lambda x: x.to_smarts(), [pb, pc])))
                subgraph.vp.primitives[vnew] = pc
                subgraph.vp.parent_idx[vnew] = int(nbr)

                e = subgraph.add_edge(n, vnew)
                subgraph.ep.primitives[e] = pb

        return modified


    def distinguish_nx(subgraphs, graphmol):

        keys = list(subgraphs)

        groups = []
        while len(keys) > 0:
            n = keys.pop()
            new_group = [n]
            sg_n = subgraphs[n]
            for m in keys:
                iso = networkx.algorithms.isomorphism.is_isomorphic(
                    sg_n, subgraphs[m], node_match=node_eq, edge_match=edge_eq
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
                modified |= extend_subgraph_nx(sg, graphmol)

        if modified:
            distinguish_nx(subgraphs, graphmol)


    def distinguish_gt(subgraphs, graphmol):

        keys = list(subgraphs)

        groups = []
        while len(keys) > 0:
            n = keys.pop()
            new_group = [n]
            sg_n = subgraphs[n]
            for m in keys:
                sg_m = subgraphs[m]
                iso = graph_tool.topology.subgraph_isomorphism(
                    sg_n,
                    sg_m,
                    vertex_label=(sg_n.vp.primitives, sg_m.vp.primitives),
                    edge_label=(sg_n.ep.primitives, sg_m.ep.primitives),
                    max_n=1,
                )
                if len(iso) > 0:
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
                modified |= extend_subgraph_gt(sg, graphmol)

        if modified:
            distinguish_gt(subgraphs, graphmol)


    def perceive_nx(graphmol, hydrogen=False):

        subgraphs = {}
        h_graphs = {}

        # generate a subgraph with each atom as the root
        for i, n in enumerate(graphmol.nodes):
            payload = graphmol.nodes[n]["primitive"]
            tree = networkx.DiGraph()
            tree.add_node(n, primitive=payload)
            if (not hydrogen) and payload._symbol[1]:
                h_graphs[i] = tree
            else:
                subgraphs[i] = tree

        distinguish_nx(subgraphs, graphmol)

        subgraphs.update(h_graphs)
        subgraphs = list(subgraphs.values())
        return {
            i: x for i, x in enumerate(sorted(subgraphs, key=lambda x: list(x.nodes)[0]))
        }


    def perceive_gt(graphmol, hydrogen=False):

        subgraphs = {}
        h_graphs = {}

        # generate a subgraph with each atom as the root
        for i, v in enumerate(graphmol.vertices()):
            primitive = graphmol.vp.primitives[v]
            g = graph_tool.Graph(directed=True)
            g.vertex_properties["primitives"] = g.new_vertex_property("object")
            g.vertex_properties["parent_idx"] = g.new_vertex_property("int")
            g.edge_properties["primitives"] = g.new_edge_property("object")

            vi = g.add_vertex()
            g.vp.primitives[vi] = primitive
            g.vp.parent_idx[vi] = int(v)

            if (not hydrogen) and primitive._symbol[1]:
                h_graphs[i] = g
            else:
                subgraphs[i] = g

        distinguish_gt(subgraphs, graphmol)

        subgraphs.update(h_graphs)
        subgraphs = list(subgraphs.values())
        return {
            i: x
            for i, x in enumerate(sorted(subgraphs, key=lambda x: list(x.vertices())[0]))
        }


    def max_path_nx(G, visited, source=None):
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


    def max_path_gt(G, visited, source=None):

        directed = G.get_directed()

        G.set_directed(False)
        mst = graph_tool.minimum_spanning_tree(G)
        paths = graph_tool.shortest_path(mst)

        pair = [None, []]

        for i in paths:
            for j, path in paths[i].items():
                A = len(path) > len(pair[1])
                B = all(x not in visited for x in path)
                C = source is None or (source is not None and i == source)
                if A and B and C:
                    pair = [(i, j), path]

        G.set_directed(directed)

        return pair[1]


    def dag_angle_to_smarts_nx(A, B, C, graphmol):
        G = networkx.compose_all([A, B, C]).to_undirected()
        a = list(A.nodes)[0]
        b = list(B.nodes)[0]
        c = list(C.nodes)[0]
        ab = graphmol.edges[a, b]
        bc = graphmol.edges[b, c]

        if a not in G.adj[b]:
            G.add_edge(b, a, **ab)
        if c not in G.adj[b]:
            G.add_edge(b, c, **bc)

        return dag_to_smarts_nx(G, tag={x: i for i, x in enumerate([a, b, c], 1)})


    def dag_bond_to_smarts_nx(A, B, graphmol):
        G = networkx.compose_all([A, B]).to_undirected()
        a = list(A.nodes)[0]
        b = list(B.nodes)[0]
        ab = graphmol.edges[a, b]

        if a not in G.adj[b]:
            G.add_edge(b, a, **ab)

        return dag_to_smarts_nx(G, tag={x: i for i, x in enumerate([a, b], 1)})


    def dag_torsion_to_smarts_nx(A, B, C, D, graphmol):
        G = networkx.compose_all([A, B, C, D]).to_undirected()
        a = list(A.nodes)[0]
        b = list(B.nodes)[0]
        c = list(C.nodes)[0]
        d = list(D.nodes)[0]
        ab = graphmol.edges[a, b]
        bc = graphmol.edges[b, c]
        cd = graphmol.edges[c, d]

        if a not in G.adj[b]:
            G.add_edge(b, a, **ab)
        if c not in G.adj[b]:
            G.add_edge(b, c, **bc)
        if d not in G.adj[c]:
            G.add_edge(c, d, **cd)

        return dag_to_smarts_nx(G, tag={x: i for i, x in enumerate([a, b, c, d], 1)})


    def dag_outofplane_to_smarts_nx(A, B, C, D, graphmol):
        G = networkx.compose_all([A, B, C, D]).to_undirected()
        a = list(A.nodes)[0]
        b = list(B.nodes)[0]
        c = list(C.nodes)[0]
        d = list(D.nodes)[0]
        ba = graphmol.edges[b, a]
        bc = graphmol.edges[b, c]
        bd = graphmol.edges[b, d]

        if a not in G.adj[b]:
            G.add_edge(b, a, **ba)
        if c not in G.adj[b]:
            G.add_edge(b, c, **bc)
        if d not in G.adj[b]:
            G.add_edge(b, d, **bd)

        return dag_to_smarts_nx(G, tag={x: i for i, x in enumerate([a, b, c, d], 1)})


    def dag_angle_to_smarts_gt(A, B, C, graphmol):
        G = graph_tool.generation.graph_union(
            A,
            B,
            include=False,
            props=(
                (A.vp.primitives, B.vp.primitives),
                (A.ep.primitives, B.ep.primitives),
                (A.vp.parent_idx, B.vp.parent_idx),
            ),
        )
        G = graph_tool.generation.graph_union(
            G,
            C,
            include=True,
            props=(
                (G.vp.primitives, C.vp.primitives),
                (G.ep.primitives, C.ep.primitives),
                (G.vp.parent_idx, C.vp.parent_idx),
            ),
        )
        a = _get_root_v_gt(A)
        b = _get_root_v_gt(B)
        c = _get_root_v_gt(C)

        ab = graphmol.edges[a, b]
        bc = graphmol.edges[b, c]

        if a not in G.adj[b]:
            G.add_edge(b, a, **ab)
        if c not in G.adj[b]:
            G.add_edge(b, c, **bc)

        return dag_to_smarts_gt(G, tag={x: i for i, x in enumerate([a, b, c], 1)})


    def _get_root_v_gt(G):

        for v in G.iter_vertices():
            if v.in_neighbors == 0:
                return v


    def dag_angle_to_smarts_gt(A, B, C, graphmol):
        G = graph_tool.generation.graph_union(
            A,
            B,
            include=False,
            props=(
                (A.vp.primitives, B.vp.primitives),
                (A.ep.primitives, B.ep.primitives),
                (A.vp.parent_idx, B.vp.parent_idx),
            ),
        )
        G = graph_tool.generation.graph_union(
            G,
            C,
            include=True,
            props=(
                (G.vp.primitives, C.vp.primitives),
                (G.ep.primitives, C.ep.primitives),
                (G.vp.parent_idx, B.vp.parent_idx),
            ),
        )
        a = _get_root_v_gt(A)
        b = _get_root_v_gt(B)
        c = _get_root_v_gt(C)

        ab = graphmol.edge(b, a)
        bc = graphmol.edge(b, c)

        if ab not in G.edges():
            G.add_edge(b, a)
            G.ep.primitives[ab] = graphmol.ep.primitives[ab]
        if c not in G.adj[b]:
            G.add_edge(b, c, **bc)

        return dag_to_smarts_gt(G, tag={x: i for i, x in enumerate([a, b, c], 1)})


    def dag_to_smarts_gt(G, visited=None, todo=None, tag=None, source=None):

        smarts = ""

        if visited is None:
            visited = []
        if todo is None:
            todo = set()
        if tag is None:
            tag = {}

        # path = networkx.dag_longest_path(G)
        path = max_path_gt(G, visited=visited, source=source)
        [todo.add(i) for i in path]
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

                lhs, rhs = "", ""
                if nbr not in path:
                    lhs, rhs = "(", ")"

                    g = set(
                        x
                        for x in networkx.algorithms.dag.descendants(G, nbr)
                        if x not in visited
                    )
                    # g = networkx.algorithms.dag.descendants(G, nbr)
                    g.add(nbr)
                    bond_edge = G.adj[src][node_i]
                    bond = bond_edge["primitive"].to_smarts()
                    ret = dag_to_smarts_nx(
                        G.subgraph(g), visited=visited, tag=tag, todo=todo, source=nbr
                    )
                    if len(ret) > 0:
                        smarts += lhs + bond + ret + rhs
                else:
                    next_node = nbr
            if next_node is not None:
                lhs, rhs = "", ""
                nbr = next_node
                g = set(
                    x
                    for x in networkx.algorithms.dag.descendants(G, nbr)
                    if x not in visited
                )
                # g = networkx.algorithms.dag.descendants(G, nbr)
                g.add(nbr)
                bond_edge = G.adj[src][node_i]
                bond = bond_edge["primitive"].to_smarts()
                ret = dag_to_smarts_nx(
                    G.subgraph(g), visited=visited, tag=tag, todo=todo, source=nbr
                )
                if len(ret) > 0:
                    smarts += lhs + bond + ret + rhs

            src = node_i

        return smarts


    def _dag_descend_nx(G, visited, tag, source, target, branch, encloser=("", "")):

        lhs, rhs = encloser

        smarts = ""
        g = set(
            x for x in networkx.algorithms.dag.descendants(G, branch) if x not in visited
        )
        # g = networkx.algorithms.dag.descendants(G, nbr)
        g.add(branch)
        bond_edge = G.adj[source][target]
        bond = bond_edge["primitive"].to_smarts()
        ret = dag_to_smarts_nx(G.subgraph(g), visited=visited, tag=tag, source=branch)
        if len(ret) > 0:
            smarts += lhs + bond + ret + rhs
        return smarts


    def dag_to_smarts_nx(G, visited=None, tag=None, source=None):

        smarts = ""

        if visited is None:
            visited = []
        if tag is None:
            tag = {}

        # path = networkx.dag_longest_path(G)
        path = max_path_nx(G.to_undirected(), visited=visited, source=source)
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
                    smarts += _dag_descend_nx(
                        G, visited, tag, src, node_i, nbr, encloser=("(", ")")
                    )
                else:
                    next_node = nbr

            if next_node is not None:
                smarts += _dag_descend_nx(
                    G, visited, tag, src, node_i, next_node, encloser=("", "")
                )

            src = node_i

        return smarts


    def build_graph_mol_from_primitives_nx(bond_primitives):
        g = networkx.Graph()

        for (i, j), (atomi, bond, atomj) in bond_primitives.items():

            prim_i = offsb.chem.types.AtomType.from_smarts(atomi)
            prim_j = offsb.chem.types.AtomType.from_smarts(atomj)
            prim_b = offsb.chem.types.BondType.from_smarts(bond)
            g.add_node(i, primitive=prim_i)
            g.add_node(j, primitive=prim_j)
            g.add_edge(i, j, primitive=prim_b)

        return g


    def build_graph_mol_from_primitives_gt(bond_primitives):
        g = graph_tool.Graph(directed=False)

        g.vertex_properties["primitives"] = g.new_vertex_property("object")
        g.vertex_properties["parent_idx"] = g.new_vertex_property("int")
        g.edge_properties["primitives"] = g.new_edge_property("object")

        for (i, j), (atomi, bond, atomj) in bond_primitives.items():

            prim_i = offsb.chem.types.AtomType.from_smarts(atomi)
            vi = g.vertex(i, add_missing=True)
            g.vp.primitives[vi] = prim_i
            g.vp.parent_idx[vi] = i

            prim_j = offsb.chem.types.AtomType.from_smarts(atomj)
            vj = g.vertex(j, add_missing=True)
            g.vp.primitives[vj] = prim_j
            g.vp.parent_idx[vj] = j

            prim_e = offsb.chem.types.BondType.from_smarts(bond)
            e = g.add_edge(vi, vj)
            g.ep.primitives[e] = prim_e

        return g


    def build_graph_mol_from_smarts_nx(smarts):
        g = networkx.Graph(tags={})

        mol = rdkit.Chem.MolFromSmarts(smarts)

        for i, atom in enumerate(mol.GetAtoms()):
            idx = atom.GetIdx()
            atom_smarts = atom.GetSmarts()
            primitive = offsb.chem.types.AtomType.from_smarts(atom_smarts)
            g.add_node(idx, primitive=primitive)
            tag = atom.GetAtomMapNum()
            if tag > 0:
                g.graph["tags"][idx] = tag

        for b, bond in enumerate(mol.GetBonds()):
            bond_smarts = bond.GetSmarts()
            primitive = offsb.chem.types.BondType.from_smarts(bond_smarts)
            idx_i, idx_j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            g.add_edge(idx_i, idx_j, primitive=primitive)

        return g


    def build_graph_mol_from_smarts_gt(smarts):

        g = graph_tool.Graph(directed=False)
        g.vertex_properties["primitives"] = g.new_vertex_property("object")
        g.edge_properties["primitives"] = g.new_edge_property("object")

        g.graph_properties["tags"] = g.new_graph_property("object")
        g.graph_properties["tags"] = dict()

        mol = rdkit.Chem.MolFromSmarts(smarts)

        for i, atom in enumerate(mol.GetAtoms()):

            idx = atom.GetIdx()

            atom_smarts = atom.GetSmarts()
            primitive = offsb.chem.types.AtomType.from_smarts(atom_smarts)

            vi = g.vertex(idx, add_missing=True)
            g.vp.primitives[vi] = primitive

            tag = atom.GetAtomMapNum()
            if tag > 0:
                g.graph_properties["tags"][idx] = tag

        for b, bond in enumerate(mol.GetBonds()):

            bond_smarts = bond.GetSmarts()
            primitive = offsb.chem.types.BondType.from_smarts(bond_smarts)

            idx_i, idx_j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()

            vi, vj = g.vertex(idx_i), g.vertex(idx_j)
            e = g.add_edge(vi, vj)
            g.ep.primitives[e] = primitive

        return g


