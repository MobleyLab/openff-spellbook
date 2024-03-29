import logging
import sys
from abc import ABC, abstractmethod
import traceback

import tqdm

from offsb.tools.util import flatten_list
from offsb.treedi.node import DIRTY, Node

LOGSTDOUT = logging.StreamHandler(sys.stdout)
# FORMAT="[%(asctime)s] %(levelname)s <%(name)s::%(funcName)s>: %(message)s"
FORMAT = "%(message)s"
logging.basicConfig(format=FORMAT)

DEFAULT_DB = dict


def link_iter_depth_first(t):
    for c in t.link.values():
        yield from link_iter_depth_first(c)
    yield t


def link_iter_breadth_first(t):
    for c in t.link.values():
        yield c
    for c in t.link.values():
        yield from link_iter_breadth_first(c)


def link_iter_dive(t):
    yield t
    for c in t.link.values():
        yield from link_iter_dive(c)


def link_iter_to_root(t, select, state):
    yield t
    if t.source is not None:
        yield from link_iter_to_root(t.source, select, state)


class DatabaseKeyCollisionException:
    pass


class DebugDict(dict):

    # level = logging.DEBUG
    # level = logging.INFO
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def getLogger(self):
        if not (hasattr(self, "_logger") and self._logger is not None):
            self._logger = logging.getLogger("{}".format(id(self)))
            # self._logger.addHandler(LOGSTDOUT)
            self._logger.setLevel(level=logging.DEBUG)
            self._logger.debug("DebugDict online")

    def __setitem__(self, k, v):
        if LOG:
            self.getLogger()
            v0 = super().get(k, None)
            if v0 is not None:
                self._logger.debug(
                    "DBSWP {} KEY {} FROM {} TO {}".format(id(self), k, v0, v)
                )
                if DBSWP_IS_FATAL:
                    raise DatabaseKeyCollisionException()
            else:
                self._logger.debug("DBPUT {} KEY {} VAL {}".format(id(self), k, v))
        super().__setitem__(k, v)

    def __getitem__(self, k):
        if LOG:
            self.getLogger()
        try:
            v = super().__getitem__(k)
        except KeyError as e:
            if LOG:
                self._logger.debug("DBGET {} KEY {} KeyError".format(id(self), k))
            raise
        if LOG:
            self._logger.debug("DBGET {} KEY {} VAL {}".format(id(self), k, v))
        return v

    def get(self, k):
        v = super().get(k)
        if LOG:
            self.getLogger()
            self._logger.debug("DBGET {} KEY {} VAL {}".format(id(self), k, v))
        return v


LOG = False
DBSWP_IS_FATAL = False
# TREELOGLEVEL = logging.DEBUG
TREELOGLEVEL = logging.INFO

DEFAULT_DB = dict
# DEFAULT_DB=DebugDict


class Tree(ABC):
    index = 0
    """ 
    A structure that holds an index for fast lookups.
    Also holds associations for links. Linking will create a new tree with
    empty payloads; will keep IDs

    """

    def __init__(
        self,
        name,
        root_payload=None,
        node_index=None,
        db=None,
        index=None,
        payload=None,
        dereference=False,
        verbose=True,
    ):

        self.dereference = dereference
        self.verbose = verbose

        if db is not None:
            self.db = db
        else:
            self.db = DEFAULT_DB()

        if root_payload is not None:
            ROOT = "ROOT"
            root = Node(name=ROOT)
            root.payload = ROOT

            self.db[ROOT] = {"data": root_payload}

            root.index = "0"
            self.root_index = root.index
            self.node_index = {root.index: root}
        elif node_index is not None:
            self.node_index = node_index
            self.root_index = self.source.root_index
        else:
            self.node_index = DEFAULT_DB()

        self.processes = 1
        self.link = {}
        self.name = name
        self.modified = set()
        self.N = 1
        if index is None:
            self.index = "TREE-" + str(Tree.index)
            Tree.index += 1
        else:
            self.index = str(index)
        # self.node_index = nodes
        # for node in self.node_index:
        #    self.node_index.get( node).tree = self.name
        self.n_levels = (
            0 if len(self.node_index) == 0 else 1
        )  # max( [node_level( node) for node in nodes.values()])
        self.n_nodes = len(self.node_index)
        # self.root = None if self.n_levels == 0 else get_root( list(nodes.values())[0])
        self.ID = ".".join([self.index, self.name])

        self.logger = logging.getLogger("{}::{}".format(str(type(self)), self.ID))
        # self.logger.addHandler(LOGSTDOUT)
        self.logger.setLevel(level=TREELOGLEVEL)
        self.logger.debug("Tree database is {}".format(str(type(self.db))))

    def to_pickle(self, db=True, index=True, name=None):

        import pickle

        if name is None:
            name = self.name + ".p"
        # self.isolate()
        if not db:
            tmp = self.db
            self.db = None
        if not index:
            tmp_index = self.node_index
            self.node_index = None
        with open(name, "wb") as fid:
            pickle.dump(self, fid)
        if not db:
            self.db = tmp
        if not index:
            self.node_index = tmp_index

    @abstractmethod
    def to_pickle_str(self):
        pass

    def set_root(self, root):
        self.node_index = {
            node.index: node
            for node in self.node_iter_depth_first(root, dereference=False)
        }
        # self.n_levels = max( [node_level( node) for node in self.node_index.values()])
        # self.n_nodes = node_descendents( root) + 1
        self.root = root
        self.root.tree = self.name

    def register_modified(self, node, state=DIRTY):
        self.modified.add(node.index)
        node.set_state(state)

    def link_tree(self, tree):
        self.link[tree.index] = tree
        tree.source = self

    def link_generate(self):
        tree = self.copy_skel()
        self.link_tree(tree)
        tree.source = self
        return tree

    def root(self):
        return self[self.root_index]

    def __setitem__(self, k, v):
        if self.dereference:
            self.db[self.node_index[k]]["data"] = v
            self.db[self.node_index[k]]["type"] = type(v)
        else:
            self.node_index[k] = v

    def __getitem__(self, k):
        return self.node_index[k]

    def join(self, nodes, fn=link_iter_breadth_first):
        ret = {}
        if not hasattr(nodes, "__iter__"):
            nodes = [nodes]
        for node in nodes:
            ret[node.index] = {}
            for tree in fn(self):
                ret[node.index][tree.name] = tree[node.index]
        return ret

    def copy_skel(self):
        """ provides a new tree with no payload or links """
        nodes = {
            node.index: node
            for node in [node.skel() for node in self.node_index.values()]
        }
        for node in nodes.values():
            [node.add(nodes[v.index]) for v in self[node.index].children]
        tree = Tree(nodes=nodes, name=self.name)
        [tree.register_modified(node) for node in self.node_index.values()]
        return tree

    def add(self, parent_index, child, index=None):
        """
        takes a parent node index and a fresh constructed node
        inserts the payload into the db and creates a reference in the node
        """

        assert self.N not in self.node_index
        assert isinstance(parent_index, str)
        # print("Adding", self[parent_index], "->", child)

        parent = self[parent_index]
        child.index = str(self.N)  # + '-' + child.index
        self.N += 1
        parent.add(child, index=index)
        child.tree = self.name
        self[child.index] = child
        return child
        # self.obj_index.update( child.payload)
        # self.node_index[child.] = child
        # self.n_levels = max(self.n_levels, 1 + node_level( parent))
        # self.n_nodes += 1
        # self.register_modified( child, state=NEW)

        # for tree in self.link.values():
        #    tree.add( parent_index, child.skel())

    def assemble(self):
        for ID in self.node_index:
            node = self[ID]
            if isinstance(node.parent, str):
                node.parent = self[node.parent]
            for i, _ in enumerate(node.children):
                if isinstance(node.children[i], str):
                    node.children[i] = self[node.children[i]]
        if isinstance(self.root, str):
            self.root = self[self.root]

    def _yield(self, v, dereference=None):
        dereference = self.dereference if dereference is None else dereference
        if dereference:
            ret = self.db.get(v.payload, {}).get("data")
            if ret:
                yield ret
        else:
            yield v

    def yield_if_single(self, v, select, state, dereference=None):
        if select is None or select == v.name:
            if state is None or state == v.state:
                yield from self._yield(v, dereference=dereference)

    def yield_if(self, v, select, state, dereference=None):
        if hasattr(v, "__iter__"):
            for vi in v:
                yield from self.yield_if(vi, select, state, dereference=dereference)
        else:
            yield from self.yield_if_single(v, select, state, dereference=dereference)

    def node_iter_depth_first_single(
        self, v, select=None, state=None, dereference=None
    ):
        for c in v.children:
            c = self[c]
            yield from self.node_iter_depth_first_single(
                c, select, state, dereference=dereference
            )
        yield from self.yield_if(v, select, state, dereference=dereference)

    def node_iter_depth_first(self, v, select=None, state=None, dereference=None):
        if hasattr(v, "__iter__"):
            for vi in v:
                yield from self.node_iter_depth_first(
                    vi, select, state, dereference=dereference
                )
        else:
            yield from self.node_iter_depth_first_single(
                v, select, state, dereference=dereference
            )

    def node_iter_breadth_first_single(
        self, v, select=None, state=None, dereference=None
    ):
        if v.parent is None:
            self._yield(v, dereference=dereference)
        for c in v.children:
            c = self[c]
            yield from self.yield_if(c, select, state, dereference)
        for c in v.children:
            c = self[c]
            yield from self.node_iter_breadth_first_single(
                c, select, state, dereference=dereference
            )

    def node_iter_breadth_first(self, v, select=None, state=None, dereference=None):
        if hasattr(v, "__iter__"):
            for vi in v:
                yield from self.node_iter_breadth_first(
                    vi, select, state, dereference=dereference
                )
        else:
            yield from self.node_iter_breadth_first_single(
                v, select, state, dereference=dereference
            )

    def node_iter_dive_single(self, v, select=None, state=None, dereference=None):
        yield from self.yield_if_single(v, select, state, dereference=dereference)
        for c in v.children:
            c = self[c]
            yield from self.node_iter_dive_single(
                c, select, state, dereference=dereference
            )

    def node_iter_dive(self, v, select=None, state=None, dereference=None):
        if hasattr(v, "__iter__"):
            for vi in v:
                yield from self.node_iter_dive(
                    vi, select, state, dereference=dereference
                )
        else:
            yield from self.node_iter_dive_single(
                v, select, state, dereference=dereference
            )

    def node_iter_to_root_single(self, v, select=None, state=None, dereference=None):
        yield from self.yield_if_single(v, select, state, dereference=dereference)
        if v.parent is not None:
            parent = self[v.parent]
            yield from self.node_iter_to_root_single(
                parent, select, state, dereference=dereference
            )

    def node_iter_to_root(self, v, select=None, state=None, dereference=None):
        if hasattr(v, "__iter__"):
            for vi in v:
                yield from self.node_iter_to_root(
                    vi, select, state, dereference=dereference
                )
        else:
            yield from self.node_iter_to_root_single(
                v, select, state, dereference=dereference
            )

    def get_root(self, node):
        if node.parent is None:
            return node
        return self.get_root(self[node.parent])

    def node_depth(self, node):
        n = node
        l = 0
        while n.parent is not None:
            n = self[n.parent]
            l += 1
        return l

    def node_descendents(self, node):
        if len(node.children) == 0:
            return 0
        return 1 + sum([self.node_descendents(self[v]) for v in node.children])


class PartitionTree(Tree):
    """A parition tree holds indices and applies them to nodes
    Importantly, partition trees are not trees...
    They are a dictionary, where it copies the source tree index as keys,
    and puts data in there
    """

    def __init__(self, source_tree, name, verbose=True):
        self.source = source_tree

        super().__init__(node_index=source_tree.node_index, name=name, verbose=verbose)
        self.logger.debug("Building PartitionTree {}".format(name))

    def to_pickle(self, name=None, index=False, db=True):
        import pickle

        if not index:
            tmp = self.node_index
            self.node_index = None
        tmp_source = self.source
        self.source = tmp_source.ID
        if not db:
            tmp_db = self.db
            self.db = None
        super().to_pickle(name=name, db=db)
        self.source = tmp_source
        if not index:
            self.node_index = True
        if not db:
            self.db = tmp_db

    def to_pickle_str(self):
        import pickle

        return pickle.dumps(self)

    def apply(self):
        pass


class TreeOperation(PartitionTree):
    def __init__(self, source_tree, name, verbose=True):
        super().__init__(source_tree, name, verbose=verbose)
        self.logger.debug("This tree is TreeOperation")

    @abstractmethod
    def op(self, node, partition):
        pass

    @abstractmethod
    def _unpack_result(self, val):
        pass

    @abstractmethod
    def _generate_apply_kwargs(self, i, target, kwargs={}):
        pass

    def _apply_initialize(self, targets):
        pass

    def _apply_finalize(self, targets):
        pass

    def _print_result(self, n, n_targets, tgt, ret):
        # print("{:d} / {:d} {:s}".format(n, n_targets, str(tgt)))
        if ret is not None and ret != "":
            self.logger.info(ret)

    def _unpack_result_common(self, val, n, n_targets):
        for tgt, ret in val.items():
            if tgt == "return":
                self._unpack_result(ret)
            elif self.verbose and tgt != "debug":
                self._print_result(n, n_targets, tgt, ret)
            if tgt == "debug":
                if ret != "":
                    self.logger.debug(ret)

    def _unpack_work(self, work, exceptions_are_fatal=True):

        n_work = len(work)
        for n, future in tqdm.tqdm(
            enumerate(work, 1),
            total=n_work,
            ncols=80,
            disable=not self.verbose,
            desc=self.name,
        ):
            try:
                val = future.get()
            except RuntimeError as e:
                self.logger.error("RUNTIME ERROR; race condition??")
                self.logger.error(str(e))
                if exceptions_are_fatal:
                    raise
            if val is None:
                self.logger.info("data is None?!?")
                continue

            self._unpack_result_common(val, n, n_work)

    def _apply_parallel(self, targets, exceptions_are_fatal=True):

        from multiprocessing.pool import Pool as Pool

        exe = Pool(processes=self.processes)

        try:
            work = []
            for i, tgt in tqdm.tqdm(
                enumerate(targets, 1), ncols=80, desc="Submit", disable=False
            ):
                kwargs = self._generate_apply_kwargs(i, tgt)
                task = exe.apply_async(self.apply_single, (i, tgt), kwargs)
                work.append(task)

            self._unpack_work(work)
        except Exception as e:
            self.logger.error("Exception!")
            traceback.print_exc()
            self.logger.error(e)
            breakpoint()
            # for concurrent, use this and shutdown(wait=False) instead
            # [job.cancel() for job in work]
            exe.terminate()
        else:
            exe.close()

    def apply(self, select, targets=None):
        if targets is None:
            root = self.source.source.root()
            targets = list(
                self.source.source.node_iter_depth_first(
                    root, select=select, dereference=False
                )
            )
        elif not hasattr(targets, "__iter__"):
            targets = [targets]

        # since we can only operate on self._select, force it
        targets = flatten_list(
            [
                self.source.source.node_iter_depth_first(
                    x, select=select, dereference=False
                )
                for x in targets
            ]
        )

        # expand if a generator
        targets = list(targets)

        n_targets = len(targets)

        self._apply_initialize(targets)

        if self.processes is None or self.processes > 1:
            self._apply_parallel(targets)

        elif self.processes == 1:

            for n, target in tqdm.tqdm(
                enumerate(targets, 1),
                total=len(targets),
                ncols=80,
                disable=not self.verbose,
                desc=self.name,
            ):
                o = str(target)
                self.logger.debug("Begin processing target {}".format(o))
                kwargs = self._generate_apply_kwargs(n, target, kwargs=DEFAULT_DB())
                val = self.apply_single(n, target, **kwargs)
                self._unpack_result_common(val, n, n_targets)

        self._apply_finalize(targets)

    # def apply(self, targets=None, select="Molecule"):
    #     calcs = 0
    #     self.source.apply(targets=targets)
    #     if targets is None:
    #         entries = list(self.source.source.iter_entry())
    #     else:
    #         entries = targets
    #     if not hasattr(entries, "__iter__"):
    #         entries = [entries]

    #     for entry in entries:
    #         mol_calcs = 0
    #         obj = self.source.db[self.source[entry.index].payload]

    #         masks = obj["data"]

    #         for mol_node in self.node_iter_depth_first(entry, select=select):
    #             mol = self.source.source.db[mol_node.payload]
    #             for mask in masks:
    #                 ret = {}
    #                 if mol_node.payload in self.db:
    #                     ret = self.db.get( mol_node.payload)
    #                 else:
    #                     self.db[mol_node.payload] = dict()

    #                 ret[tuple(mask)] = self.op( mol["data"], [i-1 for i in mask])
    #                 self.db[mol_node.payload].update( ret)
    #                 mol_calcs += 1

    #         calcs += mol_calcs
    #     print(self.name + " calculated: {}".format( calcs))
