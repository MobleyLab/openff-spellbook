#!/usr/bin/env python3 

from datetime import datetime
import numpy as np
from ..tools.util import flatten_list

import qcfractal.interface as ptl
import treedi.tree as Tree
import treedi.node as Node
from treedi.tree import DebugDict

from .qca_comparisons import match_canonical_isomeric_explicit_hydrogen_smiles

DEFAULT_DB = DebugDict
DEFAULT_DB = dict


def wrap_fn(fn, ids, i, j, **kwargs):
    """
    A wrapper for the QCA batch downloader
    """

    out_str = kwargs['out_str']
    kwargs.pop('out_str')
    prestr = "\r{:20s} {:4d} {:4d}     ".format(out_str, i, j)

    elapsed = datetime.now()
    objs = [dict(obj) for obj in fn(ids, **kwargs)]
    elapsed = str(datetime.now() - elapsed)

    N = len(objs)
    poststr = "... Received {:6d}  | elapsed: {:s}".format(N, elapsed)

    out_str = prestr + poststr
    return objs, out_str, N


def wrap_fn_hessian(fn, i, j, **kwargs):
    """
    A wrapper for the QCA batch downloader for Hessians
    """

    out_str = kwargs['out_str']
    kwargs.pop('out_str')
    prestr = "\r{:20s} {:4d} {:4d}     ".format(out_str, i, j)

    elapsed = datetime.now()
    objs = [dict(obj) for obj in fn(**kwargs)]
    elapsed = str(datetime.now() - elapsed)

    N = len(objs)
    poststr = "... Received {:6d}  | elapsed: {:s}".format(N, elapsed)

    out_str = prestr + poststr
    return objs, out_str, N


class QCATree( Tree.Tree):

    def __init__(self, obj, root_payload=None, node_index=None, db=None,
                  payload=None):
        print("Building QCATree")
        if isinstance(obj, str):
            super().__init__(obj, root_payload=root_payload,
                             node_index=node_index, db=db, payload=payload)

    def from_QCATree(self, name, obj):
        if isinstance(obj, type(self)):
            rpl = obj.db["ROOT"]
            nodes = obj.node_index.copy()
            db = obj.db.copy()
            pl = obj.root().payload
            newtree = QCATree(name, root_payload=rpl, node_index=nodes,
                db=db, payload=pl)
            return newtree
        else:
            raise Exception("Tried to call from_QCATree(obj) and obj is not a QCATree")


    def vtable( self, obj):
        """
        This a mapping between QCArchive objects and the function that
        knows how to crawl it. 
        
        Should do this in a more elegant way using e.g. dispatching
        """
        from qcfractal.interface.collections.torsiondrive_dataset \
            import TorsionDriveDataset
        from qcfractal.interface.models.torsiondrive \
            import TorsionDriveRecord 

        from qcfractal.interface.collections.optimization_dataset \
            import OptimizationDataset
        from qcfractal.interface.models.records \
            import OptimizationRecord

        from qcfractal.interface.collections.gridoptimization_dataset \
            import GridOptimizationDataset
        from qcfractal.interface.models.gridoptimization \
            import GridOptimizationRecord

        if isinstance(obj, TorsionDriveDataset):
            return self.branch_torsiondrive_ds

        elif isinstance(obj, TorsionDriveRecord): 
            return self.branch_torsiondrive_record

        elif isinstance(obj, OptimizationDataset):
            return self.branch_optimization_ds

        elif isinstance(obj, OptimizationRecord):
            return self.branch_optimization_record

        elif isinstance(obj, GridOptimizationDataset):
            return self.branch_gridopt_ds

        elif isinstance(obj, OptimizationRecord):
            return self.branch_gridopt_record

        raise ValueError("QCA type '" + str(type(obj)) + "' not understood")
        return None

    def to_pickle_str(self):
        import pickle
        self.isolate()
        return pickle.dumps(self)

    def combine_by_entry(self,
            fn=match_canonical_isomeric_explicit_hydrogen_smiles,
            targets=None):
        """
        compare entries using fn, and collect into a parent node
        fn is something that compares 2 entries
        returns a node where children match key
        """
        new_nodes = []

        if targets is None:
            entries = list(self.node_iter_depth_first(self.root(), select="Entry"))
        elif hasattr(targets, "__iter__"):
            entries = list(targets)
        else:
            entries = [targets]
        if len(entries) == 0:
            return new_nodes

        used = set()
        for i in range(len(entries)):
            if i in used:
                continue
            ref = entries[i].copy()
            ref_obj = self.db[ref.payload]['data']
            used.add(i)

            node = Node.Node(name="Folder", index="", payload=repr(fn))
            node.add(ref)

            for j in range(i + 1, len(entries)):
                entry = entries[j].copy()
                entry_obj = self.db[entry.payload]['data']
                if fn(ref_obj, entry_obj):
                    node.add(entry)
                    used.add(j)
            new_nodes.append(node)

        return new_nodes

    def isolate(self):
        for link in self.link:
            self.link.__setitem(link, self.link.ID)

    def associate(self, link_trees):
        for link in link_trees:
            self.link[link.ID] = link

    def _obj_is_qca_collection(self, obj):
        """ TODO """
        return True

    def build_index(self, ds, drop=None, **kwargs):
        """
        Take a QCA DS, and create a node for it.
        Then expand it out.
        """
        assert self._obj_is_qca_collection(ds)

        ds_id = 'QCD-' + str(ds.data.id)
        # the object going into the data db
        pl = {'data': ds}
        self.db[ds_id] = pl

        # create the index node for the tree and integrate it
        ds_node = Node.Node( name=ds.data.name, payload=ds_id )
        ds_node = self.add( self.root_index, ds_node)

        self.expand_qca_dataset_as_tree( ds_node.index, skel=True, drop=drop,
                **kwargs)

    def expand_qca_dataset_as_tree(self, nid, skel=False, drop=None, **kwargs):
        """
        Recursively visit the objects in the dataset, and build an index.
        If skel is True, just build the lightweight index without downloading
        any heavy data (e.g. coordinates). If skel is False, it will download
        full objects as they are found (e.g. not using any projection).

        Drop specificies which records to ignore, for example ["Hessian"] will
        prevent searching for any Hessians, which is usually true for torsion
        drives. This is useful since Hessian searches are somewhat costly.

        Drop strings correspond to the names applied to the nodes in this order:
            "DS name" (The name of the dataset)
            "Specification"
            "TorsionDrive" / "GridOpt" (if present in the dataset)
            "Optimization"
            "Gradient"
            "Molecule"
            "Hessian"

        """

        if drop is not None:
            self.drop = drop
        else:
            self.drop = []

        node = self[nid]
        payload = self
        oid = node.payload
        obj = self.db[oid]["data"]

        fn = self.vtable(obj)
        fn(nid, skel=skel, **kwargs)

        self.drop = []

    def node_iter_entry( self, nodes=None, select=None, 
            fn=Tree.Tree.node_iter_depth_first):

        # Assume here that level 3 are the entries
        if nodes is None:
            nodes = [self.root()]
        elif not hasattr( nodes, "__iter__"):
            nodes = [nodes]
        for top_node in nodes:
            yield from fn( self, top_node, select="Entry")
                # if node.payload not in self.db:
                #     continue
                # obj = self.db[node.payload]
                # if "entry" in obj:
                #     yield node

    def iter_entry( self, select=None):
        yield from self.node_iter_entry(self.root(), select=select)


    def cache_torsiondriverecord_minimum_molecules(self, tdr_nodes=None):
        if tdr_nodes is None:
            tdr_nodes = self.iter_entry( select="TorsionDrive")
        fn = __class__.node_iter_torsiondriverecord_minimum
        return self.cache_minimum_molecules( tdr_nodes, fn)

    def cache_final_molecules( self, nodes=None):
        """
        Visit the entries and download the initial molecules of each procedure
        """

        mols = dict()
        if nodes is None:
            nodes = self.node_iter_dive(self.root(), select="Optimization")
        for opt_node in nodes:
            opt = self[opt_node.payload]
            final_mol_id = opt["data"]["final_molecule"]

            if not final_mol_id is None:
                final_mol_id = "QCM-" + str(final_mol_id)
                mols.append( final_mol_id)

        if len(mols) == 0:
            print("No final molecules found!")
            return
        mol_ids_flat = mols 

        client = self.db["ROOT"]["data"]
        fresh_obj_map = self.batch_download(mol_ids_flat, client.query_molecules)

        for key, val in fresh_obj_map.items():
            if key not in self.db:
                self.db[key] = DEFAULT_DB({ "data": val })
            else:
                self.db[key]["data"] = val


    def cache_initial_molecules( self):
        """
        Visit the entries and download the initial molecules of each procedure
        """

        mols = dict()
        for entry in self.iter_entry():
            eobj = self.db[entry.payload]['data'].__dict__
            if 'initial_molecule' in eobj:
                init_mol_id = eobj["initial_molecule"]
            elif 'initial_molecules' in eobj:
                init_mol_id = list(eobj["initial_molecules"])

            if isinstance( init_mol_id, list):
                init_mol_id = init_mol_id[0]

            init_mol_id = "QCM-" + str(init_mol_id)
            mols[ entry.index] = init_mol_id

        if len(mols) == 0:
            print("No initial molecules found!")
            return
        mol_ids_flat = flatten_list([x for x in mols.values()], -1)

        client = self.db["ROOT"]["data"]
        fresh_obj_map = self.batch_download(mol_ids_flat, client.query_molecules)

        for key, val in fresh_obj_map.items():
            if key not in self.db:
                self.db[key] = DEFAULT_DB({ "data": val })
            else:
                self.db[key]["data"] = val


    def cache_minimum_molecules( self, nodes, fn):

        if not hasattr( nodes, "__iter__"):
            nodes = [nodes]

        # make a list since we iterate it twice. if its a generator the first 
        # iteration would consume the generator
        nodes = list(nodes)
        mols = {}
        for top_node in nodes:
            mols[top_node.index] = []
            for node in fn(self, top_node, select="MoleculeStub"):
                mols[top_node.index].append(node.payload)
            if mols[top_node.index] == []:
                mols.pop(top_node.index)
        if len(mols) == 0:
            print("No molecules found")
            return 0

        mol_ids_flat = flatten_list([x for x in mols.values()], -1)
        client = self.db["ROOT"]["data"]

        print( "Downloading", len(mol_ids_flat), "minimum molecules using", fn)
        fresh_obj_map = self.batch_download(mol_ids_flat, client.query_molecules)

        for top_node in nodes:

            found=0
            #candidates = sum(
            #    [1 for _ in fn(self, top_node, select="MoleculeStub")])

            for node in fn(self, top_node, select="MoleculeStub"):
                mol = fresh_obj_map[node.payload]
                assert mol is not None

                self.db[node.payload] = DEFAULT_DB({"data": mol})
                node.name = "Molecule"
                node.stamp = datetime.now()
                self.register_modified(node, state=Node.CLEAN)
        return len(mol_ids_flat)

    def cache_hessians( self, nodes):
        self.cache_results( nodes, select="HessianStub")

    def cache_results( self, nodes, 
            fn=Tree.Tree.node_iter_depth_first, select=None):
        """
        Generate method to cache results depending on select.
        """
        
        if nodes is None:
            nodes = self.root()
        elif not hasattr( nodes, "__iter__"):
            nodes = [nodes]

        results = {}
        for top_node in nodes:
            results[ top_node.index] = []
            for node in fn( self, top_node, select=select):
                results[ top_node.index].append( node.payload)
            if results[ top_node.index] == []:
                results.pop( top_node.index)
        if len(results) == 0:
            return
        ids_flat = flatten_list([x for x in results.values()], -1)
        client = self.db["ROOT"]["data"]

        print("Caching results using iterator", str(fn), "on", len(ids_flat))
        fresh_obj_map = self.batch_download( ids_flat, client.query_results)

        for top_node in nodes:
            for node in fn( self, top_node, select=select):
                result = fresh_obj_map[ node.payload]
                assert result is not None

                self.db[node.payload] = DEFAULT_DB({ "data": result })
                self.register_modified( node, state=Node.CLEAN)
                if "Stub" in node.name:
                    node.name = node.name[:-4]
                node.stamp = datetime.now()

    def cache_optimization_minimum_molecules( self, nodes=None):

        if nodes is None:
            nodes = [self.root()]
        fn = __class__.node_iter_optimization_minimum
        return self.cache_minimum_molecules( nodes, fn)

    def batch_download_hessian_parallel( self, full_ids, fn, max_query=1000, projection=None):
        import math
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool()
        chunks = []
        L = len(full_ids)
        if max_query > 1000:
            max_query = 1000

        # Hessian queries take awhile, and some timeouts seem to occur
        # Reduce this a bit since there is no fault tolerance yet
        max_query=100

        # Assemble the chunk lengths
        n_chunks = math.ceil(L / max_query)
        for i in range(n_chunks):
            j = (i+1)*max_query
            if j > L:
                j = L
            if i*max_query == j:
                break
            chunks.append((i*max_query, j))

        result = []
        objs = []
        out_str = "Chunk "
        if projection is not None:
            out_str += "with projection "
        kwargs = {'out_str': out_str }

        from datetime import datetime, timedelta
        total=datetime.now()

        # Strip the [A-Z]{3}- identifier before sending off to the QCA server
        if len(full_ids[0].split("-")) > 1:
            ids = [x.split("-")[-1] for x in full_ids]
            id_suf = full_ids[0].split("-")[0] + "-"
        else:
            ids = full_ids
            id_suf = ""

        for i,j in chunks:
            kwargs['molecule'] = ids[i:j]
            args = (fn, i, j)
            result.append(pool.apply_async(wrap_fn_hessian, args, kwargs))

        total_received = 0
        for ret in result:
            ret = ret.get()
            objs += ret[0]
            print('\r'+ret[1], end='')
            total_received += ret[2]

        pool.close()

        print("\nTotalTime: {:s} Received: {:d}\n".format(
            str( datetime.now() - total), total_received), end="")

        # Reapply the stripped identifier
        obj_map = {id_suf + str(obj['id']): obj for obj in objs}
        return obj_map

    def batch_download_hessian( self, full_ids, fn, max_query=1000, projection=None):
        return self.batch_download_hessian_parallel( full_ids, fn,
            max_query=max_query, projection=projection)
        import math
        chunks = []
        L = len(full_ids)
        if max_query > 1000:
            max_query = 1000

        # Assemble the chunk lengths
        n_chunks = math.ceil(L / max_query)
        for i in range(n_chunks):
            j = (i+1)*max_query
            if j > L:
                j = L
            if i*max_query == j:
                break
            chunks.append((i*max_query, j))

        objs = []
        out_str = "Chunk "
        if projection is not None:
            out_str += "with projection "
        from datetime import datetime, timedelta
        total=datetime.now()

        # Strip the [A-Z]{3}- identifier before sending off to the QCA server
        if len(full_ids[0].split("-")) > 1:
            ids = [x.split("-")[-1] for x in full_ids]
            id_suf = full_ids[0].split("-")[0] + "-"
        else:
            ids = full_ids
            id_suf = ""
        for i,j in chunks:
            if True:
                print( "\r{:20s} {:4d} {:4d}     ".format(out_str, i, j), end="")
            elapsed = datetime.now()

            objs += [dict(obj) for obj in \
                fn( molecule=ids[i:j], driver="hessian", \
                    limit=max_query, include=projection)]

            elapsed = str(datetime.now() - elapsed)
            if True:
                print( "... Received {:6d}  | elapsed: {:s}".format( \
                    len(objs), elapsed), end="")

        print("\nTotalTime: {:s}\n".format( str( datetime.now() - total)), end="")

        # Reapply the stripped identifier
        obj_map = { id_suf + str(obj['id']): obj for obj in objs }
        return obj_map


    def batch_download_parallel( self, full_ids, fn, max_query=1000, \
                        procedure=None, projection=None):
        from multiprocessing import Pool, TimeoutError
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(10)
        import math
        chunks = []
        L = len(full_ids)
        if max_query > 1000:
            max_query = 1000
        n_chunks = math.ceil( L / max_query)
        for i in range(n_chunks):
            j = (i+1)*max_query
            if j > L:
                j = L
            if i*max_query == j:
                break
            chunks.append( (i*max_query, j))

        objs = []
        out_str = "Chunk "
        if projection is not None:
            out_str += "with projection "
        from datetime import datetime, timedelta
        total=datetime.now()
        ids = [x.split("-")[1] for x in full_ids]
        id_suf = full_ids[0].split("-")[0] + "-"
        result = []
        kwargs = {'out_str': out_str}

        if procedure is None:
            if projection is None:
                kwargs['limit'] = max_query
            else:
                kwargs['limit'] = max_query
                kwargs['include'] = projection
        else:
            kw = { "limit": max_query, "procedure": procedure }
            if projection is not None:
                kw.__setitem__( "projection", projection)
            kwargs.update(kw)

        for i,j in chunks:
            args = (fn, ids[i:j], i, j)
            result.append(pool.apply_async(wrap_fn, args, kwargs))

        total_received = 0
        for ret in result:
            ret = ret.get()
            objs += ret[0]
            print('\r',ret[1], end='')
            total_received += ret[2]

        pool.close()
        obj_map = {id_suf + str(obj.get('id')): obj for obj in objs}
        print("\nTotalTime: {:s} Received: {:d}\n".format(
            str( datetime.now() - total), total_received), end="")
        return obj_map

    def batch_download( self, full_ids, fn, max_query=1000, \
                        procedure=None, projection=None):
        return self.batch_download_parallel(full_ids, fn, max_query=max_query,
            procedure=procedure, projection=projection)
        import math
        chunks = []
        L = len(full_ids)
        if max_query > 1000:
            max_query = 1000
        n_chunks = math.ceil( L / max_query)
        for i in range(n_chunks):
            j = (i+1)*max_query
            if j > L:
                j = L
            if i*max_query == j:
                break
            chunks.append( (i*max_query, j))

        objs = []
        out_str = "Chunk "
        if projection is not None:
            out_str += "with projection "
        from datetime import datetime, timedelta
        total=datetime.now()
        ids = [x.split("-")[1] for x in full_ids]
        id_suf = full_ids[0].split("-")[0] + "-"
        for i,j in chunks:
            #if i == 0:
            if True:
                print( "\r{:20s} {:4d} {:4d}     ".format(out_str, i, j), end="")
            elapsed = datetime.now()
            if procedure is None:
                if projection is None:
                    objs += [dict(obj) for obj in fn( ids[i:j], limit=max_query)]
                else:
                    objs += [dict(obj) for obj in fn( ids[i:j], \
                                                      limit=max_query, \
                                                      include=projection)]
            else:
                kw = { "limit": max_query, "procedure": procedure }
                if projection is not None:
                    kw.__setitem__( "projection", projection)
                objs += [ dict( obj) for obj in fn( ids[i:j], **kw)]
                kw = None

            elapsed = str(datetime.now() - elapsed)
            #if i == 0:
            if True:
                print( "... Received {:6d}  | elapsed: {:s}".format( \
                    len(objs), elapsed), end="")
        print("\nTotalTime: {:s}\n".format( str( datetime.now() - total)), end="")
        obj_map = {id_suf + str(obj.get('id')): obj for obj in objs}
        return obj_map



    def branch_ds( self, nid, name, fn, skel=False, start=0, limit=0, keep_specs=None):
        """ Generate the individual entries from a dataset """

        if "Entry" in self.drop:
            return
        suf = "QCP-"
        ds_node = self[nid]
        ds = self.db[ds_node.payload]["data"]
        ds_specs = ds.data.specs
        records = ds.data.records
        if limit > 0 and limit < len(records):
            records = dict(list(records.items())[start:start+limit])

        client = self.db["ROOT"]["data"]

        specs = [list(entry.object_map.keys()) for entry in records.values()]
        specs = list(set(flatten_list(specs, -1)))
        print("Specs gathered:")
        [print("    ",x) for x in specs]
        if not keep_specs is None:
            print("Specs wanted", keep_specs)
            specs = [s for s in specs if s in keep_specs]
        #specs = list(set(flatten_list([ x for x in specs if len(x) > 0 ])))

        spec_map_ids = {}

        for spec in specs:
            new_ids = [suf + str(entry.object_map[spec]) 
                for entry in records.values() if spec in entry.object_map]
            spec_map_ids[spec] =  new_ids 

        print("Specs found ({:d}) : {}\n".format(len(specs),specs))
        #[suf + str(entry.object_map[spec]) 
        #    for entry in records.values() if spec in entry.object_map]
        ids = flatten_list(list(spec_map_ids.values()), times=-1)
        print("Downloading", name, "information for", len( ids))

        # returns an inverse map, where the values are keys to the dl data
        obj_map = self.batch_download( ids, client.query_procedures) 

        # retreive all of the initial molecule IDs
        init_mol_ids = [obj["initial_molecule"] for obj in obj_map.values()]
        init_mols_are_lists = False
        if isinstance( init_mol_ids[0], list):
            init_mols_are_lists = True
            # init_mol_ids = [ str(x[0]) for x in init_mol_ids]
            init_mol_ids = flatten_list( init_mol_ids)
        init_mol_ids = ["QCM-" + x for x in init_mol_ids]

        print("Downloading", name, "initial molecules for  for", len( init_mol_ids))
        init_mol_map = self.batch_download( init_mol_ids, client.query_molecules)
        
        # add this entry node
        nodes = []
        for rec in records:
            # add ds.data.id to ensure that entries are unique across dict
            # TODO this should probably have server/db info as well
            # since two mirror servers could have same ids but different data

            pl_name = "QCE." + ds.data.id + "-" + str(rec)
            if pl_name in self.db:
                if records[rec].name == self.db[pl_name]['data'].name:
                    # Not sure why, but this Entry was already added.
                    # Just assume we can skip it.
                    continue
            node = Node.Node(name="Entry", payload=pl_name)
            node = self.add( ds_node.index, node)
            self.db[pl_name] = DEFAULT_DB({ "data": records[rec] })
            # add the specs for this entry
            if "Specification" in self.drop:
                return
            for spec,procedures in spec_map_ids.items():
                pl_name = "QCS-"+str(spec)
                spec_node = Node.Node(name="Specification", payload=pl_name)
                spec_node = self.add(node.index, spec_node)
                self[pl_name] = DEFAULT_DB({"data": ds_specs[spec.lower()].dict()})

                if name in self.drop:
                    return

                # if the spec is not here, then somehow data was never 
                # generated. This is not a good thing.
                if spec not in records[rec].object_map:
                    print("Spec", spec, "for entry", rec, "missing!")
                    continue


                for proc in procedures:
                    if proc != "QCP-" + records[rec].object_map[spec]:
                        continue
                    pl_name = proc
                    pload = DEFAULT_DB({"data": obj_map[proc]})
                    proc_node = Node.Node(name=name, payload=pl_name)
                    proc_node = self.add(spec_node.index, proc_node)
                    self.db[pl_name] = pload

                    # expand these nodes below
                    nodes.append(proc_node)

                    # perhaps inefficient, but add init mols as we travel
                    # the data, rather than all at once
                    # should help with debugging
                    molid = pload["data"]["initial_molecule"]
                    if init_mols_are_lists:
                        for molid_i in molid:
                            molid_i = "QCM-" + molid_i
                            mol_obj = init_mol_map[molid_i]
                            self.db[molid_i] = DEFAULT_DB({"data": mol_obj})
                    else:
                        molid = "QCM-" + molid
                        mol_obj = init_mol_map[molid]
                        self.db[molid] = DEFAULT_DB({"data": mol_obj})

#        nodes = []
#        spec_data = None
#        for index,obj in obj_map.items():
#            entry_match = None
#            for entry in records.values():
#                if suf + str(entry.object_map[spec]) == index:
#                    entry_match = entry
#                    break
#
#            if entry_match is None:
#                raise IndexError("Could not match Entry to Record")
#            pl = { "entry": entry_match, "data": obj}
#            self.db[index] = pl
#            if spec_data is None:
#                spec_data = obj['qc_spec'].dict()
#                self.db['QCS-' + spec] = spec_data
#            procedure = Node.Node(name=name, payload=index)
#            procedure = self.add( spec_node.index, procedure)
#
#            # add to list to expand later below
#            nodes.append(procedure)

#        for node in nodes:
#            qcid = self.db[node.payload]
#            if init_mols_are_lists:
#                molid = 'QCM-' + str(qcid["data"]["initial_molecule"][0])
#            else:
#                molid = 'QCM-' + str(qcid["data"]["initial_molecule"])
#            mol_obj = init_mol_map[molid]
#            self.db[molid] = { "data": mol_obj}

        # descend into the next type of nodes
        fn([node.index for node in nodes], skel=skel)

    def branch_torsiondrive_ds( self, node, skel=False, **kwargs):
        """ Generate the individual torsiondrives """
        self.branch_ds( node, "TorsionDrive", self.branch_torsiondrive_record, skel=skel, **kwargs)

    def branch_optimization_ds( self, node, skel=False, **kwargs):
        """ Generate the individual optimizations """
        self.branch_ds( node, "Optimization", self.branch_optimization_record, skel=skel, **kwargs)

    def branch_gridopt_ds( self, node, skel=False, **kwargs):
        """ Generate the individual optimizations """
        self.branch_ds( node, "GridOpt", self.branch_gridopt_record, skel=skel, **kwargs)

    def branch_gridopt_record( self, nids, skel=False):
        """ Generate the optimizations under Grid Optimization record
        The optimizations are separated by one or more Constraints
        """
        if "GridOpt" in self.drop:
            return
        
        if not hasattr( nids, "__iter__"):
            nids = [nids]
        nodes = [self.node_index.get( nid) for nid in nids]
        opt_ids = [list( self.db.get( node.payload).get("data").get("grid_optimizations").values()) for node in nodes]
        client = self.db["ROOT"]["data"]
        print("Downloading optimization information for",
            len( flatten_list( opt_ids, times=-1)))
        
        #projection = { 'optimization_history': True } if skel else None
        projection = None
        flat_ids = ['QCP-' + str(x) for x in flatten_list( opt_ids, times=-1)]
        opt_map = self.batch_download( flat_ids, client.query_procedures, projection=projection)
        
        # add the constraint nodes
        if "Constraint" in self.drop:
            return
        opt_nodes = []

        for node in nodes:

            obj = self.db[node.payload]
            scans = obj["data"]["keywords"].scans
            if len(scans) == 0:
                print("This grid opt has no scans:", node.payload)
            elif len(scans) > 1:
                print("This grid opt multiple scans!:", node.payload)
            scan = scans[0].__dict__
            for  constraint, opts in obj["data"]["grid_optimizations"].items():
                #TODO need to cross ref the index to the actual constraint val
                
                val = eval(constraint)

                # handle when index is "preoptimization" rather than e.g. [0]
                if isinstance( val, str):
                    continue
                else:
                    step = scan.get( "steps")[val[0]]
                pl = (scan.get( "type")[:], 
                        tuple(scan.get( "indices")), 
                        step)
                constraint_node = Node.Node( payload=pl , name="Constraint")
                constraint_node = self.add( node.index, constraint_node)

                index = 'QCP-' + opts
                opt_node = Node.Node( name="Optimization", payload=index)
                opt_nodes.append(opt_node)
                opt_node = self.add(constraint_node.index, opt_node)
                self.db[index] = DEFAULT_DB({ "data": opt_map[index] })

        self.branch_optimization_record( [x.index for x in opt_nodes], skel=skel)


    def branch_torsiondrive_record( self, nids, skel=False):
        """ 
        Generate the optimizations under a TD or an Opt dataset
        The optimizations are separated by a Constraint
        """
        if "Optimization" in self.drop:
            return
        
        if not hasattr( nids, "__iter__"):
            nids = [nids]

        nodes = [self[nid] for nid in nids]
        opt_ids = [list( self.db[node.payload]["data"]["optimization_history"].values()) for node in nodes]
        client = self.db["ROOT"]["data"]

        if len(opt_ids) == 0:
            print("No optimizations! For", nodes, ":")
            print("No optimizations! IDs", nids, ":")
            for node in nodes:
                print(self.db[node.payload]["data"])

        flat_ids = flatten_list( opt_ids, times=-1)
        if len(flat_ids) == 0:
            print("No optimizations to download?!?")
            return
        print("Downloading optimization information for", len(flat_ids))
        
        projection = None
        flat_ids = ['QCP-' + str(x) for x in flat_ids]
        opt_map = self.batch_download(flat_ids, client.query_procedures, projection=projection)
        
        # add the constraint nodes
        if "Constraint" in self.drop:
            return
        opt_nodes = []

        # have the opt map, which is the optimizations with their ids
        # nodes are the torsiondrives
        for node in nodes:
            obj = self.db[node.payload]
            entry = next(self.node_iter_to_root_single(node, select="Entry"))
            entry = self.db[entry.payload]["data"]
            indices = entry.td_keywords.dihedrals[0]
            for constraint, opts in obj["data"]["optimization_history"].items():
                #val = eval(constraint)

                pl = ( "dihedral", 
                        indices, 
                        eval( constraint)[0])
                constraint_node = Node.Node( payload=pl , name="Constraint")
                constraint_node = self.add(node.index, constraint_node)
                
                for index in opts:
                    index = 'QCP-' + index
                    opt_node = Node.Node( name="Optimization", payload=index)
                    opt_nodes.append( opt_node)
                    opt_node = self.add(constraint_node.index, opt_node)
                    self.db[index] = DEFAULT_DB({"data": opt_map[index]})

        self.branch_optimization_record( [x.index for x in opt_nodes], skel=skel)


    def branch_optimization_record( self, nids, skel=False):
        """ Gets the gradients from the optimizations """

        if nids is None:
            return
        if "Gradient" in self.drop:
            return
        suf = "QCR-"
        if not hasattr( nids, "__iter__"):
            nids = [nids]
        print("Parsing optimation trajectory records from index...")
        nodes = [ self[nid] for nid in nids]
        try:
            if "Intermediates" in self.drop:
                result_ids = []
                for node in nodes:
                    ids = self.db[node.payload]["data"]["trajectory"]
                    if not ids is None:
                        result_ids.append(ids[-1:])
                allids = flatten_list(
                    [self.db[node.payload]["data"]["trajectory"]
                        for node in nodes], times=-1)

                if len(result_ids) == 0:
                    print("Appears nothing has completed... moving on")
                else:
                    print("Dropped intermediates: reduced to ", len(flatten_list(result_ids, times=-1)), "from", len(allids))
            else:
                result_ids = [ self.db[node.payload]["data"]["trajectory"] for node in nodes]
        except AttributeError:
            print( nodes)
            assert False
        client = self.db["ROOT"]["data"]

        flat_result_ids = []
        if len(result_ids) > 0:
            flat_result_ids = list(set(
                [ suf + str(x) for x in flatten_list( result_ids, times=-1)]))
        
        result_nodes = []
        track = []
        if skel:
            # the easy case where we have the gradient indices
            if len( flat_result_ids) > 0:
                print("Collecting gradient stubs for", len( flat_result_ids))
            result_map = None

        else:
            print("Downloading gradient information for", len( flat_result_ids))
            result_map = self.batch_download( flat_result_ids, client.query_results)

        incompletes = 0
        completes = 0
        errors = 0
        for node in nodes:
            obj = self.db[node.payload]
            traj = obj["data"]["trajectory"]
            status = obj["data"]["status"][:]
            if status == "COMPLETE":
                node.state = Node.CLEAN
                completes += 1
            elif status == "INCOMPLETE":
                incompletes += 1
                print("QCA: This optimization incomplete ("+node.payload+")")
                continue
            else:
                errors += 1
                print("QCA: This optimization failed ("+node.payload+")")
                continue
            if traj is not None and len(traj) > 0: 
                if "Intermediates" in self.drop:
                    traj = traj[-1:]
                for index in traj:
                    index = suf + index
                    name = "GradientStub" if skel else "Gradient"
                    result_node = Node.Node(name=name, payload=index)
                    result_node.set_state(Node.CLEAN)
                    result_nodes.append(result_node)
                    result_node = self.add(node.index, result_node)
                    pl = {} if skel else result_map[index]
                    self.db[index] = DEFAULT_DB({"data" : pl })
            else:
                print("No gradient information for", node,": Not complete?")

        data = [completes, len(nodes), completes/len(nodes) * 100.0]
        print("Completes:   {:8d}/{:8d} ({:6.2f}%)".format(*data))

        data = [errors, len(nodes), errors / len(nodes) * 100.0]
        print("Errors:      {:8d}/{:8d} ({:6.2f}%)".format(*data))

        data = [incompletes, len(nodes), incompletes / len(nodes) * 100.0]
        print("Incompletes: {:8d}/{:8d} ({:6.2f}%)".format(*data))

        self.branch_result_record([x.index for x in result_nodes], skel=skel)

    def branch_result_record( self, nids, skel=False):
        """ Gets the molecule from the gradient """

        if not hasattr( nids, "__iter__"):
            nids = [nids]
        if len(nids) == 0:
            print("No gradients!")
            return
        if "Molecule" in self.drop:
            return
        nodes = [self[nid] for nid in nids]
        client = self.db["ROOT"]["data"]

        mol_nodes = []
        gradstubs = [node for node in nodes if ("molecule" not in self.db[node.payload]["data"])]
        fullgrads = [node for node in nodes if ("molecule"     in self.db[node.payload]["data"])]

        mol_map = {}

        suf = "QCM-"

        if len(gradstubs) > 0:
            print( "Downloading molecule information from grad stubs", len(gradstubs))
            projection = { 'id': True, 'molecule': True }
            projection = [ 'id', 'molecule']
            mol_map = self.batch_download([node.payload for node in gradstubs],
                client.query_results, projection=projection)
            for node in gradstubs:
                obj = self.db[node.payload]["data"]
                obj.update(mol_map[node.payload])
        if skel:
            # need to gather gradient stubs to get the molecules
            if len(fullgrads) > 0:
                print("Gathering molecule information from gradients", len( fullgrads))
                mol_map.update(
                        {node.payload: self.db[node.payload]["data"]['molecule']
                            for node in fullgrads})
        else:
            print("Downloading molecule information for", len( nodes))
            mol_map = self.batch_download([suf+self.db[node.payload]["data"]['molecule'] for node in nodes], client.query_molecules)

        for node in nodes:
            if node.payload is None:
                print(node)
                assert False
            else:
                index = suf + self.db[node.payload]["data"]['molecule']
                if index in mol_map:
                    name = "MoleculeStub" if skel else "Molecule"
                    state = "NEW" if skel else "CLEAN"
                    pl = DEFAULT_DB({"data": mol_map[index]})
                else:
                    name = "MoleculeStub"
                    state = "NEW"
                    pl = DEFAULT_DB({"data": DEFAULT_DB({'id' : index})})

                mol_node = Node.Node( name=name, payload=index)
                mol_node.state = state
                mol_node = self.add(node.index, mol_node)
                mol_nodes.append(mol_node)

                if skel and (index not in self.db):
                    self.db[index] = pl
            assert len(node.children) > 0

        # hessian stuff
        if "Hessian" in self.drop:
            return

        ids = [x.payload for x in mol_nodes]
        name = "Hessian"
        projection = None

        if skel:
            projection = { 'id': True , 'molecule': True }
            projection = [ 'id', 'molecule' ]
            name = "HessianStub"

        print( "Downloading", name, "for", len(mol_nodes))
        hess_objs = self.batch_download_hessian( ids, client.query_results, projection=projection)

        if len(hess_objs) > 0:
            for mol in mol_nodes:
                payload = mol.payload
                for hess in hess_objs:
                    pl = hess_objs[hess]
                    if payload == ("QCM-" + pl["molecule"]):
                        hess_node = Node.Node( name="Hessian", payload=hess)
                        hess_node.state = Node.CLEAN
                        hess_node = self.add(mol.index, hess_node)
                        self.db[hess] = DEFAULT_DB({ "data" : pl })

    def torsiondriverecord_minimum(self, tdr_nodes):

        if not hasattr( tdr_nodes, "__iter__"):
            tdr_nodes = list(tdr_nodes)
        ret = {}
        for tdr_node in tdr_nodes:
            tdr = self.db[tdr_node.payload]["data"]
            minimum_positions = tdr["minimum_positions"]
            opt_hist = tdr["optimization_history"]
            min_nodes = []
            for cnid in tdr_node.children:
                constraint_node = self[cnid]
                point = constraint_node.payload
                min_opt_id = minimum_positions[point]
                min_ene_id = "QCR-" + str(opt_hist[point][min_opt_id])
                for opt_nid in constraint_node.children:
                    opt_node = self[opt_nid]
                    if min_ene_id == opt_node.index:
                        min_nodes.append( opt_node)
                        break
            ret.update({tdr_node.index: min_nodes})
        return ret

    def node_iter_optimization_minimum( self, nodes, select=None, fn=Tree.Tree.node_iter_depth_first):
        if not hasattr( nodes, "__iter__"):
            nodes = [nodes]
        for top_node in nodes:
            for opt_node in fn( self, top_node, select="Optimization"):
                if len( opt_node.children) > 0:
                    # bit ugly, but it is taking the last gradient record, and 
                    # the first (and only) molecule from it
                    n = self[self[opt_node.children[-1]].children[0]]
                    if select:
                        if select == n.name:
                            yield n
                    else:
                        yield n

    def node_iter_contraints( self, nodes, fn=Tree.Tree.node_iter_depth_first):
        yield from fn(self, nodes, select="Constraint")


    def find_mol_attr_in_tree( self, key, value, top_node, select=None):
        assert False
        hits = []
        for node in self.node_iter_depth_first(top_node):
            if "entry" not in node.payload:
                continue
            tgt = node.payload["entry"].dict()["attributes"][key]
            if value == tgt: 
                hits.append( node)
        return hits

    def node_iter_torsiondriverecord_minimum(self, tdr_nodes=None, select=None,
            sort=True, fn=Tree.Tree.node_iter_depth_first):
        if tdr_nodes is None:
            tdr_nodes = [n for n in self.node_iter_breadth_first(
                self.root(), select="TorsionDrive")]
        if not hasattr(tdr_nodes, "__iter__"):
            tdr_nodes = [tdr_nodes]

        tdr_nodes = flatten_list(
            [self.node_iter_depth_first(x, select="TorsionDrive")
                for x in tdr_nodes])

        for tdr_node in tdr_nodes:

            tdr = self.db[tdr_node.payload]["data"]
            minimum_positions = tdr['minimum_positions']
            opt_hist = tdr['optimization_history']

            cids = tdr_node.children
            if sort:
                cids = sorted(tdr_node.children,
                    key=lambda i: self[i].payload[2])

            for constraint_nid in cids:
                constraint_node = self[constraint_nid]
                point = '[' + str(constraint_node.payload[2]) + "]"
                if point not in minimum_positions:
                    continue
                min_opt_id = minimum_positions[point]
                min_ene_id = "QCP-" + opt_hist[point][min_opt_id]
                for opt_nid in constraint_node.children:
                    opt_node = self[opt_nid]
                    if min_ene_id == opt_node.payload:
                        yield from fn( self, self[opt_node.children[-1]], select)
                        break
            



