    
from datetime import datetime
import numpy as np
from ..tools.util import flatten_list

import qcfractal.interface as ptl
import treedi.tree as Tree
import treedi.node as Node

def match_canonical_isomeric_explicit_hydrogen_smiles( entryA, entryB):
    key = 'canonical_isomeric_explicit_hydrogen_smiles'
    #if entryA is None or entryB is None:
    #    return False
    A = entryA.attributes[ key]
    B = entryB.attributes[ key]
    if A == B:
        return True

    return False


class QCATree( Tree.Tree):

    def __init__( self, name, root_payload=None, node_index=None, db=None, \
                  payload=None):
        print("Building QCATree")
        super().__init__( name, root_payload=root_payload, \
                          node_index=node_index, db=db, payload=payload)

    def vtable( self, obj):
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

        if isinstance( obj, TorsionDriveDataset):
            return self.branch_torsiondrive_ds
        elif isinstance( obj, TorsionDriveRecord): 
            return self.branch_torsiondrive_record
        elif isinstance( obj, OptimizationDataset):
            return self.branch_optimization_ds
        elif isinstance( obj, OptimizationRecord):
            return self.branch_optimization_record
        elif isinstance( obj, GridOptimizationDataset):
            return self.branch_gridopt_ds
        elif isinstance( obj, OptimizationRecord):
            return self.branch_gridopt_record

        raise ValueError("QCA type '" + str(type(obj)) + "' not understood")
        return None

    def to_pickle_str( self):
        import pickle
        self.isolate()
        return pickle.dumps( self)

    

    def combine_by_entry( self, fn, targets=None):
        """
        compare entries using fn, and collect into a parent node
        fn is something that compares 2 entries 
        returns a node where children match key
        """
        new_nodes = []
        
        if targets is None:
            entries = list(self.iter_entry())
        elif hasattr( targets, "__iter__"):
            entries = list(targets)
        else:
            entries = [targets]
        if len( entries) == 0:
            return new_nodes

        used = set()
        for i in range( len( entries)):
            if i in used:
                continue
            ref = entries[i]
            ref_obj = self.db[ ref.payload]['entry']
            used.add( i)

            print("Adding", ref, "to nodes")
            node = Node.Node( name="Folder", payload=repr( fn))
            node.add( ref)

            for j in range( i+1, len( entries)):
                entry = entries[j]
                entry_obj = self.db[ entry.payload]['entry']
                if fn( ref_obj, entry_obj):
                    print("MATCH!", ref, entry)
                    node.add( entry)
                    used.add( j)
                #else:
                #    print("NOT A MATCH!", ref, entry)
                #    node = Node.Node( name="Folder", payload=repr( fn))
                #    print("Adding", entry, "to nodes")
                #    node.add( entry)
                #    new_nodes.append( node)
                #    ref_obj = entry_obj
                #    ref = entry
            new_nodes.append( node)

        return new_nodes






#    def to_pickle(self, db=True, name=None):
#        import pickle
#        if name is None:
#            name = self.name + ".p"
#        #self.isolate()
#        if not db:
#            tmp = self.db
#            self.db = None
#        with open( name, 'wb') as fid:
#            pickle.dump( self, fid)
#        if not db:
#            self.db = tmp

    def isolate( self):
        for link in self.link:
            self.link.__setitem( link, self.link.ID)

    def associate(self, link_trees):
        for link in link_trees:
            self.link[ link.ID] = link

    def _obj_is_qca_collection( self, obj):
        return True

    def build_index( self, ds, drop=None):
        """
        Take a QCA DS, and create a node for it.
        Then expand it out
        """
        assert self._obj_is_qca_collection( ds)
        
        # the object going into the data db
        pl = { 'data': ds }
        self.db.__setitem__( str(ds.data.id), pl)

        # create the index node for the tree and integrate it
        ds_node = Node.Node( name=ds.data.name, payload=str(ds.data.id) )
        self.add( self.root_index, ds_node)

        
        self.expand_qca_dataset_as_tree( ds_node.index, skel=True, drop=drop)

    def expand_qca_dataset_as_tree(self, nid, skel=False, drop=None):
        if drop is not None:
            self.drop = drop
        else:
            self.drop = []

        node = self.node_index.get( nid)
        payload = self
        oid = self.node_index.get( nid).payload
        obj = self.db.get( oid).get( "data")

        fn = self.vtable( obj)
        fn( nid, skel=skel)

        self.drop = []

    def node_iter_entry( self, nodes, select=None, fn=Tree.Tree.node_iter_depth_first):
        if not hasattr( nodes, "__iter__"):
            nodes = [nodes]
        for top_node in nodes:
            for node in fn( self, top_node):
                obj = self.db.get( node.payload)
                if "entry" in obj:
                    yield node

    def iter_entry( self, select=None):
        yield from self.node_iter_entry( \
            self.node_index.get( self.root_index), select=select)


    def cache_torsiondriverecord_minimum_molecules(self, tdr_nodes=None):
        if tdr_nodes is None:
            tdr_nodes = self.iter_entry( select="TorsionDrive")
        fn = __class__.node_iter_torsiondriverecord_minimum
        self.cache_minimum_molecules( tdr_nodes, fn)
        return

    def cache_initial_molecules( self):
        
        mols = dict()
        for entry in self.iter_entry():
            eobj = self.db.get( entry.payload)
            init_mol_id = eobj.get( "data").get( "initial_molecule")
            #print( "RAW", init_mol_id, type(init_mol_id))
            #print( "CHECK", init_mol_id[0], type(init_mol_id[0]))
            if isinstance( init_mol_id, list):
                #print( "TRUE", init_mol_id)
                init_mol_id = init_mol_id[0]
            init_mol_id = "QCM-" + str(init_mol_id)
            mols[ entry.index] = init_mol_id

        if len(mols) == 0:
            return
        mol_ids_flat = flatten_list([x for x in mols.values()], -1)
        client = self.db.get( "ROOT").get( "data")
        fresh_obj_map = self.batch_download(mol_ids_flat, client.query_molecules)

        for key, val in fresh_obj_map.items():
            if key not in self.db:
                self.db.__setitem__( obj, { "data": val })
            else:
                self.db.get( key).__setitem__( "data", val)

        #for entry in self.iter_entry():
        #    eobj = self.db.get( entry.payload)
        #    init_mol_id = eobj.get( "data").get( "initial_molecule")
        #    #print( "RAW", init_mol_id, type(init_mol_id))
        #    #print( "CHECK", init_mol_id[0], type(init_mol_id[0]))
        #    if isinstance( init_mol_id, list):
        #        #print( "TRUE", init_mol_id)
        #        init_mol_id = init_mol_id[0]
        #    init_mol_id = "QCM-" + str(init_mol_id)


    def cache_minimum_molecules( self, nodes, fn):

        if not hasattr( nodes, "__iter__"):
            nodes = [nodes]

        # make a list since we iterate it twice. if its a generator the first 
        # iteration would consume the generator
        nodes = list(nodes)
        mols = {}
        for top_node in nodes:
            mols[ top_node.index] = []
            for node in fn( self, top_node, select="MoleculeStub"):
                mols[ top_node.index].append( node.payload)
            if mols[ top_node.index] == []:
                mols.pop( top_node.index)
        if len(mols) == 0:
            return
        mol_ids_flat = flatten_list([x for x in mols.values()], -1)
        client = self.db.get( "ROOT").get( "data")
        fresh_obj_map = self.batch_download(mol_ids_flat, client.query_molecules)

        for top_node in nodes:

            found=0
            candidates = len( list( fn( self, top_node, select="MoleculeStub")))

            for node in fn(self, top_node, select="MoleculeStub"):
                #mol = None
                #for mol_candidate in fresh_obj_map:
                #    print("MIN NODE", node.index, "CANDIDATE", mol_candidate)
                #    if node.index == mol_candidate:
                #        found += 1
                #        mol = mol_candidate
                #        print( "MATCHED!", node.index, mol_candidate)
                #        break
                #if mol is None:
                #    print( "COULD NOT PLACE A DOWNLOAD??", node.index)
                #    assert False
            
                mol = fresh_obj_map[ node.payload]
                assert mol is not None

                self.db.__setitem__( node.payload, { "data": mol })
                node.name = "Molecule"
                node.stamp = datetime.now()
                self.register_modified( node, state=Node.CLEAN)
            #for tree in self.link.values():
            #    tree.register_modified( Tree.node_index.get( node.index), state=DIRTY)

    def cache_hessians( self, nodes):
        self.cache_results( nodes, select="HessianStub")

    def cache_results( self, nodes, fn=Tree.Tree.node_iter_depth_first, select=None):
        
        if not hasattr( nodes, "__iter__"):
            nodes = [nodes]
        results = {}
        for top_node in nodes:
            results[ top_node.index] = []
            for node in fn( self, top_node, select=select):
                results[ top_node.index].append( node.index)
            if results[ top_node.index] == []:
                results.pop( top_node.index)
        if len(results) == 0:
            return
        ids_flat = flatten_list([x for x in results.values()], -1)
        client = get_root( nodes[0]).payload
        print("Caching results using iterator", str(fn), "on", len(ids_flat))
        fresh_obj_map = self.batch_download( ids_flat, client.query_results)

        for top_node in nodes:
            for node in fn( self, top_node, select=select):
                node.payload = fresh_obj_map[ node.index]
                if "Stub" in node.name:
                    node.name = node.name[:-4]
                node.stamp = datetime.now()

    def cache_optimization_minimum_molecules( self, nodes=None):
        if nodes is None:
            nodes = self.node_index.get( self.root_index)
        self.cache_minimum_molecules( nodes, __class__.node_iter_optimization_minimum)
        return

    def batch_download_hessian( self, full_ids, fn, max_query=1000, projection=None):
        import math
        chunks = []
        L = len(full_ids)
        if max_query > 1000:
            max_query = 1000
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
        if len(full_ids[0].split("-")) > 1:
            ids = [x.split("-")[-1] for x in full_ids]
            id_suf = full_ids[0].split("-")[0] + "-"
        else:
            ids = full_ids
            id_suf = ""
        for i,j in chunks:
            #if i == 0:
            if True:
                print( "{:20s} {:4d} {:4d} ".format(out_str, i, j), end="")
            elapsed = datetime.now()
            objs += [dict(obj) for obj in \
                fn( molecule=ids[i:j], driver="hessian", \
                    limit=max_query, include=projection)]

            elapsed = str(datetime.now() - elapsed)
            #if i == 0:
            if True:
                print( "... Received {:6d}  | elapsed: {:s}\n".format( \
                    len(objs), elapsed), end="")
        print("TotalTime: {:s}\n".format( str( datetime.now() - total)), end="")
        obj_map = {id_suf + str(obj.get('id')): obj for obj in objs}
        return obj_map


    def batch_download( self, full_ids, fn, max_query=1000, \
                        procedure=None, projection=None):
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
                print( "{:20s} {:4d} {:4d} ".format(out_str, i, j), end="")
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
                print( "... Received {:6d}  | elapsed: {:s}\n".format( \
                    len(objs), elapsed), end="")
        print("TotalTime: {:s}\n".format( str( datetime.now() - total)), end="")
        obj_map = {id_suf + str(obj.get('id')): obj for obj in objs}
        return obj_map

    def branch_torsiondrive_ds( self, nid, skel=False):
        """ Generate the individual torsion drives """
        if "TorsionDrive" in self.drop:
            return
        suf = "QCP-"
        node = self.node_index.get( nid)
        ds = self.db.get( node.payload).get( "data")
        records = ds.data.records

        #[entry.object_map.get("default") for entry in records.values()]
        td_ids = [suf + str(entry.object_map.get("default")) for entry in records.values()]
        #td_ids = td_ids[:1]
        client = self.db.get( "ROOT").get( "data") 
        print("Downloading TorsionDrive information for", len( flatten_list( td_ids, times=-1)))
        td_map = self.batch_download( td_ids, client.query_procedures) 


        td_nodes = []
        for index, obj in td_map.items():
            entry_match = None
            for entry in records.values():
                if suf + str(entry.object_map.get("default")) == index:
                    entry_match = entry
            if entry_match is None:
                raise IndexError("Could not match TDEntry to a TDRecord")

            #td_nodes.append( Node(index="".join([ suf, index]), name="TorsionDrive", payload={"meta": entry_match, "record": obj}))
            pl = { "entry": entry_match, "data": obj}
            self.db.__setitem__( index, pl)
            td_nodes.append( \
                Node.Node( name="TorsionDrive", payload=index))
        [ self.add( node.index, v) for v in td_nodes]

        init_mol_ids = ['QCM-' + str(td.get( "initial_molecule")[0]) for \
            td in td_map.values()]
        print("Downloading TorsionDrive initial molecules for  for", \
            len( init_mol_ids))
        init_mol_map = self.batch_download( init_mol_ids, client.query_molecules)
            
        #[ td_node.payload.__setitem__("initial_molecule", \
        #    init_mol_map.get( 'QCM-' + str(td_node.payload.get( "record").get( "initial_molecule")[0])))\
        #    for td_node in td_nodes] 

        for td_node in td_nodes:
            qcid = self.db.get( td_node.payload)
            molid = 'QCM-' + str(qcid.get( "data").get( "initial_molecule")[0])
            mol_obj = init_mol_map.get( molid)
            self.db.__setitem__( molid , { "data": mol_obj})
            #print( "ADDED INIT MOL", molid, self.db.get( molid) )

        #print( "*********************************")
        #[print( x ) for x in self.db.values()]
        #print( "*********************************")
        #print( self.node_index)
        #print( "*********************************")
        self.branch_torsiondrive_record( [ n.index for n in td_nodes], skel=skel)

    def branch_ds( self, nid, name, fn, skel=False, start=0, limit=0):
        """ Generate the individual entries from a dataset """
        if name in self.drop:
            return
        suf = "QCP-"
        node = self.node_index.get( nid)
        ds = self.db.get( node.payload).get( "data")
        records = ds.data.records
        if limit > 0 and limit < len(records):
            records = dict(list(records.items())[start:start+limit])


        #[entry.object_map.get("default") for entry in records.values()]
        ids = [suf + str(entry.object_map.get("default")) for entry in records.values()]
        #ids = ids[:1]
        client = self.db.get( "ROOT").get( "data") 
        print("Downloading", name, "information for", len( flatten_list( ids, times=-1)))
        obj_map = self.batch_download( ids, client.query_procedures) 

        nodes = []
        for index,obj in obj_map.items():
            entry_match = None
            for entry in records.values():
                if suf + str(entry.object_map.get("default")) == index:
                    entry_match = entry
                    break
            if entry_match is None:
                raise IndexError("Could not match Entry to Record")
            pl = { "entry": entry_match, "data": obj}
            self.db.__setitem__( index, pl)
            nodes.append( Node.Node( name=name, payload=index))
        [ self.add( node.index, v) for v in nodes]

        #print( list(obj_map.values())[0] )
        init_mol_ids = [obj.get( "initial_molecule") for obj in obj_map.values()]
        init_mols_are_lists = False
        if isinstance( init_mol_ids[0], list):
            init_mols_are_lists = True
            init_mol_ids = [ str(x)[0] for x in init_mol_ids]
        init_mol_ids = ["QCM-" + x for x in init_mol_ids]

        #print( init_mol_ids)
        print("Downloading", name, "initial molecules for  for", len( init_mol_ids))
        init_mol_map = self.batch_download( init_mol_ids, client.query_molecules)
            
        #[ td_node.payload.__setitem__("initial_molecule", \
        #    init_mol_map.get( 'QCM-' + str(td_node.payload.get( "record").get( "initial_molecule")[0])))\
        #    for td_node in td_nodes] 

        for node in nodes:
            qcid = self.db.get( node.payload)
            if init_mols_are_lists:
                molid = 'QCM-' + str(qcid.get( "data").get( "initial_molecule")[0])
            else:
                molid = 'QCM-' + str(qcid.get( "data").get( "initial_molecule"))
            mol_obj = init_mol_map.get( molid)
            self.db.__setitem__( molid , { "data": mol_obj})

        fn( [node.index for node in nodes], skel=skel)

    def branch_optimization_ds( self, node, skel=False):
        """ Generate the individual optimizations """
        self.branch_ds( node, "Optimization", self.branch_optimization_record, skel=skel)

    def branch_gridopt_ds( self, node, skel=False):
        """ Generate the individual optimizations """
        self.branch_ds( node, "GridOpt", self.branch_gridopt_record, skel=skel)

    def branch_gridopt_record( self, nids, skel=False):
        """ Generate the optimizations under Grid Optimization record
        The optimizations are separated by one or more Constraints
        """
        if "GridOpt" in self.drop:
            return
        
        if not hasattr( nids, "__iter__"):
            nids = [nids]
        nodes = [self.node_index.get( nid) for nid in nids]
        #print( nodes)
        #print( self.db)
        #assert False
        opt_ids = [list( self.db.get( node.payload).get( "data").get( "grid_optimizations").values()) for node in nodes]
        client = self.db.get( "ROOT").get( "data")
        #client = get_root( nodes[0]).payload
        print("Downloading optimization information for", len( flatten_list( opt_ids, times=-1)))
        
        #projection = { 'optimization_history': True } if skel else None
        projection = None
        flat_ids = ['QCP-' + str(x) for x in flatten_list( opt_ids, times=-1)]
        opt_map = self.batch_download( flat_ids, client.query_procedures, projection=projection)
        
        # add the constraint nodes
        if "Constraint" in self.drop:
            return
        opt_nodes = []

        #breakpoint()
        for node in nodes:

            obj = self.db.get( node.payload)
            #status = obj.get( "data").get( "status")[:]
            scans = obj.get( "data").get( "keywords").__dict__.get( "scans")
            assert len(scans) == 1
            scan = scans[0].__dict__
            for  constraint, opts in obj.get( "data").get( "grid_optimizations").items():
                #TODO need to cross ref the index to the actual constraint val
                
                #cidx = 'CSR-' + node.payload.split("-")[1]
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
                self.add( node.index, constraint_node)
                #self.db.__setitem__( cidx, { "data": scan })
                
                
                #for index in opts:
                #    index = 'QCP-' + index
                #    opt_node = Node.Node( name="Optimization", payload=index)
                #    opt_nodes.append( opt_node)
                #    self.add( constraint_node.index, opt_node)
                #    self.db.__setitem__( index, { "data": opt_map.get( index) })

                index = 'QCP-' + opts
                opt_node = Node.Node( name="Optimization", payload=index)
                opt_nodes.append( opt_node)
                self.add( constraint_node.index, opt_node)
                self.db.__setitem__( index, { "data": opt_map.get( index) })
        #for i,n in enumerate(opt_nodes[:-1]):
        #    idx = n.index
        #    for j,m in enumerate(opt_nodes[i+1:],i+1):
        #        assert idx != m.index
        self.branch_optimization_record( [x.index for x in opt_nodes], skel=skel)

        #breakpoint()
        #for topnode in nodes:
        #    allsuccess = True
        #    for opt_nidx in topnode.children:
        #        opt_node = self.node_index.get( opt_nidx)
        #        obj = self.db.get( opt_node.payload)
        #        status = obj.get( "data").get( "status")[:]
        #        if status == "COMPLETE":
        #            opt_node.state = Node.CLEAN
        #            for node in node_iter_depth_first( opt_node):
        #                if "Stub" in node.name:
        #                    node.state = Node.NEW
        #                if "Grad" in node.name or "Mol" in node.name:
        #                    node.state = Node.CLEAN
        #        else:
        #            opt_node.state = Node.DIRTY
        #            tdnode.state = Node.DIRTY
        
        #for group in opt_ids:
         #   for opt in group:


    def branch_torsiondrive_record( self, nids, skel=False):
        """ Generate the optimizations under a TD or an Opt dataset
        The optimizations are separated by a Constraint
        """
        if "Optimization" in self.drop:
            return
        
        if not hasattr( nids, "__iter__"):
            nids = [nids]
        nodes = [self.node_index.get( nid) for nid in nids]
        #print( nodes)
        #print( self.db)
        #assert False
        opt_ids = [list( self.db.get( node.payload).get( "data").get( "optimization_history").values()) for node in nodes]
        client = self.db.get( "ROOT").get( "data")
        #client = get_root( nodes[0]).payload
        print("Downloading optimization information for", len( flatten_list( opt_ids, times=-1)))
        
        #projection = { 'optimization_history': True } if skel else None
        projection = None
        flat_ids = ['QCP-' + str(x) for x in flatten_list( opt_ids, times=-1)]
        opt_map = self.batch_download( flat_ids, client.query_procedures, projection=projection)
        
        # add the constraint nodes
        if "Constraint" in self.drop:
            return
        opt_nodes = []

        # have the opt map, which is the optimizations with their ids
        # nodes are the torsiondrives
        for node in nodes:
            obj = self.db.get( node.payload)
            for constraint, opts in obj.get( "data").get( "optimization_history").items():
                constraint_node = Node.Node( payload=constraint , name="Constraint")
                self.add( node.index, constraint_node)
                
                for index in opts:
                    index = 'QCP-' + index
                    opt_node = Node.Node( name="Optimization", payload=index)
                    opt_nodes.append( opt_node)
                    self.add( constraint_node.index, opt_node)
                    self.db.__setitem__( index, { "data": opt_map.get( index) })
        #for i,n in enumerate(opt_nodes[:-1]):
        #    idx = n.index
        #    for j,m in enumerate(opt_nodes[i+1:],i+1):
        #        assert idx != m.index
        self.branch_optimization_record( [x.index for x in opt_nodes], skel=skel)
        
        #check to make sure they all were successful
        #breakpoint()
        #for tdnode in nodes:
        #    allsuccess = True
        #    for opt_node in opt_nodes:
        #        obj = self.db.get( opt_node.payload)
        #        status = obj.get( "data").get( "status")[:]
        #        if status == "COMPLETE":
        #            opt_node.state = Node.CLEAN
        #            for node in node_iter_depth_first( opt_node):
        #                if "Stub" in node.name:
        #                    node.state = Node.NEW
        #                if "Grad" in node.name or "Mol" in node.name:
        #                    node.state = Node.CLEAN
        #        else:
        #            opt_node.state = Node.DIRTY
        #            tdnode.state = Node.DIRTY



    def branch_optimization_record( self, nids, skel=False):
        """ Gets the gradients from the optimizations """
        if nids is None:
            return
        if "Gradient" in self.drop:
            return
        suf = "QCR-"
        if not hasattr( nids, "__iter__"):
            nids = [nids]
        nodes = [ self.node_index.get( nid) for nid in nids]
        try:
            result_ids = [ self.db.get( node.payload).get( "data").get( "trajectory") for node in nodes]
        except AttributeError:
            print( nodes)
            assert False
        client = self.db.get( "ROOT").get( "data")
        flat_result_ids = list( set([ suf + str(x) for x in flatten_list( result_ids, times=-1)]))
        
        result_nodes = []
        track = []
        if skel:
            # the easy case where we have the gradient indexs
            print("Collecting gradient stubs for", len( flat_result_ids))
            result_map = None

        else:
            print("Downloading gradient information for", len( flat_result_ids))
            result_map = self.batch_download( flat_result_ids, client.query_results)

        #breakpoint()
        for node in nodes:
            #node = self.node_index.get( node.index)
            obj = self.db.get( node.payload)
            traj = obj.get( "data").get( "trajectory")
            status = obj.get( "data").get( "status")[:]
            if status == "COMPLETE":
                node.state = Node.CLEAN
            else:
                print("QCA: This optimization failed ("+node.payload+")")
                continue
            if traj is not None and len(traj) > 0: 
                for index in traj:
                    index = suf + index
                    name = "GradientStub" if skel else "Gradient"
                    result_node = Node.Node( name="GradientStub", payload=index)
                    resutl_node = Node.CLEAN
                    result_nodes.append( result_node)
                    self.add( node.index, result_node)
                    pl = {} if skel else result_map.get( index)
                    self.db.__setitem__( index, { "data" : pl } )
            else:
                print("No gradient information for", node,": Not complete?")
                        
        self.branch_result_record( [x.index for x in result_nodes], skel=skel)

    def branch_result_record( self, nids, skel=False):
        """ Gets the molecule from the gradient """
        
        if not hasattr( nids, "__iter__"):
            nids = [nids]
        if len(nids) == 0:
            assert False
            return
        if "Molecule" in self.drop:
            assert False
            return
        #print( nodes)
        nodes = [ self.node_index.get( nid) for nid in nids]
        client = self.db.get( "ROOT").get( "data")
        
        mol_nodes = []
        gradstubs = [node for node in nodes if ("molecule" not in self.db.get( node.payload).get( "data"))]
        fullgrads = [node for node in nodes if ("molecule"     in self.db.get( node.payload).get( "data"))]

        mol_map = {}

        suf = "QCM-"

        if len(gradstubs) > 0:
            print( "Downloading molecule information from grad stubs", len( gradstubs))
            projection = { 'id': True, 'molecule': True }
            projection = [ 'id', 'molecule']
            mol_map = self.batch_download( [node.payload for node in gradstubs], client.query_results, projection=projection )
            for node in gradstubs:
                obj = self.db.get( node.payload).get( "data")
                obj.update( mol_map.get( node.payload))
        if skel:
            # need to gather gradient stubs to get the molecules
            if len(fullgrads) > 0:
                print("Gathering molecule information from gradients", len( fullgrads))
                mol_map.update( {node.payload: self.db.get( node.payload).get( "data").get( 'molecule') for node in fullgrads})
        else:
            print("Downloading molecule information for", len( nodes))
            mol_map = self.batch_download( [self.db.get( node.payload).get( "data").get( 'molecule') for node in nodes], client.query_molecules)

        for node in nodes:
            if node.payload is None:
                print(node)
                assert False
            else:
                name = "MoleculeStub" if skel else "Molecule"
                state = "NEW" if skel else "CLEAN"
                index = suf + self.db.get( node.payload).get( "data").get( 'molecule')
                mol_node = Node.Node( name=name, payload=index)
                mol_node.state = state
                self.add( node.index, mol_node)
                mol_nodes.append( mol_node)

                if skel and index not in self.db:
                    self.db.__setitem__( index, { "data": mol_map.get( index) })
            assert len(node.children) > 0
        
            
        #print( self.db)
        #print( self.node_index )
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
                    if payload == ("QCM-" + hess_objs.get( hess).get( "molecule")):
                        hess_node = Node.Node( name="Hessian", payload=hess)
                        hess_node.state = Node.CLEAN
                        self.add( mol.index, hess_node)
                        pl = hess_objs.get( hess)
                        self.db.__setitem__( hess, { "data" : pl } )

    def torsiondriverecord_minimum(self, tdr_nodes):
        if not hasattr( tdr_nodes, "__iter__"):
            tdr_nodes = list(tdr_nodes)
        ret = {}
        for tdr_node in tdr_nodes:
            
            tdr = self.db.get( tdr_node.payload).get( "data")  
            minimum_positions = tdr.get( "minimum_positions")
            opt_hist = tdr.get( "optimization_history")
            min_nodes = []
            for cnid in tdr_node.children:
                constraint_node = self.node_index.get( cnid)
                point = constraint_node.payload
                min_opt_id = minimum_positions.get( point)
                min_ene_id = "QCR-" + str(opt_hist.get( point)[ min_opt_id])
                for opt_nid in constraint_node.children:
                    opt_node = self.node_index.get( opt_nid)
                    if min_ene_id == opt_node.index:
                        min_nodes.append( opt_node)
                        break
            ret.update( {tdr_node.index: min_nodes})
        return ret

    def node_iter_optimization_minimum( self, nodes, select=None, fn=Tree.Tree.node_iter_depth_first):
        if not hasattr( nodes, "__iter__"):
            nodes = [nodes]
        for top_node in nodes:
            for opt_node in fn( self, top_node, select="Optimization"):
                if len( opt_node.children) > 0:
                    yield self.node_index.get( self.node_index.get( opt_node.children[-1]).children[0])
                #yield from fn( opt_node.children[-1], select=select)

    def node_iter_entry( self, nodes, select=None, fn=Tree.Tree.node_iter_depth_first):
        if not hasattr( nodes, "__iter__"):
            nodes = [nodes]
        for top_node in nodes:
            for node in fn( self, top_node):
                obj = self.db.get( node.payload)
                if obj and "entry" in obj:
                    yield node

    def find_mol_attr_in_tree( self, key, value, top_node, select=None):
        assert False
        hits = []
        for node in self.node_iter_depth_first( top_node):
            if "entry" not in node.payload:
                continue
            tgt = node.payload.get( "entry").dict().get( "attributes").get( key) 
            if value == tgt: 
                hits.append( node)
        return nodes

    def node_iter_torsiondriverecord_minimum( self, tdr_nodes, select=None, \
                                              fn=Tree.Tree.node_iter_depth_first):
        if not hasattr( tdr_nodes, "__iter__"):
            tdr_nodes = [tdr_nodes]
        ret = {}
        for tdr_node in tdr_nodes:
            tdr = self.db.get( tdr_node.payload).get( "data")
            minimum_positions = tdr.get( "minimum_positions")
            opt_hist = tdr.get( "optimization_history")
            min_nodes = []
            for constraint_nid in tdr_node.children:
                constraint_node = self.node_index.get( constraint_nid)
                point = constraint_node.payload
                min_opt_id = minimum_positions.get( point)
                min_ene_id = "QCP-" + opt_hist.get( point)[ min_opt_id]
                #print( "grad index with min energy (min_ene_id)=", min_ene_id)
                for opt_nid in constraint_node.children:
                    opt_node = self.node_index.get( opt_nid)
                    #print( "comparing to opt_node (constraint child)=", opt_node.index, min_ene_id, opt_node.payload)
                    if min_ene_id == opt_node.payload:
                        yield from fn( self, self.node_index.get( opt_node.children[-1]), select)
                        break
            
def filter_entry(obj):
    if obj.payload == {} or obj.payload is None: 
        return False
    return "entry" in obj.payload



