#!/usr/bin/env python3
import os
import pdb
import treedi
import treedi.tree
import simtk.unit
import simtk.unit as unit
from simtk import openmm
from simtk.openmm.app.simulation import Simulation as OpenMMSimulation
import openforcefield as oFF
from openforcefield.typing.engines.smirnoff.parameters import \
    UnassignedProperTorsionParameterException
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
import smirnoff99frosst as ff
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import FragmentMatcher
from rdkit import Geometry as RDGeom
from ..tools import const
from .. import rdutil
from .. import qcarchive as qca
import offsb.op.geometry
import copy
import numpy as np
from multiprocessing import Pool
import threading
import sys
import tempfile

def load_geometric_opt_trj_xyz( fnm, dims=3):
    N = 0
    xyz = []
    with open( fnm, 'r') as fd:
        N = int( fd.readline().split()[0])
        total_N = 1
        for line in fd:
            total_N += 1
        n_frames = total_N // (N+2)
    ene_in_au = []
    sym = list([""] * N)
    with open( fnm, 'r') as fd:
        xyz = np.empty((n_frames,N,dims))
        for j in range(n_frames):
            N = int(fd.readline().split()[0])
            energy = float(fd.readline().split()[-1])
            ene_in_au.append(energy)
            for i in range(N):
                line = fd.readline().split()
                pos  = line[1:]
                sym[i] = line[0]
                xyz[j][i][:] = list( map( float, pos))

    # xyz in angstrom
    return sym, xyz, np.array(ene_in_au)

class OpenMMEnergy( treedi.tree.PartitionTree):
    """ Creates an openMM system of each entry
        Stores the total energy of molecules
    """

    def __init__( self, filename, source_tree, name):
        super().__init__( source_tree, name)
        import logging
        logger = logging.getLogger()
        level = logger.getEffectiveLevel()
        logger.setLevel(level=logging.ERROR)
        from pkg_resources import iter_entry_points
        self.filename = filename

        self.minimize = False
        self.constrain = False
        search_pth = list(iter_entry_points(group='openforcefield.smirnoff_forcefield_directory'))
        abspth = os.path.join(".", filename)
        print("Searching", ".")
        found = False
        if os.path.exists( abspth):
            self.abs_path = abspth
            print("Found")
            found = True
        if not found:
            for entry_point in search_pth:
                pth = entry_point.load()()[0]
                abspth = os.path.join(pth, filename)
                print("Searching", abspth)
                if os.path.exists( abspth):
                    self.abs_path = abspth
                    print("Found")
                    break
                raise Exception("Forcefield could not be found")
        self.forcefield= oFF.typing.engines.smirnoff.ForceField( self.abs_path, disable_version_check=True, allow_cosmetic_attributes=True)
        logger.setLevel(level=level)
        print("My db is", self.db)

    def to_pickle( self, db=True, name=None):
        #tmp = self.forcefield
        super().to_pickle( db=db, name=name)
        #self.forcefield = tmp

    def isolate( self):
        super().isolate()

    def associate( self, source):
        super().associate( source)

    def _apply_initialize(self, targets):
        pass

    def _apply_finalize(self, targets):
        pass

    def _unpack_result(self, ret):
        self.db.update(ret) 

    def apply_single( self, i, target):

        def unmap( xyz, map_idx):
            inv = [ ( map_idx[ i] - 1) for i in range(len(xyz))]
            return xyz[ inv]

        def remap( xyz, map_idx):
            remap_idx = {v-1:k for k,v in map_idx.items()}
            inv = [ remap_idx[ i] for i in range(len(xyz))]
            return xyz[ inv]

        ret_str = []
        #if n < 192:
        #    continue
        entry_node = next(self.source.node_iter_to_root(target, select="Entry"))
        attrs = self.source.db[entry_node.payload]['data'].dict()['attributes']
        # attrs = self.source.db[target.payload]['data'].dict()['attributes']

        # Since we now have spec nodes, need one under the spec
        # Just grab the first.. only need metadata (e.g. mapping)
        opt_node = next(self.source.node_iter_depth_first(entry_node, select="Optimization"))
        qcmolid = self.source.db.get(opt_node.payload).get('data').get( 'initial_molecule')

        if isinstance( qcmolid, list):
            qcmolid = qcmolid[0]
        qcmolid = 'QCM-' + str(qcmolid)
        if qcmolid in self.source.db:
            qcmol = self.source.db.get(qcmolid).get( "data")
        else:
            ret_str.append("ERROR: Molcule ID {:s} not in the local database\n".format(qcmolid))
            return { target: ret_str }

        smiles_pattern = attrs.get('canonical_isomeric_explicit_hydrogen_mapped_smiles')

        mol = rdutil.mol.build_from_smiles( smiles_pattern)
        map_idx  = rdutil.mol.atom_map( mol)
        ret = rdutil.mol.embed_qcmol_3d( mol, qcmol)
        if ret < 0:
            ret_str.append("ERROR: Could not generate a conformation in RDKit. {} {}\n".format(qcmolid, target.payload))
            qca.qcmol_to_xyz( qcmol,
                fnm="mol."+qcmolid+"."+target.index+".rdconfgenfail.xyz", comment=qcmolid + " rdconfgen fail " + target.payload)
            return {target: ret_str }
        conf = mol.GetConformer()
        #ids = AllChem.EmbedMultipleConfs( mol, numConfs=1)
        #try:
        #    conf = mol.GetConformer(ids[0])
        #except IndexError:
        #    print("ERROR: Could not generate a conformation in RDKit.", qcmolid, target.payload)
        #    qca.qcmol_to_xyz( qcmol,
        #        fnm="mol."+qcmolid+"."+target.payload+".rdconfgenfail.xyz", comment=qcmolid + " rdconfgen fail " + target.payload)
        #    continue
        #conf = mol.GetConformer(ids[0])



        use_min_mol_for_charge = True
        if use_min_mol_for_charge:
            minidx = None
            minene = None
            minmol = None
            try:
                for opt in self.source.node_iter_depth_first( target, select="Optimization"):
                    status = self.source.db.get( opt.payload).get( "data").get("status")[:]
                    if status != "COMPLETE":
                        ret_str.append("This opt is not complete.. skipping..\n")
                        continue
                    allene = self.source.db.get( opt.payload).get( "data").get( "energies")
                    if allene is None:
                        ret_str.append("ERROR: No energies. {} {}\n".format( qcmolid, target.payload))
                        ret_str.append(str(e) + '\n')
                        qca.qcmol_to_xyz( qcmol, \
                            fnm="mol."+qcmolid+"."+target.index+".noenefail.xyz",\
                            comment=qcmolid + " noene fail " + target.payload)
                        return { target: ret_str }
                    ene = allene[ -1]
                    if minene is None:
                        minene = ene
                        minidx = opt.children[-1]
                    elif ene < minene:
                        minene = ene
                        minidx = opt.children[-1]
            except (TypeError, IndexError) as e:
                # ene is not a list above if TypeError
                # IndexError it is []
                ret_str.append("ERROR: No energies. {} {}\n".format( qcmolid, target.payload))
                ret_str.append(str(e) + '\n')
                qca.qcmol_to_xyz( qcmol, \
                    fnm="mol."+qcmolid+"."+target.index+".noenefail.xyz",\
                    comment=qcmolid + " noene fail " + target.payload)
                return { target: ret_str }


            if minidx == None:
                ret_str.append("EMPTY. SKIPPING\n")
                return { target: ret_str }
            grad_node = self.source.node_index[ minidx]
            mol_node  = self.source.node_index[ grad_node.children[0]]
            qcmol = self.source.db[ mol_node.payload][ 'data']

            ret_str.append("min mol is {} ene is {} au\n".format(mol_node, minene))

        xyz = qcmol.get( "geometry")
        sym = qcmol.get( "symbols")
        #for i, a in enumerate(mol.GetAtoms()):
        #    conf.SetAtomPosition(i, xyz[ map_idx[ i] - 1] * const.bohr2angstrom)

        #Chem.rdmolops.AssignStereochemistryFrom3D( mol, ids[0], replaceExistingTags=True)
        #Chem.rdmolops.AssignStereochemistryFrom3D( mol, ids[0], replaceExistingTags=True)
        Chem.rdmolops.AssignAtomChiralTagsFromStructure( mol, -1, replaceExistingTags=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            mmol = oFF.topology.Molecule.from_rdkit( mol, allow_undefined_stereo=True)
            try:
                top = oFF.topology.Topology().from_molecules( mmol)
            except AssertionError:
                ret_str.append("ERROR: Could not setup molecule in oFF. {} {}\n".format( qcmolid, target.payload))
                qca.qcmol_to_xyz( qcmol,
                    fnm="mol."+qcmolid+"."+target.index+".offmolfail.xyz", comment=qcmolid + " oFF fail " + target.payload)
                #pdb.set_trace()
                return { target: ret_str }
            try:
                mmol.compute_partial_charges_am1bcc()

            except Exception as e:
                os.chdir(cwd)
                ret_str.append("ERROR: Could not compute partial charge. {} {}\n".format( qcmolid, target.payload))
                ret_str.append( str(e.__traceback__.__repr__()) + '\n' )
                ret_str.append( str(e) + '\n')
                qca.qcmol_to_xyz( qcmol,
                    fnm="mol."+qcmolid+"."+target.index+".chrgfail.xyz", comment=qcmolid + " charge fail "+target.payload)
                #pdb.set_trace()
                return { target: ret_str }
            os.chdir(cwd)

        gen_MM_charge = [mmol.partial_charges]

        fail = True
        nodes = list(self.source.node_iter_depth_first(
            target, select="Molecule"))
        order = np.arange( len( nodes))
        vals = []
        for mol_node in nodes:
            val = tuple([ c.payload[2] for c in \
                self.source.node_iter_to_root( mol_node,
                    select="Constraint")])
            if len(val) > 0:
                vals.append(val)
        if len(vals) > 0:
            vals = np.array( vals)
            order = np.lexsort( vals.T)
            nodes_in_order = [nodes[i] for i in order]
        else:
            nodes_in_order = nodes

        ret_obj = {}

        for mol_node in nodes_in_order:
            fail = True
            qcmol = self.source.db[ mol_node.payload][ 'data']
            xyz = qcmol[ 'geometry']
            xyz = xyz * const.bohr2angstrom

            # this will take the xyz from qcmol and put them in the order of 
            # the cmiles, based on the mapped indices (1-index)
            # sends qcmol to mmmol
            xyz = unmap( xyz, map_idx)

            constraints = None
            if self.constrain:
                # constraints = [ [c.payload[1], c.payload[2]] for c in \
                # self.source.node_iter_to_root( mol_node, select="Constraint")]
                # print("constraints are", constraints)
                swap_map = {v-1:k for k,v in map_idx.items()}
                constraints = [ [[swap_map[i] for i in c.payload[1]], c.payload[2]] for c in \
                self.source.node_iter_to_root( mol_node, select="Constraint")]
                # qca.qcmol_to_xyz( {"symbols": unmap(qcmol['symbols'],  map_idx), "geometry":xyz*const.angstrom2bohr},
                #     fnm="mol."+qcmolid+"."+target.index+".map.xyz", comment=qcmolid + " map fail " + str(constraints))
                # constraints = [ [unmap(np.array(c.payload[1]), map_idx), c.payload[2]] for c in \
                # self.source.node_iter_to_root( mol_node, select="Constraint")]
                
                # print("angle before OpenMM is ", self.calculate_dihedral(xyz, *constraints[0][0])/unit.degrees)
                # print("constraints are", constraints)
            try:
                exc_info = sys.exc_info()
                total_ene, pos = self.calc_mm_energy(top, xyz, charge=gen_MM_charge, constraints=constraints)
                fail = False
            except UnassignedProperTorsionParameterException as e:
                ret_str.append("ERROR: oFF could not assign torsion for {} {}\n".format( qcmolid, target.payload))
                ret_str.append( str(e) + '\n')
                break
            except Exception as e:
                ret_str.append("ERROR: oFF exception for {} {}\n".format( qcmolid, target.payload))
                ret_str.append( str(e.__traceback__.__repr__()) + '\n' )
                ret_str.append( str(e) + '\n')
                fail=True
                break

            

            constraints = [c.payload for c in self.source.node_iter_to_root( mol_node, select="Constraint")]
            # print("    {} {} {}\n".format( mol_node, constraints, total_ene), end="")
            ret_str.append("    {} {} {}\n".format( mol_node, constraints, total_ene))
            
            pl = {}
            if self.minimize and pos is not None:
                pl = qcmol.copy()
                pl['geometry'] = remap(np.array(pos) * const.angstrom2bohr, map_idx)

            pl['energy'] = total_ene


            ret_obj.update({mol_node.payload : { "data": pl}})
            # self.db.__setitem__( )


        if fail:
            qca.qcmol_to_xyz( qcmol,
                fnm="mol."+qcmolid+"."+target.index+".labelfail.xyz",
                comment=qcmolid + " label fail " + target.payload)
        return { target: ret_str , "return": ret_obj}

    # def apply( self, targets=None, procs=1):
    #     if targets is None:
    #         targets = list(self.source.iter_entry())
    #     elif not hasattr( targets, "__iter__"):
    #         targets = [targets]

    #     # expand if a generator
    #     targets = list(targets)

    #     n_targets = len(targets)




    #     if self.processes > 1:
    #         import concurrent.futures
    #         exe = concurrent.futures.ProcessPoolExecutor(max_workers=self.processes)

    #         work = [ exe.submit( OpenMMEnergy.apply_single, self, n, target ) for n, target in enumerate(targets, 1) ]
    #         for n,future in enumerate(concurrent.futures.as_completed(work), 1):
    #             if future.done:
    #                 try:
    #                     val = future.result()
    #                 except RuntimeError:
    #                     print("RUNTIME ERROR; race condition??")
    #             if val is None:
    #                 print("data is None?!?")
    #                 continue
    #             for tgt, ret in val.items():
    #                 if tgt == "return":
    #                     self.db.update(ret) 
    #                 else:
    #                     print( n,"/", n_targets, tgt)
    #                     for line in ret:
    #                         print(line, end="")

    #         exe.shutdown()

        # This version seems to be more stable, but all of the results must
        # finish before iterating

        # exe = Pool(processes=self.processes)
        # work = [ exe.apply_async( OpenMMEnergy.apply_single, ( self, n, target) ) for n, target in enumerate(targets, 1) ]
        # out = [result.get() for result in work if result is not None]
        # for n, val in enumerate(out, 1):
        #     for tgt, ret in val.items():
        #         if tgt == "return":
        #             self.db.update(ret) 
        #         else:
        #             print( n,"/", n_targets, tgt)
        #             for line in ret:
        #                 print(line, end="")
        # exe.close()

        # for i, val in enumerate( out):
        #     for tgt, ret in val.items():
        #         print( n,"/", n_targets, tgt)
        #         print( ret)



        # # single process mode; does not launch any new processes
        # if self.processes == 1:
        #     for n, target in enumerate( targets, 1):
        #         val = self.apply_single( n, target)
        #         for tgt, ret in val.items():
        #             if tgt == "return":
        #                 self.db.update(ret) 
        #             else:
        #                 print( n,"/", n_targets, tgt)
        #                 for line in ret:
        #                     print(line, end="")
    ########################################
    def calculate_dihedral ( self, coords, idx1, idx2, idx3, idx4 ):
        # from:
            # https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
            # https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
        b1 = (coords[idx1] - coords[idx2] )
        b1 /= np.sqrt(np.dot(b1,b1))
        b2 = coords[idx2] - coords[idx3]
        b2 /= np.sqrt(np.dot(b2,b2))
        b3 = coords[idx3] - coords[idx4]
        b3 /= np.sqrt(np.dot(b3,b3))
        # print (b1, b2, b3)
        n1 = np.cross( b1, b2 )
        n2 = np.cross( b2, b3 )
        m1 = np.cross( n1, b2 )
        # print ( n1,n2,m1 )
        x  = np.dot ( n2, n1 )
        y  = np.dot ( n2, m1 )
        theta = np.arctan2( y, x ) * unit.radian
        # print ( theta, np.degrees ( theta ) * unit.degree )
        return theta

    def add_harmonic_dihedral_restraints ( self, system, coords, k, hw, indices_to_restrain, debug=False ):
        print ( "Adding harmonic dihedral restraints: " )
        if len(indices_to_restrain) != 4:
            print ( "WARNING: Must be exactly 4 atoms specified for a dihedral restraint.  Skipping ..." )
            return system
        theta0 = self.calculate_dihedral ( coords, *indices_to_restrain )    # in radians
        print ( theta0 )
        expr = "0.5*k*max(0,( min(d_theta,2*pi-d_theta) - hw ))^2;"
        expr += "d_theta = abs(theta - theta0);"
        expr += f'pi = {np.pi:.10f}'
        force = openmm.CustomTorsionForce( expr )
        force.addGlobalParameter ( "k", k )
        force.addGlobalParameter ( "hw", hw )
        force.addPerTorsionParameter ( "theta0" )
        force.addTorsion ( *indices_to_restrain, [ theta0 ] )
        system.addForce ( force )
        return system

    def mm_potential(self, forcefield, top, xyz, charge=False, constraints=None, use_geometric=False):

        if isinstance( charge, bool):
            if( charge):
                system = forcefield.create_openmm_system( top)
            else:
                mols = [Molecule(mol.reference_molecule) for mol in top.topology_molecules]
                for i,_ in enumerate(mols):
                    mols[i].partial_charges = simtk.unit.Quantity( np.zeros( mols[i].n_atoms), simtk.unit.elementary_charge)
                system = forcefield.create_openmm_system(top, charge_from_molecules=mols)
        else:
            mols = [Molecule( mol.reference_molecule) for mol in top.topology_molecules]
            for i,_ in enumerate( mols):
                mols[ i].partial_charges = charge[ i]
            system = forcefield.create_openmm_system( top, charge_from_molecules=mols)


        integrator = openmm.VerletIntegrator(0.1 * simtk.unit.femtoseconds)
        sim = openmm.app.simulation.Simulation(top, system, integrator)
        sim.context.setPositions(xyz * const.angstrom2nm)


        # params = forcefield.label_molecules(top)[0]
        # for p in params:
        #     print(p)
        #     print(dict(params[p]))

        getPositions = self.minimize
        # system = sim.context.getSystem()
        # forces = system.getForces()
        use_geometric=True
        if True and use_geometric and self.minimize:

            import geometric.optimize as geoopt
            with tempfile.TemporaryDirectory() as tmpdir:
                cwd = os.getcwd()
                os.chdir(tmpdir)
                system = forcefield.create_openmm_system( top, charge_from_molecules=mols)
                xml = openmm.XmlSerializer.serialize(system)
                open("system.xml",'w').write(xml)
                with open("mol.pdb", 'w') as fid:
                    openmm.app.pdbfile.PDBFile.writeFile(top.to_openmm(),xyz,fid)

                args = {"input": "system.xml",
                         "pdb": "mol.pdb",
                         "openmm": True,
                         "maxiter": 2000 }
                if constraints is not None:

                    with open("constraints", 'w') as fid:
                        fid.write("$set\n")
                        for constr in constraints:
                            ids = [x+1 for x in constr[0]]
                            val = constr[1]
                            out_str = "dihedral {:d} {:d} {:d} {:d} {:12.5f}\n".format(*ids, val)
                            # out_str = "dihedral {:d} {:d} {:d} {:d}\n".format(*ids)
                            fid.write(out_str)
                            # print(out_str, end="")
                    args["constraints"] = "constraints"

                success = False
                with tempfile.TemporaryFile('w') as null:
                    sys.stderr = null
                    try:
                        geoopt.run_optimizer(**args)
                        success = True
                    except Exception as e:
                        print("This optimization failed!")
                        print(e)
                    sys.stderr = sys.__stderr__
                    if success:
                        _, traj, ene = load_geometric_opt_trj_xyz("system_optim.xyz")
                os.chdir(cwd)

            if not success:
                return None, None

            return ene[-1]*const.hartree2kcalmol, traj[-1]


        elif False and self.minimize and constraints is not None:
            print("Using frozen torsion...")
            for constr in constraints:
                # assume torsion...
                # print("Searching torsion", constr)
                for idx in constr[0]:
                    system.setParticleMass(idx, 0.0)
            if self.minimize:
                sim.minimizeEnergy()
            state = sim.context.getState(getEnergy=True, getPositions=getPositions)
            energy = state.getPotentialEnergy().in_units_of(simtk.unit.kilocalories_per_mole)
            pos = None
            if self.minimize:
                pos = state.getPositions(asNumpy=True) / simtk.unit.angstroms

        elif False and self.minimize and constraints is not None:
            print("Using custom torsion force...")
            cforce = openmm.CustomTorsionForce("0.5*k*max(0.0,min(dtheta, 2*pi-dtheta))^2; dtheta = abs(theta-theta0); pi = 3.1415926535")
            cforce.addPerTorsionParameter("k")
            cforce.addPerTorsionParameter("theta0")
            for constr in constraints:
                # assume torsion...
                # print("Searching torsion", constr)
                if len(constr[0]) == 4:
                    print("adding torsion", constr[0])
                    cforce.addTorsion(*constr[0],[300.0, constr[1] * np.pi/180])

            # print("adding force...")
            iforce = system.addForce(cforce)

            integrator = openmm.VerletIntegrator(0.1 * simtk.unit.femtoseconds)
            sim = openmm.app.simulation.Simulation(top, system, integrator)
            sim.context.setPositions(xyz * const.angstrom2nm)
            getPositions = self.minimize
            # system = sim.context.getSystem()
            # forces = system.getForces()
            state = sim.context.getState(getEnergy=True, getPositions=getPositions)
            energy = state.getPotentialEnergy().in_units_of(simtk.unit.kilocalories_per_mole)
            print("energy is", energy)
            if self.minimize:
                sim.minimizeEnergy()
            state = sim.context.getState(getEnergy=True, getPositions=getPositions)
            energy = state.getPotentialEnergy().in_units_of(simtk.unit.kilocalories_per_mole)
            pos = state.getPositions(asNumpy=True) / simtk.unit.nanometers
            print("energy is", energy)

            system2 = forcefield.create_openmm_system( top, charge_from_molecules=mols)
            integrator = openmm.VerletIntegrator(0.1 * simtk.unit.femtoseconds)
            sim = openmm.app.simulation.Simulation(top, system2, integrator)
            sim.context.setPositions(pos)
            state = sim.context.getState(getEnergy=True, getPositions=getPositions)
            energy = state.getPotentialEnergy().in_units_of(simtk.unit.kilocalories_per_mole)
            print("energy is\n", energy)
            pos = None
            if self.minimize:
                pos = state.getPositions(asNumpy=True) / simtk.unit.angstroms
            return energy, pos

        elif self.minimize and constraints is not None:
            forces = system.getForces()
            print("Using iterative torsion force...")
            state = sim.context.getState(getEnergy=True, getPositions=getPositions)
            pos = state.getPositions(asNumpy=True) / simtk.unit.angstroms
            print("angle before OpenMM opt is ", self.calculate_dihedral(pos, *constraints[0][0])/unit.degrees)
            dx = list([np.inf]*len(constraints))
            dk = 1.0 * simtk.unit.kilojoules_per_mole
            dE = 0.0
            energy = 0.0
            eps = 0.001
            maxsteps = 100000
            nstep = 0
            pha = None
            k = None
            per = 0
            ref_k = None
            ref_pha = None
            while nstep < maxsteps and all([abs(dxi) > eps for dxi in dx]):
                for constr in constraints:
                    # assume torsion...
                    # print("Searching torsion", constr)
                    if len(constr[0]) == 4:


                        torsions = [f for f in forces if type(f) == openmm.PeriodicTorsionForce][0]
                        N = torsions.getNumTorsions()
                        found = False
                        # print("Found N existing torsions:", N)

                        if pha is None:
                            pha = constr[1]*np.pi/180
                        for idx in range(N):
                            p1,p2,p3,p4,per,phai,ki = torsions.getTorsionParameters(idx)
                            if k is None:
                                k = ki
                                ref_k = ki
                                ref_pha = phai
                            target = tuple(constr[0])
                            if target[-1] < target[0]:
                                target = tuple(target[::-1])
                            # print("Retreived", p1,p2,p3,p4,per,pha,ki, target)
                            if tuple([p1,p2,p3,p4]) == target:
                                # print("retreived", idx, p1,p2,p3,p4,per,phai,ki)
                                # print("Found torsion", target, constr[1])
                                # print("mass: ", [system.getParticleMass(x) for x in constr[0]])
                                k += dk
                                # per = 1
                                # print("setting", idx, p1,p2,p3,p4,per,pha,k)
                                torsions.setTorsionParameters(idx, p1,p2,p3,p4,per,pha,k)
                                found = True
                                # print("Set to", idx, p1, p2, p3, p4, per, pha, k)
                        if not found:
                            raise Exception("The constrained torsion was not found!")
                            # p1,p2,p3,p4 = constr[0]
                            # pha = constr[1]*np.pi/180.
                            # k = 9999999.0
                            # per = 1
                            #constr[1],  torsions.addTorsion(p1,p2,p3,p4,per,pha,k)
                        torsions.updateParametersInContext(sim.context)
                sim.minimizeEnergy(tolerance=1e-5, maxIterations=1)
                state = sim.context.getState(getEnergy=True, getPositions=getPositions)
                energy2 = state.getPotentialEnergy() / simtk.unit.kilocalories_per_mole
                pos = state.getPositions(asNumpy=True) / simtk.unit.angstroms

                # now check to see if the constraints moved
                for i,constr in enumerate(constraints):
                    # assume torsion...
                    # print("Calculating torsion", constr)
                    if len(constr[0]) == 4:
                        angle = offsb.op.geometry.TorsionOperation.measure_praxeolitic_single({"geometry":pos}, list(target))
                        dx[i] = abs(angle - constr[1])
                        dE = energy2 - energy
                        energy=energy2
                        # print("Dx", i, "is ", dx[i], "k=", k, target, constr[1], angle,"dE=",dE, "pha=", pha * 180/np.pi)
                        angle2 = self.calculate_dihedral(pos, *list(target)) / unit.degrees
                        if nstep == 0 or nstep == maxsteps or all([abs(dxi) < eps for dxi in dx]):
                            print("C {:d} dt= {:14.8e} k= {:12.3f} {:s} t_0= {:6.2f} t= {:6.2f} t2= {:6.2f} dE= {:16.8e} pha= {:14.8e}".format(i, dx[i], k/unit.kilojoules_per_mole, str(target), constr[1], angle, angle2, dE, pha * 180/np.pi))

                        # angle = constr[1] *np.pi/180
                        # if angle < constr[1] and angle < 0:
                        #     pha *= 1. - (.01 / np.log10(nstep+2)
                        #     # print("decrease1")
                        # elif angle < constr[1] and angle > 0:
                        #     pha *= 1. + (.1 /np.log10(nstep+2))
                        #     # print("increase1")
                        # elif angle > constr[1] and angle > 0:
                        #     pha *= 1. - (.1 / np.log10(nstep+2)
                        #     # print("decrease2")
                        # elif angle > constr[1] and angle < 0:
                        #     pha *= 1. + (.1 /np.log10(nstep+2))
                        #     # print("increase2")
                        pha -= (0.75)*((angle - constr[1])*np.pi/180)
                        if pha < -2*np.pi:
                            pha += 2*np.pi
                        elif pha > 2*np.pi:
                            pha -= 2*np.pi

                nstep += 1

            for constr in constraints:
                # assume torsion...
                # print("Searching torsion", constr)
                if len(constr[0]) == 4:


                    torsions = [f for f in forces if type(f) == openmm.PeriodicTorsionForce][0]
                    N = torsions.getNumTorsions()
                    found = False
                    # print("Found N existing torsions:", N)

                    for idx in range(N):
                        p1,p2,p3,p4,per,phai,ki = torsions.getTorsionParameters(idx)
                        target = tuple(constr[0])
                        if target[-1] < target[0]:
                            target = tuple(target[::-1])
                        if tuple([p1,p2,p3,p4]) == target:
                            # print("Found torsion", target, constr[1])
                            # per = 1
                            torsions.setTorsionParameters(idx, p1,p2,p3,p4,per,ref_pha,ref_k)

            state = sim.context.getState(getEnergy=True, getPositions=getPositions)
            energy = state.getPotentialEnergy().in_units_of(simtk.unit.kilocalories_per_mole)
            # print("energy (remove)", energy)
            torsions.updateParametersInContext(sim.context)
            state = sim.context.getState(getEnergy=True, getPositions=getPositions)
            energy = state.getPotentialEnergy().in_units_of(simtk.unit.kilocalories_per_mole)
            # print("energy (remove)", energy)
            pos = None
            if self.minimize:
                pos = state.getPositions(asNumpy=True) / simtk.unit.angstroms



        else:
            if self.minimize:
                sim.minimizeEnergy()
            state = sim.context.getState(getEnergy=True, getPositions=getPositions)
            energy = state.getPotentialEnergy().in_units_of(simtk.unit.kilocalories_per_mole)
            pos = None
            if self.minimize:
                pos = state.getPositions(asNumpy=True) / simtk.unit.angstroms
        return energy, pos

    def calc_mm_energy(self, top, xyz, component=None, charge=False, constraints=None):
        forcefield = self.forcefield
        if(component is None):
            ene,pos = self.mm_potential(forcefield, top, xyz, charge=charge, constraints=constraints)
            return ene,pos

        modff = copy.deepcopy(forcefield)

        force = modff.get_parameter_handler(component)
        for term in force.parameters:
            if(component == "vdW"):
                term.epsilon *= 0.0
            if(component in ["Bonds", "Angles"]):
                term.k *= 0.0
            if(component in ["ProperTorsions", "ImproperTorsions"]):
                for i,_ in enumerate(term.k):
                    term.k[i] *= 0.0

        ene, pos = self.mm_potential(modff, top, xyz, charge=charge, constraints=constraints)
        return ene, pos

#    def calc_vdw_direct(xyz, labels):
#        """ doesn't work yet"""
#        na = len(xyz)
#        atoms = range(na)
#        ene = 0.0
#        r = distance.cdist(xyz,xyz)
#        for i in atoms:
#            ii = (i,)
#            for j in atoms:
#                if(j >= i):
#                    break
#                jj = (j,)
#                eps = np.sqrt(labels[ii].epsilon * labels[jj].epsilon) / labels[jj].epsilon.unit
#                rmin = (labels[ii].rmin_half + labels[jj].rmin_half)/2.0
#                rmin = rmin / rmin.unit
#                rij = r[i,j]
#                a = rmin/rij
#                a = a**6
#                ene += eps * (a**2 - 2*a)
#                #print(i,j,"r", rij, "ene", ene, "rmin", rmin, "eps", eps, "a", a)
#        return ene * simtk.unit.kilocalorie_per_mole

#    def apply( self, targets=None):
#        gen_MM_charge = True
#        try:
#            total_ene = self.calc_mm_energy(forcefield, top, xyz, charge=gen_MM_charge)
#        except Exception as e:
#            log.write(str(e) + '\n')
#            success = False
#            break
        #log.write("  Conformation energy {:4d}/{:4d}\n".format(qcmol_i+1,n_qcmol))
        #log.flush()
        #if(get_qm_energy):
        #    ene_str = "{:13.8f} a.u.".format(ene[qcmol_i])
        #    log.write("    QM {:18s}= {:s} \n".format("Energy", ene_str))
        #    log.flush()

        #ene_name_str = "Energy" if gen_MM_charge else "EnergyNoElec"
        #log.write("    MM {:18s}= {:10.5f} {:s}\n".format(ene_name_str,
        #    (total_ene/total_ene.unit ),
        #    str(total_ene.unit)))
        #log.flush()

        #sim.context.setPositions(xyz * angstrom2nm)
        #state = sim.context.getState(getEnergy = True, getPositions=True)
        #sim_ene = state.getPotentialEnergy()
        #reporter.report(sim, state)
        #log.write("    MM {:18s}= {:9.4f} {:s}\n\n".format("App Energy",
        #    (sim_ene.value_in_unit(simtk.unit.kilocalorie_per_mole)),
        #    str(simtk.unit.kilocalorie_per_mole)))
        #log.flush()
        #sim.minimizeEnergy()
        #state = sim.context.getState(getEnergy = True, getPositions=True)
        #sim_min_ene = state.getPotentialEnergy()
        #reporter.report(sim, state)
        #log.write("    MM {:18s}= {:9.4f} {:s}\n\n".format("App Min Energy",
        #    (sim_min_ene.value_in_unit(simtk.unit.kilocalorie_per_mole)),
        #    str(simtk.unit.kilocalorie_per_mole)))


        #log.flush()
#        ene_sum = total_ene
        #if('direct_vdw' not in mol_data['energy']['oFF']):
        #    mol_data['energy']['oFF']['direct_vdw'] = []
        #mol_data['energy']['oFF']['direct_vdw'].append(calc_vdw_direct(xyz, labels['vdW']))
        #print(mol_data['energy']['oFF']['direct_vdw'])

#        if('epot' not in mol_data['energy']['oFF']):
#            mol_data['energy']['oFF']['epot'] = []
#        mol_data["energy"]['oFF']['epot'].append(total_ene)
#        for component in ['vdW' ] + valence_params:
#            energy_sans_component = calc_mm_energy(forcefield, top, xyz, component=component, charge=gen_MM_charge)
#            energy_component = total_ene - energy_sans_component
#
#            if(component not in mol_data["energy"]['oFF']):
#                mol_data["energy"]['oFF'][component] = []
#            mol_data["energy"]['oFF'][component].append(energy_component)
#            ene_sum -= energy_component
#            log.write("    MM {:18s}= {:10.5f} {:s}\n".format(component,
#                (energy_component/energy_component.unit ),
#                str(energy_component.unit)))
#            log.flush()
#        if(gen_MM_charge):
#            if('Electrostatics' not in mol_data['energy']['oFF']):
#                mol_data['energy']['oFF']['Electrostatics'] = []
#            mol_data['energy']['oFF']['Electrostatics'].append(ene_sum)
#            log.write("    MM {:18s}= {:10.5f} {:s}\n\n".format("Electrostatics",
#                (ene_sum/ene_sum.unit ),
#                str(ene_sum.unit)))
#
#        #log.write("    MM {:18s}= {:10.5f} {:s}\n\n".format("Direct vdW",
#        #    (mol_data['energy']['oFF']['direct_vdw'][qcmol_i] / mol_data['energy']['oFF']['direct_vdw'][qcmol_i].unit ),
#        #    str(mol_data['energy']['oFF']['direct_vdw'][qcmol_i].unit)))
#        log.flush()

#    if(success == False):
#        continue
#    log.write("\n  Conformation energy summary\n")
#    if(len(ene) > 1):
#        if(get_qm_energy):
#            ene_str = "{:13.8f} +- {:13.8f} a.u.".format(np.mean(ene), np.std(ene))
#            log.write("    QM {:18s}= {:s} \n".format("Energy", ene_str))
#
#        log.write("    MM {:18s}= {:10.5f} +- {:10.5f} {:s}\n\n".format(ene_name_str,
#            np.mean([i/i.unit for i in mol_data["energy"]['oFF']['epot']]),
#            np.std([i/i.unit for i in mol_data["energy"]['oFF']['epot']]),
#            str(mol_data["energy"]['oFF']['epot'][0].unit)))
#        components_to_print = ['vdW' ] + valence_params
#        if(gen_MM_charge):
#            components_to_print += ['Electrostatics']
#        for component in components_to_print:
#            log.write("    MM {:18s}= {:10.5f} +- {:10.5f} {:s}\n".format(component,
#                np.mean([i/i.unit for i in mol_data["energy"]['oFF'][component]]),
#                np.std([i/i.unit for i in mol_data["energy"]['oFF'][component]]),
#                str(mol_data["energy"]['oFF'][component][0].unit)))

