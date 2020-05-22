#!/usr/bin/env python3
import os
import pdb
import treedi
import treedi.tree
import simtk.unit
from simtk import openmm
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
import copy
import numpy as np
from multiprocessing import Pool
import threading
import sys



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

    def apply_single( self, i, target):

        def unmap( xyz, map_idx):
            inv = [ ( map_idx[ i] - 1) for i in range(len(xyz))]
            return xyz[ inv]

        ret_str = []
        #if n < 192:
        #    continue
        attrs = self.source.db.get( target.payload).get( 'entry').dict().get( 'attributes')
        qcmolid = self.source.db.get( target.payload).get( 'data').get( 'initial_molecule')

        if isinstance( qcmolid, list):
            qcmolid = qcmolid[0]
        qcmolid = 'QCM-' + str( qcmolid)
        qcmol = self.source.db.get( qcmolid).get( "data")
        smiles_pattern = attrs.get( 'canonical_isomeric_explicit_hydrogen_mapped_smiles')

        mol = rdutil.mol.build_from_smiles( smiles_pattern)
        map_idx  = rdutil.mol.atom_map( mol)
        ret = rdutil.mol.embed_qcmol_3d( mol, qcmol)
        if ret < 0:
            ret_str.append("ERROR: Could not generate a conformation in RDKit. {} {}\n".format(qcmolid, target.payload))
            qca.qcmol_to_xyz( qcmol,
                fnm="mol."+qcmolid+"."+target.payload+".rdconfgenfail.xyz", comment=qcmolid + " rdconfgen fail " + target.payload)
            return { target: ret_str }
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
                    ene = self.source.db.get( opt.payload).get( "data").get( "energies")[ -1]
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
                ret_str.append(str(e))
                qca.qcmol_to_xyz( qcmol, \
                    fnm="mol."+qcmolid+"."+target.payload+".noenefail.xyz",\
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

        mmol = oFF.topology.Molecule.from_rdkit( mol, allow_undefined_stereo=True)
        try:
            top = oFF.topology.Topology().from_molecules( mmol)
        except AssertionError:
            ret_str.append("ERROR: Could not setup molecule in oFF. {} {}\n".format( qcmolid, target.payload))
            qca.qcmol_to_xyz( qcmol,
                fnm="mol."+qcmolid+"."+target.payload+".offmolfail.xyz", comment=qcmolid + " oFF fail " + target.payload)
            #pdb.set_trace()
            return { target: ret_str }
        try:
            mmol.compute_partial_charges_am1bcc()
        except Exception:
            ret_str.append("ERROR: Could not compute partial charge. {} {}\n".format( qcmolid, target.payload))
            qca.qcmol_to_xyz( qcmol,
                fnm="mol."+qcmolid+"."+target.payload+".chrgfail.xyz", comment=qcmolid + " charge fail "+target.payload)
            #pdb.set_trace()
            return { target: ret_str }
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
        for mol_node in nodes_in_order:
            fail = True
            qcmol = self.source.db[ mol_node.payload][ 'data']
            xyz = qcmol[ 'geometry']
            xyz = xyz * const.bohr2angstrom
            xyz = unmap( xyz, map_idx)
            try:
                exc_info = sys.exc_info()
                total_ene = self.calc_mm_energy( top, xyz, charge=gen_MM_charge)
                fail = False
            except UnassignedProperTorsionParameterException as e:
                ret_str.append("ERROR: oFF could not assign torsion for {} {}\n".format( qcmolid, target.payload))
                ret_str.append( str(e) + '\n')
                break
            except Exception as e:
                ret_str.append("ERROR: oFF exception for {} {}\n".format( qcmolid, target.payload))
                ret_str.append( str(e.__traceback__().__repr__()) + '\n' )
                ret_str.append( str(e) + '\n')
                raise

                break


            constraints = [c.payload for c in self.source.node_iter_to_root( mol_node, select="Constraint")]
            ret_str.append("    {} {} {}\n".format( mol_node, constraints, total_ene))
            self.db.__setitem__( mol_node.payload, { "data": { "energy": total_ene }})

        if fail:
            qca.qcmol_to_xyz( qcmol,
                fnm="mol."+qcmolid+"."+target.payload+".labelfail.xyz",
                comment=qcmolid + " label fail " + target.payload)
        return { target: ret_str }

    def apply( self, targets=None, procs=1):
        if targets is None:
            targets = list(self.source.iter_entry())
        elif not hasattr( targets, "__iter__"):
            targets = [targets]

        # expand if a generator
        targets = list(targets)

        n_targets = len(targets)


        #exe = Pool(processes=procs)
        #work = [ exe.apply_async( OpenMMEnergy.apply_single, ( self, n, target) ) for n, target in enumerate(targets, 1) ]
        #out = [result.get() for result in work]
        #for i, val in enumerate( out):
        #    for tgt, ret in val.items():
        #        print( n,"/", n_targets, tgt)
        #        print( ret)

        #exe.close()


        for n, target in enumerate( targets, 1):
            ret = self.apply_single( n, target)
            for tgt, ret in ret.items():
                print( n,"/", n_targets, tgt)
                [ print( line, end="") for line in ret ]


        import concurrent.futures


    def mm_potential(self, forcefield, top, xyz, charge=False):

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


        integrator = openmm.VerletIntegrator(1.0 * simtk.unit.femtoseconds)
        context = openmm.Context(system, integrator)
        context.setPositions(xyz * const.angstrom2nm)
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().in_units_of(simtk.unit.kilocalories_per_mole)
        return energy

    def calc_mm_energy(self, top, xyz, component=None, charge=False):
        forcefield = self.forcefield
        if(component is None):
            ene = self.mm_potential(forcefield, top, xyz, charge=charge)
            return ene

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

        ene = self.mm_potential(modff, top, xyz, charge=charge)
        return ene

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

