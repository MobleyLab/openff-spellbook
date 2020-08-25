
import sys
import os
import itertools
import numpy as np
from scipy.spatial import distance
from scipy.sparse import dok_matrix
import qcfractal.interface as ptl
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import FragmentMatcher
from cmiles.utils import mol_to_map_ordered_qcschema
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
import openforcefield
import smirnoff99frosst as ff
import pickle
import time
import copy
from datetime import timedelta
from QUBEKit.mod_seminario import ModSeminario
from QUBEKit.parametrisation.base_parametrisation import Parametrisation
from QUBEKit.utils import constants
from simtk import openmm
import simtk 
from pydantic.error_wrappers import ValidationError

import logging
logger = logging.getLogger()
logger.setLevel(level=logging.ERROR)

bohr2angstrom = 0.529177249
angstrom2nm = .1
hartree2kcalmol = 627.5096080306


client = ptl.FractalClient()


from geometric.internal import *
from geometric.molecule import Molecule as geometric_mol
from geometric import optimize


def CalcInternalHess(coords, gradx, Hx, IC):
    """
    coords, gradx, hessx : Cartesian coordinates, gradients, and Hessian in atomic units.
    IC : geomeTRIC internal coordinate system (look at line 1766 for how to create this)
    """
    # Internal coordinate Hessian using analytic transformation
    Hq = IC.calcHess(coords, gradx, Hx)

    verbose = False
    if verbose:
        print("-=# Hessian of the energy in internal coordinates #=-\n")
        print("{:20s} {:20s} : {:14s}\n".format('IC1 Name', 'IC2 Name', 'Value'))
        for (i,j) in itertools.product(*([range(len(IC.Internals))]*2)):
            print("{:20s} {:20s} : {: 14.6f}".format(str(IC.Internals[i]), str(IC.Internals[j]), Hq[i, j]*(hartree2kcalmol/bohr2angstrom**2)))

    return Hq

#=========================================#
#| Set up the internal coordinate system |#
#=========================================#
# First item in tuple: The class to be initialized
# Second item in tuple: Whether to connect nonbonded fragments
# Third item in tuple: Whether to throw in all Cartesians (no effect if second item is True)
CoordSysDict = {'cart':(CartesianCoordinates, False, False),
                'prim':(PrimitiveInternalCoordinates, True, False),
                'dlc':(DelocalizedInternalCoordinates, True, False),
                'hdlc':(DelocalizedInternalCoordinates, False, True),
                'tric':(DelocalizedInternalCoordinates, False, False)}
coordsys = 'prim' # kwargs.get('coordsys', 'tric')
CVals = None
CoordClass, connect, addcart = CoordSysDict[coordsys.lower()]
Cons = None



def hyphenate_int_tuple(tup, offset=0):
    return "{:d}".format(tup[0]+offset)+ ("-{:d}"*(len(tup)-1)).format(*([x+offset for x in tup[1:]]))


def oFF_key_in_IC_Hessian(index, Internals):
    #off_str = hyphenate_int_tuple(index)
    exists = False
    debug = False
    i = 0
    n_atoms = len(index)
    debug_lines = []
    for i, Internal in enumerate(Internals):
        hess_atoms = str(Internal).split()[1]
        permutations = None
        if(n_atoms == 2):
            if(type(Internal) is not Distance):
                continue
            permutations = itertools.permutations(index)
        elif(n_atoms == 3):
            if(type(Internal) is not Angle):
                continue
            permutations = [[x[0],index[1],x[1]] for x in itertools.permutations([index[0],index[2]])]
        elif(n_atoms == 4):
            if(type(Internal) is OutOfPlane):
                # geometic molecule puts center as the first index
                #hess_atoms[0], hess_atoms[1] = hess_atoms[1], hess_atoms[0]
                reordered_index = tuple([index[1], index[0], index[2], index[3]])
                permutations = [[reordered_index[0]] + list(x) for x in itertools.permutations(reordered_index[1:])]
            elif(type(Internal) is Dihedral):
                # cross our fingers that dihedrals are sequential
                permutations = [index, index[::-1]]
            else:
                continue
        else:
            raise IndexError("Invalid number of atoms.")

        for order_i in permutations:
            candidate = hyphenate_int_tuple(order_i)
            debug_lines.append("comparing oFF indices " + str(candidate) + " to Hessian " + str(hess_atoms) + " failed")
            if(candidate == hess_atoms):
                if(debug):
                    print("HIT comparing oFF indices", candidate, "to Hessian", hess_atoms )
                return True, i
            
    if(debug):
        print("MISS for", index, "DEBUG:")
        for line in debug_lines:
            print(line)
        print()
            
    return False, -1


def flatten_list(l):
    return [val for vals in l for val in vals ]
    
def get_all_qcmolid_and_grads(td, minimum=False, get_energy=False, TorsionDrive=True, get_hessians=False, get_gradients=False):

    hessian = None
    ret_rec = None
    ene = None
    if(TorsionDrive):
        angles = td.get_history(minimum=minimum).keys()
        opt_rec_list = ([td.get_history(ang, minimum=minimum) for ang in angles])


        if(minimum):
            opt_rec = opt_rec_list
            if(get_gradients):
                ret_rec = ([opt.get_trajectory()[-1] for opt in opt_rec])

            opt_ang = flatten_list(angles)
            mol_id = [opt.get_final_molecule() for opt in opt_rec]
            if(get_energy):
                ene = [opt.get_final_energy() for opt in opt_rec]

        else:
            #TODO: Energy for intermediates
            opt_rec = flatten_list(opt_rec_list)
            ret_rec_list = ([opt.get_trajectory() for opt in opt_rec])
            ret_rec = flatten_list(ret_rec_list)
            ang_expand = [a * len(l) for a,l in zip(angles, opt_rec_list)]
            ang_full = [[a]*l for a,l in zip(flatten_list(ang_expand),[len(x) for x in ret_rec_list])]

            opt_ang = flatten_list(ang_full)
            mol_id = [rec.molecule for rec in ret_rec]
            if(energy):
                ene = [opt.get_final_energy() for opt in opt_rec]
            if(not get_gradients):
                ret_rec = None


        opt_ang = np.array(opt_ang)
        srt = opt_ang.argsort()
        mol_id = [mol_id[i] for i in srt]
        if(get_gradients):
            ret_rec = [ret_rec[i] for i in srt]

        ene = np.array(ene)[srt]
            
        opt_ang =  opt_ang[srt]

        info = {}
        tddict = td.dict(encoding="json")
        for key in ['keywords', 'id']:
            info[key] = tddict[key]

        tddict = None

    else:
        opt_rec = td
        convert = hartree2kcalmol/bohr2angstrom**2
        convert = 1
        opt_ang = [None]
        if(minimum):
            if(get_gradients):
                ret_rec = [opt.get_trajectory()[-1] for opt in opt_rec]

            mol_id = [opt.get_final_molecule() for opt in opt_rec]
            if(get_energy):
                ene = [opt.get_final_energy() for opt in opt_rec]
            if(get_hessians):
                hessian = []
                for i in mol_id:
                    try:
                        h = client.query_results(molecule=i.id,driver="hessian")[0].return_result
                        hessian.append(convert*np.array(h))
                    except (IndexError, TypeError):
                        print("No hessian for this geometry!")
                        hessian.append(None)

        else:
            ret_rec = [opt.get_trajectory() for opt in opt_rec]
            if(get_energy):
                ene = flatten_list([opt.energies for opt in opt_rec])

            mol_id = [rec.molecule for a in ret_rec for rec in a]
            if(get_hessians):
                hessian = []
                for i in mol_id:
                    try:
                        h = client.query_results(molecule=i.id,driver="hessian")[0].return_result
                        hessian.append(convert*np.array(h))
                    except (IndexError, TypeError):
                        print("No hessian for this geometry!")
                        hessian.append(None)


            if(not get_gradients):
                ret_rec = None
                            


        srt = np.arange(len(mol_id))

    
        info = {}
        tddict = td[0].dict(encoding="json")
        for key in ['keywords', 'id']:
            info[key] = tddict[key]

        tddict = None

    return mol_id, info, ret_rec, hessian, opt_ang, ene, srt

def get_qcmol_from_ids(client, mol_ids):
    return [client.query_molecules(idx)[0] for idx in mol_ids]
    return client.query_molecules(mol_ids)


def dump_qcmol_to_xyz(mols, fname, comment="", idx=None):
    if(fname is None):
        fid = None
    else:
        fid = open(fname,'w') 
    ret = np.array([qcmol2xyz(mol, fid, comment=comment, idx=idx) for mol in mols])

    if(fid is not None):
        fid.close()
    return ret

def save_hessian(hessian, fid, comment="", map=None, idx=None):
    #index_str = "all"
        
    #if(idx is not None):
    #    index_str = str(idx)
    #    hessian = hessian[idx]

    n = hessian.shape[0]//3

    if(idx is None):
        idx = range(n)

    #atom_idx = idx
    #idx = [(3*j + i) for j in idx for i in range(3)]

    #print(hessian.shape)
    #print(idx)
    #if fid is not None:
    #    for line in hessian:
    #        line = [line[3*map[idx[i]] + j] for i in range(len(idx)) for j in range(3)]
    #        fid.write(("{:16.13f} "*len(line) + '\n').format(*(line)))
    hess_mapped = np.empty((3*len(idx), 3*len(idx)))
    # print(hess_mapped.shape, hessian.shape)
    for (ii,jj),(i,j) in zip(itertools.product(range(len(idx)), range(len(idx))),itertools.product(idx, idx)):
        hess_mapped[ii*3:ii*3+3,jj*3:jj*3+3] = hessian[map[idx[ii]]*3:map[idx[ii]]*3+3,map[idx[jj]]*3:map[idx[jj]]*3+3]
    #idx = range(len(idx))
    if(fid is not None):
        fid.write("#{:s}\n{:d}\n".format(comment, len(idx)))
        pairs = sorted(set([tuple(sorted(k)) for k in itertools.product(range(len(idx)), range(len(idx)))]))
        for (i, j) in pairs:
            atomi = i
            atomj = j
            if(atomj < atomi):
                continue
            # print("[", i,j,"] (",atomi, atomj,")", end=" ")
            fid.write(("{:4d} {:4d}"+(" {:12.6f}"*9)+"\n").format(idx[atomi],idx[atomj],*hess_mapped[atomi*3:(atomi*3+3),atomj*3:(atomj*3+3)].reshape(-1)))
        # print("\n")
    return hess_mapped
        

def dump_hessians(hessians, fname, comment="", map=None, idx=None):
    # print("Saving hessians to", fname)
    if(fname is None):
        fid = None
    else:
        fid = open(fname,'w')
    ret =  np.array([save_hessian(hessian_i, fid, comment=comment, map=map, idx=idx) for hessian_i in hessians])
    if(fid is not None):
        fid.close()
    return ret

def qcmol2xyz(mol, fid, comment="", idx=None):
    #fid.write(mol.to_string('xyz'))
    #return
    mol = dict(mol)
    if(idx is None):
        idx = np.arange(len(mol['symbols']))
    syms = [mol['symbols'][i] for i in idx]
    xyzs = (mol['geometry'] * bohr2angstrom)[idx]
    struct = zip(syms, xyzs)
    N = len(syms)
    if(fid is not None):
        fid.write("{:d}\n".format(N))
        fid.write("{:s}\n".format(comment))
        #print("***************")
        #print(xyzs)
        [fid.write(("{:s}"+"{:10.4f}"*3 + "\n").format(s,*xyz)) for s,xyz in struct]
    return xyzs

def get_qcarchive_xyz_grad_ang(mol_smiles, ds, minimum=False, TorsionDrive=True, get_hessians=False, get_gradients=False, get_energy=False):
    # get the torsiondrivedataset for some smiles pattern

    #opts are optimizationdataset
    isTD = (ds.data.collection == "torsiondrivedataset")
    if(isTD):
        main_rec = ds.data.records[mol_smiles]
        td = ds.df.loc[main_rec.name, 'default']
    else:
        td = client.query_procedures(ds.data.records[mol_smiles].object_map["default"])

    mol_ids, info, opt_recs, hessian, opt_ang, ene, srt = get_all_qcmolid_and_grads(td, minimum=minimum, get_energy=get_energy, TorsionDrive=isTD, get_hessians=get_hessians, get_gradients=get_gradients)
    if(minimum):
        mols = mol_ids
    else:
        mols = get_qcmol_from_ids(client, mol_ids)
    return mols, info, opt_recs, hessian, opt_ang, ene, srt


def mm_potential(forcefield, top, xyz, charge=False):

    if(charge):
        system = forcefield.create_openmm_system(top)
    else:
        mols = [Molecule(mol.reference_molecule) for mol in top.topology_molecules]
        for i,_ in enumerate(mols):
            mols[i].partial_charges = simtk.unit.Quantity(np.zeros(mols[i].n_atoms), simtk.unit.elementary_charge)
        system = forcefield.create_openmm_system(top, charge_from_molecules=mols)
                                       
    integrator = openmm.VerletIntegrator(1.0 * simtk.unit.femtoseconds)
    context = openmm.Context(system, integrator)
    context.setPositions(xyz * angstrom2nm)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().in_units_of(simtk.unit.kilocalories_per_mole)
    return energy

def something():
    """ 
    this is supposed to make the index problem go away!!
    Right now a big problem seems to be maps from rd/off and qcarchive indices
    Or, it wasn't a problem until I tried to get energy of my systems
        
    """
    
    return None

def calc_mm_energy(forcefield, top, xyz, component=None, charge=False):

    if(component is None):
        ene = mm_potential(forcefield, top, xyz, charge=charge)
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
        
    ene = mm_potential(modff, top, xyz, charge=charge)
    return ene
    

def calc_vdw_direct(xyz, labels):
    na = len(xyz)
    atoms = range(na)
    ene = 0.0
    r = distance.cdist(xyz,xyz)
    for i in atoms:
        ii = (i,)
        for j in atoms:
            if(j >= i):
                break
            jj = (j,)
            eps = np.sqrt(labels[ii].epsilon * labels[jj].epsilon) / labels[jj].epsilon.unit
            rmin = (labels[ii].rmin_half + labels[jj].rmin_half)/2.0
            rmin = rmin / rmin.unit
            rij = r[i,j]
            a = rmin/rij
            a = a**6
            ene += eps * (a**2 - 2*a)
            #print(i,j,"r", rij, "ene", ene, "rmin", rmin, "eps", eps, "a", a)
    return ene * simtk.unit.kilocalorie_per_mole


def get_frag_matches(frag, ds):
    #targets = [i.name for i in ds.data.records.values()]
    targets = ds.data.records.keys()
    #with open('test2.out', 'w') as fid:
    #    [fid.write(str(i) + '\n') for i in targets]
    #test = ds.df.index
    #with open('test3.out', 'w') as fid:
    #    [fid.write(str(i) + '\n') for i in test]
    #test = [i.name for j,i in ds.data.records.items()]
    #with open('test4.out', 'w') as fid:
    #    [fid.write(str(i) + '\n') for i in test]
    p = FragmentMatcher.FragmentMatcher()
    p.Init(frag)
    matches = {}
    hits = 0
    for mol_smiles in targets:
        smiles_pattern = ds.data.records[mol_smiles].name
        smiles_pattern = ds.data.records[mol_smiles].attributes['canonical_isomeric_explicit_hydrogen_mapped_smiles']
        mol = Chem.MolFromSmiles(smiles_pattern)
        mol = Chem.AddHs(mol)
        if(p.HasMatch(mol)):
            for match in p.GetMatches(mol, uniquify=0): # since oFF is a redundant set
                # deg = mol.GetAtomWithIdx(match[0]).GetDegree()
                if(not (mol_smiles in matches)):
                    matches[mol_smiles] = []
                matches[mol_smiles].append(list(match))
                hits += 1
    print("Found", len(matches.keys()), "molecules with", hits, "hits") 

    return matches


def split_frag_matches(matches, splits):
    split_matches = {}
    for split in splits:
        frags_from_split = {}
        for mol,match in matches.items():
            #frags_from_split[mol] = [list(np.asarray(frag)[split]) for frag in match]
            #split_matches[tuple(split)] = frags_from_split
            split_matches[mol] = {tuple(split): [list(np.asarray(frag)[split]) for frag in match] }
    return split_matches
        


def argsort_labels(l):
    letter = [i[0] for i in l]
    num = [int(i[1:]) for i in l]
    return np.lexsort((num, letter))


def measure_parameters(job_spec, ds,
                       log=sys.stdout, db_out=None, append=False, skip_if_present=False,
                       minimum=True, oFF_label=True, empty=False,
                       out_dir=None, save_mol_xyz=False, save_mol_frag=False,
                       get_gradients=False, get_qm_energy=False, get_mm_energy=False, gen_MM_charge=False,
                       get_hessians=False, save_hessian=False, save_hessian_frag=False,
                       stepTiming=False, totalTiming=False,
                       save_mol_xyz_debug=False, verbose=True):

    valence_params = ["Bonds", "Angles", "ProperTorsions", "ImproperTorsions"]
    measure_vtable = {"Bonds": calc_bond_length,
                      "Angles": calc_angle_degree,
                      "ProperTorsions": calc_proper_torsion,
                      "ImproperTorsions": calc_improper_torsion}
    if(out_dir is None):
        out_dir = os.getcwd()

    #TODO: check each job_spec for valid
    #calcs = parameters
    #matches = match_db[match_key]
    #for atom_req, measure_key in zip([2,3,4,4], valence_params):
    #    if(not (measure_key in calcs)):
    #        continue
    #    for smiles,match in matches.items():
    #        n = len(match)
    #        for n in [m for m in match]: 
    #            if (len(n) < atom_req):
    #                log.write( "{:s} calculation requested but smiles {:s} match {:d} has too few atoms\n\n".format(calc, smiles, n))
    #                return

    ffversion = 'smirnoff99Frosst-1.0.9.offxml'

    db = None
    if (db_out is not None):
        if(os.path.exists(os.path.join(out_dir,db_out)) and append):
            if(db_out.split('.')[-1] == "npz"):
                db = dict(np.load(db_out, allow_pickle=True))
            else:
                with open(os.path.join(out_dir,db_out),'rb') as fid:
                    db = pickle.load(fid)
    if(db is None):
        db = {'frag_info': [],
            'oFF': {"ff_version": ffversion, 'oFF_version': openforcefield.__version__} }
        db['qca_info'] = { "collection_type": ds.data.collection,
                        "collection_name": ds.data.name }

        db['oFF'].setdefault("a0", {})
        db['oFF'].setdefault("b0", {})
        db['oFF'].setdefault("i0", {})
        db['oFF'].setdefault("t0", {})
        db.setdefault("mol_data", {})
        # idea here is to only store unique torsiondrive angles once, use hash to determine which td_ang to use
        db.setdefault("td_ang", {})
    

    if(oFF_label):
        forcefield = ForceField(os.path.join(ff.get_forcefield_dirs_paths()[0],ffversion), disable_version_check=True)


#        for mol_n, (target, idx_list) in enumerate(matches.items()):
    totaltime = time.time()
    log.write("== Dataset {:s}\n".format(ds.data.name))
    for mol_n, (target, config) in enumerate(job_spec.items()):
        #print(target)

        mapped = ds.data.records[target].attributes['canonical_isomeric_explicit_hydrogen_mapped_smiles']
        real_smiles = ds.data.records[target].name

        target_smiles = mapped #ds.data.records[target].name

        if(real_smiles in db['mol_data']):
            mol_data = db['mol_data'][real_smiles]
            if(skip_if_present):
                continue
        else:
            mol_data = {'qca_key': target }
        
        mol = Chem.MolFromSmiles(target_smiles)
        n_heavy = mol.GetNumAtoms()
        mol = Chem.AddHs(mol)
        n_total = mol.GetNumAtoms()
        n_hydrogen = n_total - n_heavy
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=AllChem.ETKDG())
        Chem.rdmolops.AssignStereochemistryFrom3D(mol)
        superdebug = False
        if(superdebug):
            debugfid = open("debug.out", 'w')
        syms = [a.GetSymbol() for a  in mol.GetAtoms()]
        mol2 = mol_to_map_ordered_qcschema(mol, mapped)
        orig_xyz = np.array(mol.GetConformer(0).GetPositions())/bohr2angstrom

        mapped_xyz = np.array(mol2['geometry']).reshape(-1,3)
        mapped_idx = distance.cdist(orig_xyz, mapped_xyz).argmin(axis=1)
        if(superdebug):
            debugfid.write("From smiles built mol\n")
            [debugfid.write(str(i)+'\n') for i in enumerate(orig_xyz)]
            debugfid.write("Index map (rd to qc?)\n")
            [debugfid.write(str(i)+'\n') for i in enumerate(mapped_idx)]
            debugfid.write("From remapped mol from cmiles\n")
            [debugfid.write(str(i)+'\n') for i in enumerate(mapped_xyz)]
            debugfid.write("\n")

        # putting an from the QCMol indices will give you the index in the analyses (e.g. torsiondrive indices)
        # qcmol data is 0-based, so is this map; but note that hessian labels are 1-based

        log.write("Result= {:8d}     Molecule= {:50s} ".format(mol_n, real_smiles, ))
        log.flush()
        result_time=time.time()
        elapsed=time.time()
        try:
            mols, info, opts, hessian, ang, ene, srt = get_qcarchive_xyz_grad_ang(target, ds, \
            minimum=minimum, get_hessians=get_hessians, get_gradients=get_gradients, get_energy=get_qm_energy)
        except (ValidationError, TypeError) as e:
            log.write("ERROR: QCArchive returned nothing\n\n")
            continue
        log.write("\nQCID= {:>10s} QCDataRecord= {:50s}  \n".format(ds.data.records[target].object_map['default'], target))

        minstr = ".min" if minimum else ".all"
        mol_out_fname = os.path.join(out_dir,"mol_"+str(mol_n)+minstr+".xyz") if save_mol_xyz else None
        atoms = dump_qcmol_to_xyz(mols, mol_out_fname, comment=real_smiles, idx=mapped_idx)

        if(superdebug):
            debugfid.write("from qcdata base\n")
            [debugfid.write(str(j[0]) + " " + str(i[0]) + " " + \
                str(i[1]*bohr2angstrom)+'\n') \
                for j in enumerate(mols) \
                for i in enumerate(dict(j[1])['geometry'])]


        if('from_qcmol_to_label_map' not in mol_data):
            mapped_idx_inv = distance.cdist(orig_xyz, mapped_xyz).argmin(axis=0)
            mol_data['from_qcmol_to_label_map'] = mapped_idx_inv
            if(superdebug):
                debugfid.write("Index map (qc to rd?)\n")
                [debugfid.write(str(i)+'\n') for i in enumerate(mapped_idx_inv)]
                debugfid.write("mapped from qcdatabase using mapped_idx_inv\n")
                [debugfid.write(str(j[0]) + " " + str(i[0]) + " " + \
                    str(j[1].geometry[i[1]] *bohr2angstrom)+'\n') \
                    for j in enumerate(mols) \
                    for i in enumerate(mapped_idx_inv)]
        if(superdebug):
            debugfid.write("mapped from qcdatabase using mapped_idx\n")
            [debugfid.write(str(j[0]) + " " + str(i[0]) + " " + \
                str(j[1].geometry[i[1]] *bohr2angstrom)+'\n') \
                for j in enumerate(mols) \
                for i in enumerate(mapped_idx)]

            debugfid.write("first record in atoms (should be same as above\n")
            [debugfid.write(str(j[0]) + " " + str(i[0]) + " " + \
                str(i[1]*bohr2angstrom)+'\n') \
                for j in enumerate(mols) \
                for i in enumerate(atoms[j[0]])]
            debugfid.close()

        queryelapsed=str(timedelta(seconds=time.time() - elapsed))
        if(len(mols) == 0):
            if(stepTiming):
                log.write("\n QueryTime= {:s}\n".format(queryelapsed))
            log.write("ERROR: QCArchive returned nothing\n\n")
            continue
        elapsed=time.time()

        if('info' not in mol_data):
            mol_data['info'] = info

        if('n_hydrogen' not in mol_data):
            mol_data['n_hydrogen'] = n_hydrogen 
        if('n_heavy' not in mol_data):
            mol_data['n_heavy'] = n_heavy


        ang_hash = hash(tuple(ang))
        if(ang_hash not in db["td_ang"]):
            db["td_ang"][ang_hash] = ang
        if('td_ang' not in mol_data):
            mol_data["td_ang"] = ang_hash

        ic_hess = None
        IC = None
        if(hessian is not None):
            if("hessian" not in mol_data):
                mol_data["hessian"] = {}
                mol_data["hessian"]["cartesian"] = {}
            hessian = np.array([hess_i.reshape(len(syms)*3, len(syms)*3) for hess_i in hessian])
            if(save_mol_xyz):
                IC = CoordClass(geometric_mol(mol_out_fname), build=True,
                                connect=connect, addcart=addcart, constraints=Cons,
                                cvals=CVals[0] if CVals is not None else None )
                convert = hartree2kcalmol/bohr2angstrom**2
                ic_hess = CalcInternalHess(orig_xyz[mapped_idx]/bohr2angstrom,
                                        np.array(opts[0].return_result).reshape(-1,3)[mapped_idx].flatten(),
                                        hessian[0]/convert, IC)
                #ic_hess *= (hartree2kcalmol/bohr2angstrom**2)
                hessian_out_fname = os.path.join(out_dir,"mol_"+str(mol_n)+minstr+".hessian.nxmxmx9.dat") \
                    if save_hessian else None
            atom_hessian = dump_hessians(hessian, hessian_out_fname, comment=real_smiles, map=mapped_idx ,idx=None)

        if(save_mol_xyz_debug):
            rdmol_out_fname = os.path.join(out_dir,"rdmol_"+str(mol_n)+minstr+".xyz")
            with open(rdmol_out_fname,'w') as fid:
                qcmol2xyz({'symbols':syms, "geometry":orig_xyz}, fid, comment="rdkit conformer "+ real_smiles)


        if(len(syms) != len(mols[0].symbols)):
            log.write("\nERROR: conformer has different number of atoms than QCMolecule.\n")
            log.write("Likely H assignment incorrect. Check the mol and rdmol xyz files. Skipping.\n")
            continue

        if(get_qm_energy):
            if('energy' not in mol_data):
                mol_data['energy'] = {}
            mol_data['energy']['qm'] = ene


        if(oFF_label):

            mmol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
            top = Topology().from_molecules(mmol)
            #topapp = Topology().from_molecules(mmol)
            # these will use the unmapped indices, which are the indices from the matches
            labels = forcefield.label_molecules(top)[0]

            if(get_mm_energy):
                if('energy' not in mol_data):
                    mol_data['energy'] = {'oFF': {} }            
                elif('oFF' not in mol_data['energy']):
                    mol_data['energy']['oFF'] = {}

                n_qcmol = len(mols)

                #reporter = openmm.app.pdbreporter.PDBReporter("min_ene.pdb", 1, False)
                #systemapp = forcefield.create_openmm_system(topapp)
                #integratorapp = openmm.VerletIntegrator(1.0 * simtk.unit.femtoseconds)
                #sim = openmm.app.Simulation(topapp.to_openmm(), systemapp, integratorapp)
                for qcmol_i,qcmol in enumerate(atoms):
                    #xyz = dict(qcmol)['geometry']
                    #mmol.conformers[0] = qcmol
                    xyz = qcmol 
                    #print("raw0", orig_xyz[0:2], xyz[0:2], labels['vdW'][(0,)], labels['vdW'][(1,)])
                    #print("raw", orig_xyz[0], xyz[0]* bohr2angstrom)
                    #xyz2 = np.array(xyz)[mapped_idx] * bohr2angstrom
                    #print("real", orig_xyz[0], xyz2[0])
                    #xyz = np.array(xyz)[mapped_idx_inv] * bohr2angstrom
                    #print("real", orig_xyz[0], xyz[0])
                    #print("real", orig_xyz[0], mapped_xyz[0])
                    total_ene = calc_mm_energy(forcefield, top, xyz, charge=gen_MM_charge)
                    log.write("  Conformation energy {:4d}/{:4d}\n".format(qcmol_i+1,n_qcmol))
                    log.flush()
                    if(get_qm_energy):
                        ene_str = "{:13.8f} a.u.".format(ene[qcmol_i])
                        log.write("    QM {:18s}= {:s} \n".format("Energy", ene_str))
                        log.flush()

                    ene_name_str = "Energy" if gen_MM_charge else "EnergyNoElec"
                    log.write("    MM {:18s}= {:10.5f} {:s}\n".format(ene_name_str,
                        (total_ene/total_ene.unit ),
                        str(total_ene.unit)))
                    log.flush()

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
                    ene_sum = total_ene
                    #if('direct_vdw' not in mol_data['energy']['oFF']):
                    #    mol_data['energy']['oFF']['direct_vdw'] = []
                    #mol_data['energy']['oFF']['direct_vdw'].append(calc_vdw_direct(xyz, labels['vdW']))
                    #print(mol_data['energy']['oFF']['direct_vdw'])

                    if('epot' not in mol_data['energy']['oFF']):
                        mol_data['energy']['oFF']['epot'] = []
                    mol_data["energy"]['oFF']['epot'].append(total_ene)
                    for component in ['vdW' ] + valence_params:
                        energy_sans_component = calc_mm_energy(forcefield, top, xyz, component=component, charge=gen_MM_charge)
                        energy_component = total_ene - energy_sans_component

                        if(component not in mol_data["energy"]['oFF']):
                            mol_data["energy"]['oFF'][component] = []
                        mol_data["energy"]['oFF'][component].append(energy_component)
                        ene_sum -= energy_component
                        log.write("    MM {:18s}= {:10.5f} {:s}\n".format(component,
                            (energy_component/energy_component.unit ),
                            str(energy_component.unit)))
                        log.flush()
                    if(gen_MM_charge):
                        if('Electrostatics' not in mol_data['energy']['oFF']):
                            mol_data['energy']['oFF']['Electrostatics'] = []
                        mol_data['energy']['oFF']['Electrostatics'].append(ene_sum)
                        log.write("    MM {:18s}= {:10.5f} {:s}\n\n".format("Electrostatics",
                            (ene_sum/ene_sum.unit ),
                            str(ene_sum.unit)))

                    #log.write("    MM {:18s}= {:10.5f} {:s}\n\n".format("Direct vdW",
                    #    (mol_data['energy']['oFF']['direct_vdw'][qcmol_i] / mol_data['energy']['oFF']['direct_vdw'][qcmol_i].unit ),
                    #    str(mol_data['energy']['oFF']['direct_vdw'][qcmol_i].unit)))
                    log.flush()

                log.write("\n  Conformation energy summary\n")
                if(len(ene) > 1):
                    if(get_qm_energy):
                        ene_str = "{:13.8f} +- {:13.8f} a.u.".format(np.mean(ene), np.std(ene))
                        log.write("    QM {:18s}= {:s} \n".format("Energy", ene_str))

                    log.write("    MM {:18s}= {:10.5f} +- {:10.5f} {:s}\n\n".format(ene_name_str,
                        np.mean([i/i.unit for i in mol_data["energy"]['oFF']['epot']]),
                        np.std([i/i.unit for i in mol_data["energy"]['oFF']['epot']]),
                        str(mol_data["energy"]['oFF']['epot'][0].unit)))
                    components_to_print = ['vdW' ] + valence_params
                    if(gen_MM_charge):
                        components_to_print += ['Electrostatics']
                    for component in components_to_print:
                        log.write("    MM {:18s}= {:10.5f} +- {:10.5f} {:s}\n".format(component,
                            np.mean([i/i.unit for i in mol_data["energy"]['oFF'][component]]),
                            np.std([i/i.unit for i in mol_data["energy"]['oFF'][component]]),
                            str(mol_data["energy"]['oFF'][component][0].unit)))
            #for x in labels.items():
            #    print(list(x[1]))

        column_index = {}
        hits = 0

        if(stepTiming and get_mm_energy):
            elapsed=str(timedelta(seconds=time.time() - elapsed))
            log.write("\n    EnergyTime= {:s}\n\n".format(elapsed))
            elapsed=time.time()

        for job in config:
            parameters = job['measure']
            query = job['query']
            #log.write("==== Fragment {:s}\n".format(query))
            db['frag_info'].append({ "query": query, "map": job['splits'], "measurements": parameters, "minimum": minimum })
#    job[mol].append({'query': smiles,
#                     'measure': parameters,
#                     'splits': splits,
#                     'matches': split_matches[mol] })
            for complete_frag in job['matches']:
                for split in job['splits']:
                    #for split in job['splits']:
                    match = [complete_frag[i] for i in split]
                    #print("Working on", match)


                    remapped = mapped_idx[match]
                    # print(mol_n, "\nMATCH", match)

                    #calculate the requested measurements
                    for measure_key in parameters:
                        measure_fn = measure_vtable[measure_key]
                        #if(measure_key == "Bonds" and measure_key in column_index):
                        #    print("COL KEY BEFORE MOD:", column_index[measure_key])
                        if(not (measure_key in mol_data)):
                            mol_data[measure_key] = {}
                            mol_data[measure_key]["indices"] = {}
                            column_index[measure_key] = 0
                        elif(not empty):
                            column_index[measure_key] = max([x['column_idx'] for x in mol_data[measure_key]["indices"].values()]) + 1
                        #if(measure_key == "Bonds"):
                        #    print("COL KEY AFTER MOD:", column_index[measure_key])


                        measure_db, label_db = build_measurement_db(atoms, match, measure_fn, \
                            oFF_label=True, labels=labels, measure_key=measure_key, \
                            improper_flag=("ImproperTorsions" in parameters ), empty=empty)

                        db["oFF"].update(label_db)

                        param_uniq = []
                        index_uniq = []
                        # add measurement to molecules data
                        if ("values" not in mol_data[measure_key]):
                            if(not empty):
                                mol_data[measure_key]["values"] = measure_db["values"]
                            index_uniq = measure_db['indices'].keys()
                        else:
                            for param in measure_db["indices"].keys():
                                #print("Considering", param)
                                n_atoms = len(param)
                                permutations = None
                                if(n_atoms == 2):
                                    if(measure_key != "Bonds"):
                                        continue
                                    permutations = itertools.permutations(param)
                                elif(n_atoms == 3):
                                    if(measure_key != "Angles"):
                                        continue
                                    permutations = [[x[0],param[1],x[1]] for x in itertools.permutations([param[0],param[2]])]
                                elif(n_atoms == 4):
                                    if(measure_key == "ImproperTorsions"):
                                        permutations = [[x[0],param[1],x[1],x[2]] for x in itertools.permutations([param[0]]+list(param[2:]))]
                                    elif(measure_key == "ProperTorsions"):
                                        # cross our fingers that dihedrals are sequential
                                        permutations = [param, param[::-1]]
                                    else:
                                        continue
                                exists = False
                                for order_i in permutations:
                                    candidate = tuple(order_i)
                                    if(candidate in mol_data[measure_key]["indices"] ):
                                        exists=True
                                        break
                                if(not exists):
                                    if(not empty):
                                        param_uniq.append(measure_db["indices"][param]["column_idx"])
                                    #print("Not present", param)
                                    index_uniq.append(param)
                                    #measure_db["indices"][param]["column_idx"] = len(param_uniq) - 1
                                #else:
                                    #print("Already contained, not adding")
                            if(not empty):
                                #for uniq in param_uniq:
                                #print(mol_data[measure_key]["values"].shape, np.atleast_2d(measure_db["values"][:,param_uniq]).shape)
                                mol_data[measure_key]["values"] = np.hstack((mol_data[measure_key]["values"], np.atleast_2d(measure_db["values"][:,param_uniq])))


                            #print("index_uniq is", index_uniq)
                        add_data = 0
                        for index_key in index_uniq:
                            if(index_key[::-1] in mol_data[measure_key]["indices"]):
                                index_key = index_idx[::-1]
                            mol_data[measure_key]["indices"][index_key] = measure_db["indices"][index_key]
                            mol_data[measure_key]["indices"][index_key].setdefault("oFF", None)
                            if(not empty):
                                mol_data[measure_key]["indices"][index_key]["column_idx"] = column_index[measure_key]
                                column_index[measure_key] += 1

                            if(hessian is not None):
                                for (i,j) in itertools.product(*[index_key]*2):
                                    if((i,j) not in mol_data["hessian"]["cartesian"]):
                                        mol_data["hessian"]["cartesian"][(i,j)] = hessian[:,(i*3):(i*3+3),(j*3):(j*3+3)] #.reshape(-1,9)



                                prefix="frag"
                                if(oFF_label and index_key in measure_db["indices"]):
                                    prefix=measure_db["indices"][index_key]['oFF']
                                frag_fname= os.path.join(out_dir,"mol_"+str(mol_n)+"."+prefix+"_"+hyphenate_int_tuple(index_key,1)+minstr+".hessian.nx9.dat") if save_hessian_frag else None
                                #dump_hessians(hessian, frag_fname, comment=str(remapped[data_key]), idx=remapped[data_key])
                                #print("idx_key", index_key, "mapped_idx", mapped_idx, "remapped", remapped, "match", match)
                                dump_hessians(hessian, frag_fname, comment=str(index_key), map=mapped_idx, idx=index_key )
                                #dump_hessians(hessian, frag_fname, comment=str(index_key), idx=index_key)
                                #dump_hessians(hessian, frag_fname, comment=str(remapped), idx=remapped)
                                ##############################################

                            ########################################################################
                            prefix="frag"
                            if(oFF_label):
                                prefix=measure_db["indices"][index_key]['oFF']
                            #frag_fname= "mol_"+str(mol_n)+"."+prefix+"_"+"{:d}".format(match[0])+("-{:d}"*(len(match)-1)).format(*(match[1:]))+minstr+".xyz"
                            frag_fname= os.path.join(out_dir,"mol_"+str(mol_n)+"."+prefix+"_"+hyphenate_int_tuple(index_key,1)+minstr+".xyz") if save_mol_frag else None
                            #dump_qcmol_to_xyz(mols, frag_fname, comment=str(remapped), idx=remapped)
                            dump_qcmol_to_xyz(mols, frag_fname, comment=str(index_key), idx=[mapped_idx[i] for i in index_key])



        hessian_hits = []
        if(oFF_label and (IC is not None)):
            no_hess_to_label_match = set()
            hits = 0
            #print("starting IC labeling..")
            if(coordsys not in mol_data["hessian"]):
                mol_data["hessian"][coordsys] = {}
                mol_data["hessian"][coordsys]['IC'] = IC
                mol_data["hessian"][coordsys]['values'] = {}

            prune_hessian = False
            if(prune_hessian):
                for measure_key in valence_params:
                    if(measure_key not in mol_data):
                        continue
                    for (idx_i,index_key_i) in enumerate(mol_data[measure_key]['indices'].keys()):
                        print("trying", index_key_i, mol_data[measure_key]['indices'][index_key_i]['oFF'], end=" ")
                        hess_key_i = tuple([x+1 for x in index_key_i])
                        exists, i = oFF_key_in_IC_Hessian(hess_key_i, IC.Internals)
                        if(exists):
                            hessian_hits.append(i)
                            hits+=1
                            print("HIT", i, "total", hits)
                            if(index_key_i not in mol_data["hessian"][coordsys]['values']):
                                mol_data["hessian"][coordsys]['values'][index_key_i] = {}
                            mol_data["hessian"][coordsys]['values'][index_key_i][index_key_i] = ic_hess[i, i]
                            for j,Internal in enumerate(IC.Internals):
                                hess_key_j = tuple(str(Internal).split()[1].split("-"))
                                index_key_j = tuple([int(x)-1 for x in hess_key_j])
                                mol_data["hessian"][coordsys]['values'][index_key_i][index_key_j] = ic_hess[i, j]
                                if(index_key_j not in mol_data["hessian"][coordsys]['values']):
                                    mol_data["hessian"][coordsys]['values'][index_key_j] = {}
                                    mol_data["hessian"][coordsys]['values'][index_key_j][index_key_j] = ic_hess[j, j]
                                mol_data["hessian"][coordsys]['values'][index_key_j][index_key_i] = ic_hess[j, i]
                        else:
                            #print("MISS", i)
                            no_hess_to_label_match.add(hess_key_i)

            else:
                for i,Internal in enumerate(IC.Internals):
                    hess_key_i = tuple(str(Internal).split()[1].split("-"))
                    index_key_i = tuple([int(x)-1 for x in hess_key_i])
                    hessian_hits.append(i)
                    if(index_key_i not in mol_data["hessian"][coordsys]['values']):
                        mol_data["hessian"][coordsys]['values'][index_key_i] = {}
                    for j,Internal in enumerate(IC.Internals):
                        hess_key_j = tuple(str(Internal).split()[1].split("-"))
                        index_key_j = tuple([int(x)-1 for x in hess_key_j])
                        mol_data["hessian"][coordsys]['values'][index_key_i][index_key_j] = ic_hess[i, j]
                        if(index_key_j not in mol_data["hessian"][coordsys]['values']):
                            mol_data["hessian"][coordsys]['values'][index_key_j] = {}
                            mol_data["hessian"][coordsys]['values'][index_key_j][index_key_j] = ic_hess[j, j]
                        mol_data["hessian"][coordsys]['values'][index_key_j][index_key_i] = ic_hess[j, i]

        # print the parameter search results
        if(oFF_label):
            total_keys_with_labels = 0
            total_keys_any_labels = 0
            for measure_key in valence_params:
                counts = {}
                if(measure_key not in mol_data):
                    continue
                all_params = [param['oFF'] for param in list(mol_data[measure_key]['indices'].values())]
                for param in all_params:
                    if(param is None):
                        param = "None"
                    counts.setdefault(param, 0)
                    counts[param] += 1
                    if(param[1] != "0"):
                        total_keys_with_labels += 1
                    total_keys_any_labels += 1

                found_labels = list(counts.keys())
                label_sort_idx = argsort_labels(found_labels)
                label_str = ("{:3d} {:3s} "*len(counts.keys())).format(*flatten_list([(counts[found_labels[i]],found_labels[i]) for i in label_sort_idx]))
                log.write("    {:18s}= {:3d} | {:s}\n".format(measure_key,
                    sum([val for val in counts.values()]), label_str))

            all_labels = 0
            for x in valence_params:
                if(x in labels):
                    all_labels += sum([1 for l in labels[x]])
            log.write("\n    oFF label coverage:    {:8d}/{:8d}\n".format(total_keys_with_labels, all_labels))
            if( total_keys_with_labels < all_labels):
                if(verbose):
                    log.write("    oFF terms not mapped:\n")
                    for param in valence_params:
                        for term in labels[param]:
                            if(param in mol_data):
                                #print(term, term in mol_data[param]["indices"].keys() )
                                #print(term[::-1], term[::-1] in mol_data[param]["indices"].keys() )
                                if(term in mol_data[param]["indices"]):
                                    continue
                                elif(term[::-1] in mol_data[param]["indices"]):
                                    continue
                            term_str=hyphenate_int_tuple(term, 1)
                            log.write("        {:5s} {:s}\n".format(param, term_str)) 

            if(IC is not None):
                    log.write("\n    oFF map to IC Hessian: {:8d}/{:8d}\n".format(len(hessian_hits), len(IC.Internals)))
                    if(len(hessian_hits) < len(IC.Internals)):
                        #print("hessian_hits", sorted(hessian_hits))
                        hessian_misses = [x for x in range(len(IC.Internals)) if x not in hessian_hits]
                        #print("hessian_misses", hessian_misses)
                        #[print(str(internal)) for internal in IC.Internals]
                        log.write("    IC Hessian terms not mapped: {:d}\n".format(len(hessian_misses)))
                        if(verbose):
                            [log.write("        {:s}\n".format(str(IC.Internals[term]))) for term in hessian_misses]

            log.write("\n")

        # add this mol to the main db and save to disk               
        db["mol_data"][real_smiles] = mol_data
        if(db_out is not None):
            if(db_out.split('.')[-1] == "npz"):
                np.savez(os.path.join(out_dir,db_out), data=np.arange(items), **db)
            else:
                with open(os.path.join(out_dir,db_out),'wb') as fid:
                    pickle.dump(db, fid)
        elapsed=str(timedelta(seconds=time.time() - elapsed))
        result_time = str(timedelta(seconds=time.time() - result_time))
        if(stepTiming):
            log.write("    QueryTime=    {:s}\n".format(queryelapsed))
            log.write("    AnalysisTime= {:s}\n".format(elapsed))
            log.write("    ResultTime=   {:s}\n\n".format(result_time))
        else:
            log.write("\n\n")
    
    totaltime=str(timedelta(seconds=time.time() - totaltime))
    if(totalTiming):
        log.write("TotalTime= {:s}\n".format(totaltime))
    return db




def calc_bond_length(atoms, idx):
    """calculates distance from first atom to remaining atoms"""
    return np.linalg.norm(atoms[:,idx[1],:] - atoms[:,idx[0],:], axis=1)

def calc_angle_degree(atoms, idx):
    """calculates angle between origin and consecutive atom pairs"""
    mags = np.linalg.norm(atoms[:,[idx[0],idx[2]],:] - atoms[:,idx[1],:][:,np.newaxis,:], axis=2)
    atoms_trans = atoms - atoms[:,idx[1],:][:,np.newaxis,:]
    unit = atoms_trans[:,[idx[0],idx[2]],:] / mags[:,:,np.newaxis]
    costheta = (unit[:,0,:] * unit[:,1,:]).sum(axis=1)
    np.clip(costheta, -1.0, 1.0, out=costheta)
    return np.arccos(costheta)*180/np.pi

def calc_proper_torsion(atoms, idx):
    """calculates proper torsion of [i, j, k, l]"""
    noncenter = [idx[0]]+idx[2:]
    mags = np.linalg.norm(atoms[:,noncenter,:] - atoms[:,idx[1],:][:,np.newaxis,:], axis=2)
    atoms_trans = atoms - atoms[:,idx[1],:][:,np.newaxis,:]
    unit = atoms_trans[:,noncenter,:] / mags[:,:,np.newaxis]
    #these are all Nx3
    v0 = -unit[:,0,:]
    v1 = unit[:,1,:]
    v2 = unit[:,2,:]-unit[:,1,:]

    w1 = np.cross(v0,v1)
    w2 = np.cross(v1,v2)

    w1_mag = np.linalg.norm(w1,axis=1)
    w2_mag = np.linalg.norm(w2,axis=1)

    mask = (w1_mag * w2_mag) > 0
    # should be Nx1 costhetas
    costheta = np.ones((atoms.shape[0]))
    costheta[mask]= (w1[mask] * w2[mask]).sum(axis=1) / (w1_mag[mask]*w2_mag[mask])
    np.clip(costheta, -1.0, 1.0, out=costheta)

    theta = np.arccos(costheta)*180/np.pi

    #distance = np.zeros((atoms.shape[0]))
    #distance[mask] = ((w2[mask]*v0[mask]).sum(axis=1)/w2_mag[mask])
    ##theta[distance > 0] = 180 - theta[distance > 0]
    theta[np.abs(theta) >= 180] %= 180.0
    return theta

def calc_improper_torsion(atoms, idx, match_geometric=True):
    """calculates improper torsion of [i, center, j, k]"""
    noncenter = [idx[0]]+idx[2:]
    mags = np.linalg.norm(atoms[:,noncenter,:] - atoms[:,idx[1],:][:,np.newaxis,:], axis=2)
    atoms_trans = atoms - atoms[:,idx[1],:][:,np.newaxis,:]
    unit = atoms_trans[:,noncenter,:] / mags[:,:,np.newaxis]
    #these are all Nx3
    v0 = -unit[:,0,:]
    v1 = unit[:,1,:]-unit[:,0,:]
    v2 = unit[:,1,:]-unit[:,2,:]

    w1 = np.cross(v0,v1)
    w2 = np.cross(v1,v2)

    w1_mag = np.linalg.norm(w1,axis=1)
    w2_mag = np.linalg.norm(w2,axis=1)

    mask = (w1_mag * w2_mag) > 0
    # should be Nx1 costhetas
    costheta = np.ones((atoms.shape[0]))
    costheta[mask]= (w1[mask] * w2[mask]).sum(axis=1) / (w1_mag[mask]*w2_mag[mask])

    theta = np.arccos(costheta)*180/np.pi

    distance = np.zeros((atoms.shape[0]))
    distance[mask] = ((w2[mask]*v0[mask]).sum(axis=1)/w2_mag[mask])
    #theta[distance > 0] = 180 - theta[distance > 0]
    if(match_geometric):
        theta[distance < 0] *= -1
        
    return theta
    
def build_measurement_db(atoms, match, measure_fn, labels=None, oFF_label=False, measure_key=None, improper_flag=False, empty=False):
    label_db = {}
    measure_db = {"indices": {}} 

    # this is how many atoms per measurement (2=bond, 3=angle)
    width = 0
    if(measure_key == "Bonds"):
        width = 2
    elif(measure_key == "Angles"):
        width = 3
    elif(measure_key == "ImproperTorsions"):
        width = 4
    elif(measure_key == "ProperTorsions"):
        width = 4

    if(width < 1):
        print("ERROR: width of key is", width)

    #find the pattern to scan for the type, based on the number of atoms
    #           1    2    3    4
    # bond      X    1    2    3
    # angle     X    X    1    2
    # torsion   X    X    X    1
    # improper  X    X    X    1

    # if matches are len 4, then check if improper_flag set
    #    yes means order is [i, center, j, k] so do distances of ci, cj, ck
    # if flag not set
    #    order is [i,j,k,l] so do distances ij, jk, kl

    sz = len(match)
    seq = []
    if(sz == 2):
        seq = [[0,1]]
    elif(sz == 3):
        if(measure_key == "Bonds"):
            seq = [[0,1][1,2]]
        if(measure_key == "Angles"):
            seq = [[0,1,2]]
    elif(sz >= 4):
        if(improper_flag):
            if(measure_key == "Bonds"):
                seq = [[0,1],[1,2],[1,3]]
            elif(measure_key == "Angles"):
                seq = [[0,1,2],[0,1,3],[2,1,3]]
            else:
                seq = [[0,1,2,3]]
        else:
            if(measure_key == "Bonds"):
                seq = [[0,1],[1,2],[2,3]]
            elif(measure_key == "Angles"):
                seq = [[0,1,2],[1,2,3]]
            else:
                seq = [[0,1,2,3]]
    
    for col, idx in enumerate(seq):
        # key is a tuple of e.g. pairs of indices for bonds (triplets for angles)
        mapped = [match[i] for i in idx]
        if(measure_key == "Bonds" ):
            data_key = tuple(sorted(mapped))
        else:
            data_key = tuple(mapped)

        measure_db["indices"].setdefault(data_key, {})

        #print("match", match, "idx", idx, "data_key:", data_key)
        if(not empty):
            #print("atoms", atoms)
            #print("position\n", atoms[:,data_key,:])
            values = measure_fn(atoms, list(data_key))[:,np.newaxis]
            #print("values:", values)
            measure_db["indices"][data_key]["column_idx"] = col
            col += 1

            if(not ("values" in measure_db)):
                measure_db["values"] = values
            else:
                measure_db["values"] = np.hstack((measure_db["values"], values))

        if(oFF_label):
            if(data_key in labels[measure_key]):
                label_dict = labels[measure_key][data_key].to_dict()
                lid = label_dict['id']
                label_db[lid] = label_dict
                measure_db["indices"][data_key]["oFF"] = lid
            else:
                #print("MISS:", data_key, "not in", measure_key, list(labels[measure_key].keys()))
                if(measure_key == "Bonds"):
                    measure_db["indices"][data_key]["oFF"] = 'b0'
                if(measure_key == "Angles"):
                    measure_db["indices"][data_key]["oFF"] = 'a0'
                if(measure_key == "ProperTorsions"):
                    measure_db["indices"][data_key]["oFF"] = 't0'
                if(measure_key == "ImproperTorsions"):
                    measure_db["indices"][data_key]["oFF"] = 'i0'

    if(oFF_label):
        return measure_db, label_db
    else:
        return measure_db
    


#example_improper = np.array([[[-1,-1,0],[-1,0,0],[0,0,0],[0,1,0]]])
#example_improper = np.append(example_improper, [[[1,0,0],[0,0,0],[0,1,0],[0,0,1]]], axis=0)

#example_improper = np.array([[[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]])
#idx = [0,1,2,3]
#print(calc_improper_torsion(example_improper, idx, match_geometric=True))
#print(calc_proper_torsion(example_improper, idx))


#ds = client.get_collection("TorsionDriveDataset", # 
#    "openff Group1 Torsions")
#print("Caching database...")
#try:
#    load = ds.status(['default'], collapse=False, status="COMPLETE")
#except KeyError:
#    load = ds.df.head(), ds.data.records


dsopt = client.get_collection('OptimizationDataset', 'OpenFF Optimization Set 1')
print("Caching database...")
try:
    load = dsopt.status(['default'], collapse=False, status="COMPLETE")
except KeyError:
    load = dsopt.df.head(), dsopt.data.records


#client.list_collections()

#dsveh = client.get_collection('OptimizationDataset', 'OpenFF VEHICLe Set 1')
#print("Caching database...")
#try:
#    load = dsveh.status(['default'], collapse=False, status="COMPLETE")
#except KeyError:
#    print("Load failed.. trying plan B...")
#    load = dsveh.df.head(), dsveh.data.records


#smiles=  '[#7X3](-[*])(-[*])(-[#6X3]=O)'
#smiles=  '[#7X3:2](-[*:1])(-[*:4])(-[#6X3:3]=O)'
#if(1):
#    append=False
#    smiles= '*~*~*~*'
#    parameters = ["Bonds", "Angles", "ProperTorsions"]
#else:
#    append=True
#    smiles= '*~*(~*)~*'
#    parameters = ["Bonds", "Angles", "ImproperTorsions"]
#
#target_ds = dsopt
##smiles= '**'
##smiles= '[*:1]~[#6X3:2](~[*:3])~[*:4]'
##smiles= '[*:1]-[#7X3:2](-[*:4])-[#6X3:3]=O'
#
#splits = [[0,1,2,3]]
##splits = [[0,1]]
#split_key = tuple(splits[0])
#matches = get_frag_matches(smiles, target_ds)
#split_matches = split_frag_matches(matches, splits)
#job = [split_matches]
#print(list(split_matches.items())[0])



          



append=False
target_ds = dsopt
    
smiles= '*~*~*~*'
parameters = ["Bonds", "Angles", "ProperTorsions"]
splits = [[0,1,2,3]]
matches = get_frag_matches(smiles, target_ds)
split_matches = split_frag_matches(matches, splits)

job = {}
for mol in matches:
    if(mol not in job):
        job[mol] = []
    job[mol].append({'query': smiles,
                     'measure': parameters,
                     'splits': splits,
                     'matches': matches[mol] })

smiles= '*~*(~*)~*'
parameters = ["Bonds", "Angles", "ImproperTorsions"]
splits = [[0,1,2,3]]
matches = get_frag_matches(smiles, target_ds)
split_matches = split_frag_matches(matches, splits)

for mol in matches:
    if(mol not in job):
        job[mol] = []
    job[mol].append({'query': smiles,
                     'measure': parameters,
                     'splits': splits,
                     'matches': matches[mol] })

#first = list(job.keys())[0]
#job = {first: job[first]}


out_dir="test/test_new_hess_save"
db_out = 'db.pickle'
append=False
#if(1):
#    fid = sys.stdout
with open(os.path.join(out_dir,"test.out"), 'a' if append else 'w') as fid:
    #db = measure_parameters(smiles, split_matches, split_key, parameters, target_ds,
    db = measure_parameters(job, target_ds,
                       log=fid, db_out=db_out, append=append, skip_if_present=True,
                       minimum=True, oFF_label=True, empty=False,
                       out_dir=out_dir, save_mol_xyz=True, save_mol_frag=False,
                       get_gradients=True, get_qm_energy=True, get_mm_energy=False, gen_MM_charge=False,
                       get_hessians=True, save_hessian=True, save_hessian_frag=False,
                       stepTiming=True, totalTiming=True, verbose=True,
                       save_mol_xyz_debug=False)



