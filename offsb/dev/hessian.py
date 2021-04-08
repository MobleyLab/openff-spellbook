#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from geometric.internal import *
from geometric.molecule import Molecule as geometric_mol
from scipy.linalg import svd
import sys

gpermol2au = 1836
au2amu = 1/1836.
kcal2hartree = 1/627.5096080306
ang2bohr = 1/0.529177249
bohr2angstrom = 0.529177249
hartree2kcalmol = 627.5096080306
# eig2wavenumber = 1.0 /(2*np.pi * 137 * 5.29e-9) * np.sqrt(au2amu)
eig2wavenumber = 33.3564095 / (2*np.pi) 
eig2thz = eig2wavenumber * 2.99792458e10 / 1e12 
conversion = eig2wavenumber

def load_xyz(fname):

    xyzdat = np.genfromtxt(fname, skip_header=2, usecols=range(4), dtype=('U2', 'f8', 'f8', 'f8' ))
    syms = [x[0] for x in xyzdat]
    xyz  = np.array([list(x)[1:] for x in xyzdat])
    return syms, xyz

def converteig( eigs):
    return np.sign( eigs) * np.sqrt( np.abs( eigs))*conversion

# connect means whether to remove TR
#    True -> remove TR
#    False -> keep TR
CoordSysDict = {'cart':(CartesianCoordinates, False, False),
            'prim':(PrimitiveInternalCoordinates, False, False),
            'dlc':(DelocalizedInternalCoordinates, True, False),
            'hdlc':(DelocalizedInternalCoordinates, False, True),
            'tric':(DelocalizedInternalCoordinates, True, False)}
coordsys = 'prim' # kwargs.get('coordsys', 'tric')
CVals = None
CoordClass, connect, addcart = CoordSysDict[coordsys.lower()]
Cons = None

swap_internals = not connect
debug = False

def printortho( mat):
    if not debug:
       return 
    for i in mat:
        for x in i:
            x = abs(x)
            if x > 1e-8:
                print("{:3.1f}".format(x), end="")
            else:
                print("Y", end="")
        print()

def printmat( mat, dec=4):
    if not debug:
       return 
    for row in mat:
        for val in row:
            print(("{:"+str(dec+3)+"."+str(dec)+"f} ").format( val), end="")
        print()

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
            candidate = hyphenate_int_tuple(order_i, offset=1)
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

#mass_table = {'C': 12.011, 'H': 1.008, 'N': 14.007, 'O': 15.999, 'P': 30.973, \
#    'S': 32.06, 'F': 18.998, "CL": 35.45, "B": 10.81 }
mass_table = {'C': 12.000, 'H': 1.000, 'N': 14.000, 'O': 16.0, 'P': 31.00, \
    'S': 32.00, 'F': 19.0, "Cl": 35, "B": 11.00 }
#mass_table = {'C': 12.000, 'H': 1.007825032230, 'N': 14.003074004430, 'O': 15.994914619570, 'P': 31.00, \
#    'S': 32.00, 'F': 19.0, "CL": 35, "B": 11.00 }
mass_table_unit = {'C': 1.000, 'H': 1.000, 'N': 1.000, 'O': 1.0, 'P': 1.00, \
    'S': 1.00, 'F': 1.0, "Cl": 1.0, "B": 1.00 }

#mass_table = mass_table_unit


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)


def gs2(X, row_vecs=True, norm=True):
    if not row_vecs:
        X = X.T
    Y = X[0:6][:].copy()
    if debug:
        print("gs2 Y.shape")
        print(Y.shape)
    for i in range(6, X.shape[1]):
        v = np.zeros(X.shape[1])
        v[i] = 1.0
        proj = np.array([np.dot(v,Y[b])*Y[b] for b in range(i)]).sum(axis=0)
            #proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, v - proj))
        if norm:
            Y[-1] /= np.linalg.norm(Y[-1]) 
        #Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T
def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T


def save_xyz(syms, xyz, outfd=sys.stdout, comment=""):
    outfd.write(str(len(syms)) + "\n")
    outfd.write(comment + "\n")
    fmt = "{:2s}     {:16.13f} {:16.13f} {:16.13f}\n"
    [outfd.write(fmt.format(sym, *pos)) for (sym, pos) in zip(syms, xyz)]


def mode_to_xyz(syms, node, mode, outfd=sys.stdout, comment="", norm=True, magnitude=10.0):

    # from Lee-Ping Wangs version
    if(norm):
        mode = mode / np.linalg.norm(mode)/np.sqrt(mode.shape[0])
    spac = np.linspace(0, 1, 21)

    # one complete period (up, down, down, up)
    disp = magnitude*np.concatenate((spac, spac[::-1][1:], -1*spac[1:], -1*spac[::-1][1:-1]))

    m = np.array([mass_table[sym] for sym in syms])[:,np.newaxis]
    mode = mode * bohr2angstrom#  * np.sqrt(m)
    weight = np.array([mass_table[sym] for sym in syms])
    #weight = np.ones(len(syms))
    weight /= weight.sum()
    nodeCOM =  (node * weight[:,np.newaxis]).sum(axis=0)
    for dx in disp:
        save_xyz(syms, node - nodeCOM + mode*dx, outfd=outfd, comment=comment)

def mode_COM_to_xyz(syms, node, mode, outfd=sys.stdout, comment="", norm=True, magnitude=0.1):

    # from Lee-Ping Wangs version
    if(norm):
        mode = mode / np.linalg.norm(mode)/np.sqrt(mode.shape[0])
    spac = np.linspace(0, 1, 21)

    # one complete period (up, down, down, up)
    disp = magnitude*np.concatenate((spac, spac[::-1][1:], -1*spac[1:], -1*spac[::-1][1:-1]))
    

    weight = np.array([mass_table[sym] for sym in syms])
    #weight = np.ones(len(syms))
    weight /= weight.sum()
    nodeCOM =  np.atleast_2d((node + node.mean(axis=0) * weight[:,np.newaxis]).sum(axis=0))
    modeCOM =  np.atleast_2d((mode * (weight[:,np.newaxis])).sum(axis=0))

    syms = ['X']

    for dx in disp:
        save_xyz(syms, nodeCOM + modeCOM*dx, outfd=outfd, comment=comment)


def angular_momentum( xyz, mass, mode):
    COM = center_of_mass( xyz, mass)
    xyz = xyz - COM
    #print("angular_momentum: COM is ", center_of_mass( xyz, mass))
    L = np.cross( mass.reshape( -1, 1) * mode.reshape( -1, 3), xyz, axis=1)
    # print("angular_momentum: L is shape", L.shape)
    return np.linalg.norm(L.sum( axis=0))
    return L.sum( axis=0)


def center_of_mass(xyz, mass):
    mass = mass / mass.sum()
    COM = ( xyz * mass[ :, np.newaxis]).sum( axis=0)
    return COM


def inertia_tensor2(xyz, mass):
    I = np.zeros((3, 3))
    for i, xi in enumerate(xyz):
        I += mass[ i]*( np.dot( xi,xi) * np.eye( 3) - np.outer( xi, xi))
    if debug:
        print("I:")
        print(I)
    return I


def inertia_tensor(xyz, mass):
    #return inertia_tensor2(xyz,mass)
    I = np.zeros((3,3))
    COM = center_of_mass(xyz, mass)
    xyz = xyz - COM
    for (x, y, z), m in zip(xyz, mass):
        I[0,0] += m*(y*y + z*z)
        I[0,1] -= m*(x*y)
        I[0,2] -= m*(x*z)
        I[1,1] += m*(x*x + z*z)
        I[1,2] -= m*(y*z)
        I[2,2] += m*(x*x + y*y)

    I[1,0] = I[0,1]
    I[2,0] = I[0,2]
    I[2,1] = I[1,2]
    if debug:
        print("I:")
        print(I)
    return I


def trans_rot_modes(xyz, mass, X, rot=True):
    #X = np.eye(3)
    if rot:
        D = np.zeros((6, np.product(xyz.shape)))
    else:
        D = np.zeros((3, np.product(xyz.shape)))
    na, nd = xyz.shape
    if debug:
        print(D.shape, xyz.shape)
    sqrt_mass = np.sqrt(mass)
    xyz = xyz - center_of_mass(xyz, mass)
    if debug:
        print(np.tile([1, 0, 0], na).shape)
        print("sqrt_mass")
        print(sqrt_mass)
    D[0][:] = np.tile([1, 0, 0], na) * np.repeat(sqrt_mass, nd)
    D[1][:] = np.tile([0, 1, 0], na) * np.repeat(sqrt_mass, nd)
    D[2][:] = np.tile([0, 0, 1], na) * np.repeat(sqrt_mass, nd)
    if debug:
        print("D[2]")
        print(D[2])
    #P = np.array([np.dot(xyz, p_ax[i]) for i in range(nd)]).reshape(nd,na)
    P = np.dot(xyz, X)
    x,y,z = 0,1,2
    if debug:
        print("P.shape")
        print(P.shape)

    if not rot:
        return D

    D[3][:] = np.array([(P[i,y]*X[j,z] - P[i,z]*X[j,y]) * sqrt_mass[i] \
            for i in range(na) for j in range(nd)])
    D[4][:] = np.array([(P[i,z]*X[j,x] - P[i,x]*X[j,z]) * sqrt_mass[i] \
            for i in range(na) for j in range(nd)])
    D[5][:] = np.array([(P[i,x]*X[j,y] - P[i,y]*X[j,x]) * sqrt_mass[i] \
            for i in range(na) for j in range(nd)])

    if debug:
        print("TR_MODES")
        for row in np.dot(D,D.T):
            for val in row:
                print("{:11.8f} ".format( val), end="")
            print()

        print("D[3,:15", D[3][:15])
        print("D SHAPE", D.shape)

    #dot = np.dot(xyz, X)
    #print("dot.shape")
    #print(dot.shape)
    #for i in range(3):
    #    D[i+3,:] = np.concatenate([np.cross(dot, X[j])[:,i] for j in range(3)])
    #D[3:6] = np.cross(np.dot(P,X),np.dot(P,X)).reshape(3,-1)
    #D[3] /= np.linalg.norm(D[3])
    #D[4] /= np.linalg.norm(D[3])
    #D[5] /= np.linalg.norm(D[3])
    return D


def load_xyz_hessian_from_old( d, mol_number):

    mol_key = list(d['mol_data'].keys())[mol_number]
    print( "RECORD", mol_number, "=>", mol_key)
    mol_name = "mol_" + str(mol_number)
    mol_xyz_fname = mol_name + ".min.xyz"
    mol_hess_fname = mol_name + ".min.hessian.nxmxmx9.dat"
    #mol_hess_fname = '../psi4/gp2/psi4/hessian_psi4.out'
    #IC = d['mol_data'][mol_key]['hessian']['prim']['IC']
    IC = CoordClass(geometric_mol(mol_xyz_fname), build=True,
                            connect=connect, addcart=addcart, constraints=Cons,
                            cvals=CVals[0] if CVals is not None else None )

    xyzdat = np.genfromtxt(mol_xyz_fname, skip_header=2, usecols=range(4), dtype=('U2', 'f8', 'f8', 'f8' ))
    syms = [x[0] for x in xyzdat]
    xyz  = np.array([list(x)[1:] for x in xyzdat])
    weight = np.array([mass_table[sym] for sym in syms])
    #weight = np.ones(len(syms))
    weight /= weight.sum()
    nodeCOM =  (xyz * weight[:,np.newaxis]).sum(axis=0)
    #xyz -= nodeCOM
    xyzdat = None
    #syms = np.genfromtxt(mol_xyz_fname, skip_header=2, usecols=(0,), dtype='U' )
    #xyz = np.genfromtxt(mol_xyz_fname, skip_header=2, usecols=(1,2,3))
    na = len(syms)
    
    #hessmat = np.genfromtxt(mol_hess_fname)
    #hess = hessmat
    #print("Loading", mol_hess_fname)
    hessmat = np.genfromtxt(mol_hess_fname,skip_header=2)
    hess = np.zeros((na*3,na*3))
    for line in hessmat:
        row = int(line[0])
        col = int(line[1])
        hess[3*row:3*row+3,3*col:3*col+3] = line[2:].reshape(3,3)
        hess[3*col:3*col+3,3*row:3*row+3] = line[2:].reshape(3,3).T
    convert = kcal2hartree / ang2bohr**2
    hess *= convert
    convert = 1
    #with open("hessian_full.dat", 'w') as fid:
    #    [fid.write(("{:16.13f} "*len(line) + '\n').format(*(line * convert ))) for line in hess]


    return hess, xyz, syms


def hessian_modes( hess, xyz, mass, mol_number, remove=0, stdtr=False):

    # this assumes hessian in kcalmol/A^2
    xyz = xyz * ang2bohr
    convert =  1
    mass_mat = (np.dot(np.sqrt(mass).reshape(-1,1), np.sqrt(mass).reshape(1,-1)))
    mass_per_atom = mass[:,0]
    # hess = hess * hartree2kcalmol / bohr2angstrom**2
    hess = hess / mass_mat * 937583.07
    # begin the analysis
    E = np.linalg.eigh(hess * convert)# 
    debug = False
    if debug:
        print("freqs before removing trans/rot")
        print(np.sign(E[0])*np.sqrt(np.abs(E[0]))*conversion)
    esrt = E[0].argsort()

    mol_name = "mol_" + str(mol_number)
    verbose = False
    if verbose:
        for i, mode in enumerate(E[1]):
            with open(mol_name+".mode_"+str(i)+".xyz", 'w') as fid:
                mode_to_xyz(syms, xyz, mode.reshape(-1, 3), outfd=fid, 
                        comment=mol_name + " mode " + str(i))
        for i, mode in enumerate(E[1]):
            with open(mol_name+".COM.mode_"+str(i)+".xyz", 'w') as fid:
                mode_COM_to_xyz(syms, xyz, mode.reshape(-1, 3), outfd=fid, 
                        comment=mol_name + " COMmode " + str(i))

    if debug:
        print("INERTIA: xyz[:5] is", xyz[:5])
    I = inertia_tensor(xyz, mass_per_atom)
    if debug:
        print("INERTIA: xyz[:5] is", xyz[:5])
    w,X = np.linalg.eigh(I)
    if debug:
        print("Interia eigenvals:", w)

    M = np.diag(1.0/(np.repeat(np.sqrt(mass_per_atom), 3)))
    #M = np.diag((np.repeat(np.sqrt(mass_per_atom), 3)))


    if debug:
        print(hess.shape, mass_mat.shape)

    # this is the transformation matrix 
    D = trans_rot_modes(xyz, mass_per_atom, X, rot=True)
    if debug:
        print("D dots:")
        for i in range(D.shape[0]):
            print(np.linalg.norm(D[i]))
    D /= np.linalg.norm(D, axis=1)[:,np.newaxis]
    if debug:
        print("D dots after normalization:")
        for i in range(D.shape[0]):
            print(np.linalg.norm(D[i]))
    #D /= np.linalg.norm(D, axis=0)
    #for i,d in enumerate(D):
    #    norm = np.dot(d,d.T)
    #    if(norm > 1e-7):
    #        D[i] /= np.sqrt(norm)
    D = D.T
    TR = D
    if debug:
        print("TR shape")
        print(TR.shape)
        print("D trans/rot dots:")
        print(D)
        
        print("D[:,0] pre qr")
        print(D[:,0])
    testmat = (np.dot(np.dot(TR.T, hess), TR))
    if debug:
        print("TESTMAT TR' * H * TR")
        for row in testmat:
            for val in row:
                print("{:11.8f} ".format( val), end="")
            print()
    testmat = np.dot( TR.T, TR)
    if debug:
        print("TR PRE QR DOTS")
        for row in testmat:
            for val in row:
                print("{:11.8f} ".format( val), end="")
            print()
    D, _ = np.linalg.qr( D, mode='complete')
    Q, _ = np.linalg.qr( D[:,:6], mode='complete')

    for i in range(D.shape[0]):
        if np.linalg.norm( D[:,i]) < 0.0:
            D[:, i] *= -1.0

    if debug:
        print("Q VS D DOT")
        printortho( np.dot( Q.T, D))
        printmat( np.dot( Q.T, D), 2)
    #D /= np.linalg.norm( D, axis=0)
    TR = D[ :, :6]
    testmat = np.dot( TR.T, TR)
    if debug:
        print("TR AFTER QR DOTS")
        for row in testmat:
            for val in row:
                print("{:11.8f} ".format( val), end="")
            print()
    testmat = (np.dot(np.dot(TR.T, hess), TR))
    if debug:
        print("TESTMAT")
        for row in testmat:
            for val in row:
                print("{:11.8f} ".format( val), end="")
            print()
    #D /= np.linalg.norm(D, axis=1)[:,np.newaxis]
    #D = D.T
    #D = np.array(gram_schmidt(D.T))
    #D = gs2(D.T).T
    if debug:
        print("D shape immediately post qr")
        print(D.shape)
    #D *= -1
    #D = D[6:]
    #Dp = np.eye(D.shape[1])
    #Dp[:6][:] = D
    #D = D.T
    q = np.repeat(np.sqrt(mass_per_atom), xyz.shape[1])
    if debug:
        print("Q (sqrt(m)):")
        print(q)
        print("Dot first 6 with qr first 6")
        for i in range(6):
            x = D[:,i]
            y = TR[:,i]
            print(np.dot(x/np.linalg.norm(x),y/np.linalg.norm(y)))

        print("np.linalg.norm(D,axis=0)")
        print(np.linalg.norm(D,axis=0))
        print("np.linalg.norm(D,axis=1)")
        print(np.linalg.norm(D,axis=1))

        print("D[:,0] post qr")
        print(D[:,0])

        print("D.shape post qr")
        print(D.shape)
        print("full hess is")
        print(hess.shape)

    #if D[:,0].sum() < 0:
    #    D *= -1
    freq_ic,L = np.linalg.eigh(np.dot(np.dot(D.T, hess), D))
    #testmat = np.dot( np.dot( L.T, np.dot( np.dot( D.T, hess / mass_mat), D)), L)
    testmat = np.dot( np.dot( L.T, hess), L)
    if debug:
        print("TEST L D[:10] DOT")
        for row in testmat[:10]:
            for val in row[:10]:
                print("{:11.8f} ".format( val), end="")
            print()
    testmat = np.dot( np.dot( D.T, hess), D)
    if debug:
        print("TEST D H D DOT FULL")
        for row in testmat[:10]:
            for val in row[:10]:
                print("{:11.8f} ".format( val), end="")
            print()
        print("EIGS FULL:", converteig(freq_ic[:10]) )
    if not stdtr:
        _,D = np.linalg.eigh(hess)
    if remove > 0:
        TR = np.array(D[:,:remove])
        D = np.array(D[:,remove:])
        if debug:
            print("TR.shape")
            print(TR.shape)
    freq_ic,L = np.linalg.eigh(np.dot(np.dot(D.T, hess), D))
    testmat = np.dot( np.dot( L.T, np.dot( np.dot( D.T, hess), D)), L)
    if debug:
        print("AFTER REMOVE")
        print("TEST L D[:10] DOT")
        for row in testmat[:10]:
            for val in row[:10]:
                print("{:11.8f} ".format( val), end="")
            print()
    a,b = np.linalg.eigh( testmat)
    freq_mol = a
    testmat = np.dot( np.dot( D.T, hess), D)
    if debug:
        print("EIGS LDHDL:", converteig(a))
        print("TEST D H D DOT FULL")
        for row in testmat[:10]:
            for val in row[:10]:
                print("{:11.8f} ".format( val), end="")
            print()
        print("EIGS FULL:", converteig(freq_ic[:10]))
    
    #q = np.full(xyz.shape, bohr2angstrom) * np.sqrt(mass_per_atom[:,np.newaxis])
    #q = q.reshape(-1)
    if debug:
        print("q")
        print(q)
    S = np.dot(D.T,q)
    if debug:
        print("S[0:21]")
        print(S[0:21])
        print("np.dot(D.T, hess, D) shape")
        print(np.dot(np.dot(D.T, hess), D).shape)

    # Get the modes in IC, which is the transformed Hess
    freq_ic,L = np.linalg.eigh(np.dot(np.dot(D.T, hess), D))
    if debug:
        print("freq_ic")
        print(np.sign(freq_ic)*np.sqrt(np.abs(freq_ic))*conversion)

    
    if debug:
        print(M.shape, D.shape, L.shape)
    mol_modes = np.dot(np.dot(M, D), L)

    if debug:
        print("mol_modes.shape")
        print(mol_modes.shape)

    freq_TR,Ltr = np.linalg.eigh(np.dot(np.dot(TR.T, hess), TR))
    #testmat = np.dot( Ltr, TR.T)
    testmat = np.dot( np.dot( Ltr.T, np.dot( np.dot( TR.T, hess), TR)), Ltr)
    if debug:
        print("TEST Ltr' TR' H TR Ltr DOT (shape):", Ltr.shape, TR.shape)
        for row in testmat:
            for val in row:
                print("{:11.8f} ".format( converteig(val) ), end="")
            print()
    a,b = np.linalg.eigh( testmat)
    testmat = b
    freq_TR = a
    if debug:
        print("EIGS TESTMAT:", converteig(a))
        print("EIGS of EIGS")
        for row in testmat:
            for val in row:
                print("{:11.8f} ".format( val), end="")
            print()

    if debug:
        print("freq_TR")
        print(np.sign(freq_TR)*np.sqrt(np.abs(freq_TR))*conversion)

    
        print(M.shape, TR.shape, L.shape)
    TR_modes = np.dot(np.dot(M, TR), Ltr)

    if debug:
        print("TR_modes.shape")
        print(TR_modes.shape)



    if remove > 0:
        mol_modes = np.hstack((TR_modes, mol_modes))
        freq_ic = np.hstack((freq_TR, freq_mol))
        mol_modes[:,:remove] = TR_modes
    if debug:
        print("EIGS FULL AFTER COMBINE:", converteig(freq_ic[:10]))
    #else:
    #    Norm = 1./np.linalg.norm(TR, axis=0)

    #    print(TR_modes.shape, Norm.shape)
    #    for i,_ in enumerate(Norm):
    #        TR_modes[:,i] *= Norm[i]


    #for i in range( TR_modes.shape[1]):
    #    TR_modes[:,i] *=  q

    #Norm = np.linalg.norm(mol_modes, axis=0)

    #print(mol_modes.shape, Norm.shape)
    #for i,_ in enumerate(Norm):
    #    mol_modes[:,i] /= Norm[i]

    if verbose:
        for i, mode in enumerate(mol_modes.T):
            with open(mol_name+".mode_"+str(i)+".pure.xyz", 'w') as fid:
                mode_to_xyz(syms, xyz, (mode).reshape(-1, 3) , outfd=fid, 
                    comment=mol_name + " mode " + str(i))

    raw_mode = mol_modes.copy()
    for i in range( raw_mode.shape[1]):
        raw_mode[:,i] = raw_mode[:,i] / q
    if debug:
        print("cart modes norm (shape=", mol_modes.shape, ":")
        print(np.linalg.norm(mol_modes, axis=0))
        print("cart modes orthogonal (eps=1e-8)? :")
        orth = np.dot( mol_modes.T, mol_modes)
        for i in orth:
            for x in i:
                x = abs(x)
                if x > 1e-8:
                    print("{:3.1f}".format(x), end="")
                else:
                    print("Y", end="")
            print()

        print("Angular momentum of cart:")
        for i, mode in enumerate(mol_modes.T):
            print( "mode {:3d}: {:4.2e}".format( i, 
                angular_momentum( xyz, mass_per_atom, mode)))

        print("raw modes pre-norm:")
        print(np.linalg.norm( raw_mode, axis=0))
        print("raw modes orthogonal (eps=1e-8)? :")
        raw_mode /= np.linalg.norm( raw_mode, axis=0)
        print("raw modes post-norm:")
        print(np.linalg.norm( raw_mode, axis=0))
        orth = np.dot( raw_mode.T, raw_mode)
        for i in orth:
            for x in i:
                x = abs(x)
                if x > 1e-8:
                    print("{:3.1f}".format(x), end="")
                else:
                    print("Y", end="")
            print()

        print("Angular momentum of raw:")
        for i, mode in enumerate(raw_mode.T):
            print( "mode {:3d}: {:4.2e}".format( i, 
                angular_momentum( xyz, mass_per_atom, mode)))
            if verbose:
                with open(mol_name+".mode_"+str(i)+".raw.xyz", 'w') as fid:
                    mode_to_xyz(syms, xyz, (mode).reshape(-1, 3) , outfd=fid, 
                            comment=mol_name + " mode " + str(i))

    return mol_modes, converteig(freq_ic)

if 0:
    # angle_freq_file = open("angle_freq.txt", 'w')
    for mol_number, mol_name in enumerate(sys.argv[1:]):
        

        # hess, xyz, syms = load_xyz_hessian_from_old( d, mol_number)
        hess = np.loadtxt(mol_name + '.dat')
        print("hess shape:", hess.shape)
        if len(hess.shape) == 1:
            print("reshaping to a square..")
            hess = hess.reshape((int(hess.shape[0]**.5), int(hess.shape[0]**.5)))


        syms, xyz = load_xyz(mol_name + '.xyz')
        print(syms)


        # mol_name = "mol_" + str(mol_number)
        mol_xyz_fname = mol_name + ".min.xyz"
        IC = CoordClass(geometric_mol(mol_name + '.xyz'), build=True,
                            connect=connect, addcart=addcart, constraints=Cons,
                            cvals=CVals[0] if CVals is not None else None )

        #mass = np.array([[mass_table[atom]]*3 for atom in syms])
        mass = np.array([[mass_table[atom]]*3 for atom in syms])
        #mass_mat = (np.dot((mass).reshape(-1,1), (mass).reshape(1,-1)))

        # try to split the trans/rot from vibs
        #mass_per_atom = np.array([mass_table[atom] for atom in syms])


        #mol_modes_full, freq_ic_full = hessian_modes( hess, xyz, mass, mol_number, remove=0)
        if debug:
            print("AGAIN REMOVE")
        mol_modes, freq_ic = hessian_modes( hess, xyz, mass, mol_number, remove=0, stdtr=False)

        
        #print("COMPARE REMOVE")
        #printmat( np.dot(mol_modes[:,:6].T, mol_modes_full[:,:6]))
        #print("COMPARE REMOVE Trans")
        #printmat( np.dot(mol_modes_full[:,:6].T, mol_modes[:,:6]))
        
        mol_modes = mol_modes
        freq_ic = freq_ic


        #unit_mode = E[1] / np.linalg.norm(E[1],axis=0)
        unit_mode = mol_modes
        #unit_mode = raw_mode
        if(hasattr(IC, "Prims")):
            internals = IC.Prims.Internals
        else:
            internals = IC.Internals
        internals_order = np.arange(len(internals))
        if debug:
            print("internals_order")
            print(internals_order)
        B = IC.wilsonB(xyz)
        if debug:
            print("B.shape")
            print(B.shape)

        if swap_internals:
            internals_order[:6] = internals_order[-6:]
            internals_order[6:] -= 6
            internals = [internals[i] for i in internals_order]
            B = B[internals_order]

        a = np.array([np.dot(unit_mode[:,i], B[j] / np.linalg.norm(B[j])) \
            for i in range(unit_mode.shape[1]) for j in range(B.shape[0])])
        amat = a.reshape(unit_mode.shape[1], -1)
        srt = np.arange(len(amat[0]))#[::-1]
        #u, s, v = svd((amat[:,srt]**2).T, check_finite=False)
        #if debug:
        #    print(np.sqrt(s))
        #plt.plot(np.sqrt(s))
        #plt.matshow(u, cmap=plt.cm.gray_r)
        #plt.matshow(v, cmap=plt.cm.gray_r)
        #
        if debug:
            print("SIMILAR")
            for i in range( amat.shape[1]):
                for j in range( amat.shape[1]):
                    print( "{:3.1f} ".format( np.abs(np.dot( amat[:,i], amat[:,j]) )), end="")
                print()

        #plt.show()
        #srt = np.abs(amat).sum(axis=0).argsort()
        #amat[:,:6],amat[:,-6:] = amat[:,-6:],amat[:,:6]
        # this is the norm per IC
        col_norm = True
        estimate_freq = True
        amat = np.abs(amat)
        implied_k = np.zeros(max(amat.shape))
        if col_norm or estimate_freq:
            amat_norm_sum = np.sum((amat),axis=0)
            amat_normed = (amat.T) / amat_norm_sum[:,np.newaxis]
            if debug:
                print(amat_normed.sum(axis=1))
            #Ez = E[0]
            Ez = freq_ic
            implied_k = np.dot( amat_normed, 
                    np.sign(Ez)*np.sqrt( np.abs( Ez))*conversion ) \
                    #/ amat_normed.sum(axis=1)
            if debug:
                print(amat_norm_sum)
                print(amat_normed.sum(axis=1))
            if col_norm:
                srt = implied_k.argsort()
            else:
                srt = np.arange( len( implied_k))
            amat = amat[ :, srt]
        row_norm = False
        if row_norm:
            amat_norm_sum = np.sum((amat),axis=1)
            amat_normed = amat / amat_norm_sum[:,np.newaxis]
            #Ez = E[0]
            Ez = freq_ic
            #implied_k = np.dot( amat_normed, np.sign( Ez) * \
            #       np.sqrt( np.abs( Ez))*conversion ) \
            #       / amat_normed.sum(axis=0)
            implied_k = amat_norm_sum * np.sign(Ez)*np.sqrt(np.abs(Ez))*conversion
            srt = implied_k.argsort()
            amat = amat[srt]
        #srt = np.sum(np.square(amat),axis=0).argsort()
        #srt = np.arange(len(srt))[::-1]
        #implied_k *= conversion
        #implied_k = np.abs(amat).sum(axis=0) * conversion
        
        E = [freq_ic]
        n_IC = amat.shape[1]
        n_modes = amat.shape[0]
        n_min = min(n_IC, n_modes)
        tbl_dat = []
        print("FREQUENCIES (shape", freq_ic.shape, ")")
        Ez = np.sign(Ez)*np.sqrt( np.abs( Ez))*conversion
        print(Ez)
        continue
        
        #if estimate_freq:
        #    n_min = n_IC
        values = np.empty(n_IC)
        oFF_labels = ["X"] * n_IC
        all_params = ['Bonds', 'Angles', 'ImproperTorsions', 'ProperTorsions']
        for param in all_params:
            for index in d["mol_data"][mol_key][param]['indices']:
                found, loc = oFF_key_in_IC_Hessian(index, internals)
                if found:
                    oFF_labels[loc] = d["mol_data"][mol_key][param]['indices'][index]['oFF']
                    col = d["mol_data"][mol_key][param]['indices'][index]['column_idx']
                    values[loc] = d["mol_data"][mol_key][param]['values'][:,col]

        fid = open("mol_"+str(mol_number)+".oFF_freqs.dat", 'w')
        fid.write("# Mol key " + mol_key + "\n")
        for i in range(n_min):
            c0 = "{:4d}".format(i)
            sign = np.sign(E[0][i])
            if False and estimate_freq:
                c1 = "{:<9.2f}".format(implied_k[srt[i]])
            else:
                c1 = "{:<9.2f}".format(sign*np.sqrt(np.abs(E[0][i]))*conversion)
            c2 = "{:20s}".format(str(internals[srt[i]]))
            c3 = "{:9.2f}".format(implied_k[srt[i]])
            #print(c0, c1, c2,E[0][i], implied_k[srt[i]])    
            print(c0, c1, c2, c3, oFF_labels[srt[i]])    
            fid.write("{:3s} {:12.5f}\n".format(oFF_labels[srt[i]], implied_k[srt[i]]))
            # angle_freq_file.write("{:3s} {:9.4f} {:9.4f} {:s}\n".format(oFF_labels[srt[i]], implied_k[srt[i]], values[srt[i]], mol_key))
            tbl_dat.append([c0, c1, c2, c3])
        if(n_modes > n_IC):
            for i in range(n_IC, n_modes):
                c0 = "{:4d}".format(i)
                sign = np.sign(E[0][i])
                if False and estimate_freq:
                    c1 = "{:<9.2f}".format(implied_k[srt[i]])
                else:
                    c1 = "{:<9.2f}".format(sign*np.sqrt(np.abs(E[0][i]))*conversion)
                c2 = "{:20s}".format("")
                #print(c0, c1, c2,E[0][i], implied_k[srt[i]])    
                c3 = "{:9.2f}".format(implied_k[srt[i]])
                print(c0, c1, c2, c3)
                tbl_dat.append([c0, c1, c2, implied_k[srt[i]]])
        elif(n_modes < n_IC):
            for i in range(n_modes, n_IC):
                c0 = "{:4d}".format(i)
                c1 = "{:9s}".format("")
                c2 = "{:20s}".format(str(internals[srt[i]]))
                c3 = "{:9.2f}".format(implied_k[srt[i]])
                print(c0, c1, c2, c3, oFF_labels[srt[i]])    
                fid.write("{:3s} {:12.5f}\n".format(oFF_labels[srt[i]], implied_k[srt[i]]))
                # angle_freq_file.write("{:3s} {:9.4f} {:9.4f} {:s}\n".format(oFF_labels[srt[i]], implied_k[srt[i]], values[srt[i]], mol_key))
                tbl_dat.append([c0, c1, c2, c3])
        fid.close()

        fig = plt.figure(figsize=(14,8),dpi=200)
        ax = fig.add_subplot(111)
        #ax2 = fig.add_subplot(122)
        #ax2 = ax.twinx()


        #img = ax.matshow(np.dot(amat.T, amat) , cmap=plt.cm.gray_r, vmin=0.0, vmax=1.0)
        img = ax.matshow( amat, cmap=plt.cm.hot_r, vmin=0.3, vmax=.8)
        


        #similarity = np.zeros((amat.shape[1], amat.shape[1]))
        #for i in range(amat.shape[1]):
        #    ref = amat[:,i]
        #    ref /= np.linalg.norm(ref)
        #    similarity[i,i] = 1.0
        #    for j in range(i+1,amat.shape[1]):
        #        tgt = amat[:,j]
        #        tgt /= np.linalg.norm(tgt)
        #        similarity[i,j] = np.dot(ref, tgt)
        #        similarity[j,i] = similarity[i,j]
        #img2 = ax2.matshow(similarity**2)
        #img = ax.matshow(np.abs(amat), cmap=plt.cm.gray_r, vmin=0.0, vmax=1.0)

        #ax2.plot(np.abs(amat).sum(axis=0))
        #ax.set_xlim(-.5,amat.shape[0]+.5)
        #ax2.set_xlim(-.5,amat.shape[0]+.5)
        #ax.set_ylim(-.5,amat.shape[1]+.5)
        tbl = ax.table(cellText = tbl_dat, 
                colLabels=['', r'Freq. (cm$^{-1}$)', 'IC Name', 'Est.'], 
                colLoc='left', loc='left', cellLoc='left', 
                edges='open', colWidths=[.05,.1,.1, .3])
        tbl.scale(1,.65)
        tbl.set_fontsize(6)
        #tbl.AXESPAD = .1
        #fig.canvas.draw()
        fig.colorbar(img, ax=ax)
        fig.subplots_adjust(left=0.35, right=.99,wspace=.1)
        ax.set_xlabel("Primitive redundant coord")
        ax.set_ylabel("Cartesian mode (sorted)")

        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(axis='both', which='major', lw=1, ls='--', alpha=.3)
        fig.suptitle("Projection of normed Cartesian modes and B (abs. val.) for smiles "+mol_key)
        fig.savefig("test_ic."+mol_name+".png")
        plt.close(fig)
    # angle_freq_file.close()

if 0:
    with open("db.pickle", 'rb') as fid:
        d = pickle.load(fid)
    #for mol_number in [20, 21, 22, 23]:
    #for mol_number in range(30):
    angle_freq_file = open("angle_freq.txt", 'w')
    #for mol_number in range(len(d['mol_data'].keys())):
    for mol_number in [0]:
        

        hess, xyz, syms = load_xyz_hessian_from_old( d, mol_number)



        mol_name = "mol_" + str(mol_number)
        mol_xyz_fname = mol_name + ".min.xyz"
        mol_key = list(d['mol_data'].keys())[mol_number]
        IC = CoordClass(geometric_mol(mol_xyz_fname), build=True,
                            connect=connect, addcart=addcart, constraints=Cons,
                            cvals=CVals[0] if CVals is not None else None )

        #mass = np.array([[mass_table[atom]]*3 for atom in syms])
        mass = np.array([[mass_table[atom]]*3 for atom in syms])
        #mass_mat = (np.dot((mass).reshape(-1,1), (mass).reshape(1,-1)))

        # try to split the trans/rot from vibs
        #mass_per_atom = np.array([mass_table[atom] for atom in syms])


        #mol_modes_full, freq_ic_full = hessian_modes( hess, xyz, mass, mol_number, remove=0)
        if debug:
            print("AGAIN REMOVE")
        mol_modes, freq_ic = hessian_modes( hess, xyz, mass, mol_number, remove=6, stdtr=False)

        
        #print("COMPARE REMOVE")
        #printmat( np.dot(mol_modes[:,:6].T, mol_modes_full[:,:6]))
        #print("COMPARE REMOVE Trans")
        #printmat( np.dot(mol_modes_full[:,:6].T, mol_modes[:,:6]))
        
        mol_modes = mol_modes
        freq_ic = freq_ic


        #unit_mode = E[1] / np.linalg.norm(E[1],axis=0)
        unit_mode = mol_modes
        #unit_mode = raw_mode
        if(hasattr(IC, "Prims")):
            internals = IC.Prims.Internals
        else:
            internals = IC.Internals
        internals_order = np.arange(len(internals))
        if debug:
            print("internals_order")
            print(internals_order)
        B = IC.wilsonB(xyz)
        if debug:
            print("B.shape")
            print(B.shape)

        if swap_internals:
            internals_order[:6] = internals_order[-6:]
            internals_order[6:] -= 6
            internals = [internals[i] for i in internals_order]
            B = B[internals_order]

        a = np.array([np.dot(unit_mode[:,i], B[j] / np.linalg.norm(B[j])) \
            for i in range(unit_mode.shape[1]) for j in range(B.shape[0])])
        amat = a.reshape(unit_mode.shape[1], -1)
        srt = np.arange(len(amat[0]))#[::-1]
        #u, s, v = svd((amat[:,srt]**2).T, check_finite=False)
        #if debug:
        #    print(np.sqrt(s))
        #plt.plot(np.sqrt(s))
        #plt.matshow(u, cmap=plt.cm.gray_r)
        #plt.matshow(v, cmap=plt.cm.gray_r)
        #
        if debug:
            print("SIMILAR")
            for i in range( amat.shape[1]):
                for j in range( amat.shape[1]):
                    print( "{:3.1f} ".format( np.abs(np.dot( amat[:,i], amat[:,j]) )), end="")
                print()

        #plt.show()
        #srt = np.abs(amat).sum(axis=0).argsort()
        #amat[:,:6],amat[:,-6:] = amat[:,-6:],amat[:,:6]
        # this is the norm per IC
        col_norm = True
        estimate_freq = True
        amat = np.abs(amat)
        implied_k = np.zeros(max(amat.shape))
        if col_norm or estimate_freq:
            amat_norm_sum = np.sum((amat),axis=0)
            amat_normed = (amat.T) / amat_norm_sum[:,np.newaxis]
            if debug:
                print(amat_normed.sum(axis=1))
            #Ez = E[0]
            Ez = freq_ic
            implied_k = np.dot( amat_normed, 
                    np.sign(Ez)*np.sqrt( np.abs( Ez))*conversion ) \
                    #/ amat_normed.sum(axis=1)
            if debug:
                print(amat_norm_sum)
                print(amat_normed.sum(axis=1))
            if col_norm:
                srt = implied_k.argsort()
            else:
                srt = np.arange( len( implied_k))
            amat = amat[ :, srt]
        row_norm = False
        if row_norm:
            amat_norm_sum = np.sum((amat),axis=1)
            amat_normed = amat / amat_norm_sum[:,np.newaxis]
            #Ez = E[0]
            Ez = freq_ic
            #implied_k = np.dot( amat_normed, np.sign( Ez) * \
            #       np.sqrt( np.abs( Ez))*conversion ) \
            #       / amat_normed.sum(axis=0)
            implied_k = amat_norm_sum * np.sign(Ez)*np.sqrt(np.abs(Ez))*conversion
            srt = implied_k.argsort()
            amat = amat[srt]
        #srt = np.sum(np.square(amat),axis=0).argsort()
        #srt = np.arange(len(srt))[::-1]
        #implied_k *= conversion
        #implied_k = np.abs(amat).sum(axis=0) * conversion
        
        E = [freq_ic]
        n_IC = amat.shape[1]
        n_modes = amat.shape[0]
        n_min = min(n_IC, n_modes)
        tbl_dat = []
        
        #if estimate_freq:
        #    n_min = n_IC
        values = np.empty(n_IC)
        oFF_labels = ["X"] * n_IC
        all_params = ['Bonds', 'Angles', 'ImproperTorsions', 'ProperTorsions']
        for param in all_params:
            for index in d["mol_data"][mol_key][param]['indices']:
                found, loc = oFF_key_in_IC_Hessian(index, internals)
                if found:
                    oFF_labels[loc] = d["mol_data"][mol_key][param]['indices'][index]['oFF']
                    col = d["mol_data"][mol_key][param]['indices'][index]['column_idx']
                    values[loc] = d["mol_data"][mol_key][param]['values'][:,col]

        fid = open("mol_"+str(mol_number)+".oFF_freqs.dat", 'w')
        fid.write("# Mol key " + mol_key + "\n")
        for i in range(n_min):
            c0 = "{:4d}".format(i)
            sign = np.sign(E[0][i])
            if False and estimate_freq:
                c1 = "{:<9.2f}".format(implied_k[srt[i]])
            else:
                c1 = "{:<9.2f}".format(sign*np.sqrt(np.abs(E[0][i]))*conversion)
            c2 = "{:20s}".format(str(internals[srt[i]]))
            c3 = "{:9.2f}".format(implied_k[srt[i]])
            #print(c0, c1, c2,E[0][i], implied_k[srt[i]])    
            print(c0, c1, c2, c3, oFF_labels[srt[i]])    
            fid.write("{:3s} {:12.5f}\n".format(oFF_labels[srt[i]], implied_k[srt[i]]))
            angle_freq_file.write("{:3s} {:9.4f} {:9.4f} {:s}\n".format(oFF_labels[srt[i]], implied_k[srt[i]], values[srt[i]], mol_key))
            tbl_dat.append([c0, c1, c2, c3])
        if(n_modes > n_IC):
            for i in range(n_IC, n_modes):
                c0 = "{:4d}".format(i)
                sign = np.sign(E[0][i])
                if False and estimate_freq:
                    c1 = "{:<9.2f}".format(implied_k[srt[i]])
                else:
                    c1 = "{:<9.2f}".format(sign*np.sqrt(np.abs(E[0][i]))*conversion)
                c2 = "{:20s}".format("")
                #print(c0, c1, c2,E[0][i], implied_k[srt[i]])    
                c3 = "{:9.2f}".format(implied_k[srt[i]])
                print(c0, c1, c2, c3)
                tbl_dat.append([c0, c1, c2, implied_k[srt[i]]])
        elif(n_modes < n_IC):
            for i in range(n_modes, n_IC):
                c0 = "{:4d}".format(i)
                c1 = "{:9s}".format("")
                c2 = "{:20s}".format(str(internals[srt[i]]))
                c3 = "{:9.2f}".format(implied_k[srt[i]])
                print(c0, c1, c2, c3, oFF_labels[srt[i]])    
                fid.write("{:3s} {:12.5f}\n".format(oFF_labels[srt[i]], implied_k[srt[i]]))
                angle_freq_file.write("{:3s} {:9.4f} {:9.4f} {:s}\n".format(oFF_labels[srt[i]], implied_k[srt[i]], values[srt[i]], mol_key))
                tbl_dat.append([c0, c1, c2, c3])
        fid.close()

        fig = plt.figure(figsize=(14,8),dpi=200)
        ax = fig.add_subplot(111)
        #ax2 = fig.add_subplot(122)
        #ax2 = ax.twinx()


        #img = ax.matshow(np.dot(amat.T, amat) , cmap=plt.cm.gray_r, vmin=0.0, vmax=1.0)
        img = ax.matshow( amat, cmap=plt.cm.hot_r, vmin=0.3, vmax=.8)
        


        #similarity = np.zeros((amat.shape[1], amat.shape[1]))
        #for i in range(amat.shape[1]):
        #    ref = amat[:,i]
        #    ref /= np.linalg.norm(ref)
        #    similarity[i,i] = 1.0
        #    for j in range(i+1,amat.shape[1]):
        #        tgt = amat[:,j]
        #        tgt /= np.linalg.norm(tgt)
        #        similarity[i,j] = np.dot(ref, tgt)
        #        similarity[j,i] = similarity[i,j]
        #img2 = ax2.matshow(similarity**2)
        #img = ax.matshow(np.abs(amat), cmap=plt.cm.gray_r, vmin=0.0, vmax=1.0)

        #ax2.plot(np.abs(amat).sum(axis=0))
        #ax.set_xlim(-.5,amat.shape[0]+.5)
        #ax2.set_xlim(-.5,amat.shape[0]+.5)
        #ax.set_ylim(-.5,amat.shape[1]+.5)
        tbl = ax.table(cellText = tbl_dat, 
                colLabels=['', r'Freq. (cm$^{-1}$)', 'IC Name', 'Est.'], 
                colLoc='left', loc='left', cellLoc='left', 
                edges='open', colWidths=[.05,.1,.1, .3])
        tbl.scale(1,.65)
        tbl.set_fontsize(6)
        #tbl.AXESPAD = .1
        #fig.canvas.draw()
        fig.colorbar(img, ax=ax)
        fig.subplots_adjust(left=0.35, right=.99,wspace=.1)
        ax.set_xlabel("Primitive redundant coord")
        ax.set_ylabel("Cartesian mode (sorted)")

        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(axis='both', which='major', lw=1, ls='--', alpha=.3)
        fig.suptitle("Projection of normed Cartesian modes and B (abs. val.) for smiles "+mol_key)
        fig.savefig("test_ic."+mol_name+".png")
        plt.close(fig)
    angle_freq_file.close()
