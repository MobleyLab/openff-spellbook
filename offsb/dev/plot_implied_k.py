import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
import pandas
#%pylab

def plot_implied_k(fname, save=None, normalize_per_frag=False, normalize_per_set=False,
        normalize_per_mol=True):
    with open(fname, 'rb') as fid:
        d = pickle.load(fid)
    x = [] ; y = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fid = None
    norm_keys = {'b': 0, 'a': 1, 't': 1, 'i': 1}
    if(save is not None):
        fid = open(save, 'w')
    else:
        fid = sys.stdout

    counts = {'bb': 0, 'ba': 0, 'bt': 0, 'bi': 0,
              'ab': 0, 'aa': 0, 'at': 0, 'ai': 0,
              'tb': 0, 'ta': 0, 'tt': 0, 'ti': 0,
              'ib': 0, 'ia': 0, 'it': 0, 'ii': 0}
    measurements = ['Bonds', 'Angles'] # 'ImproperTorsions']
    measurements.append('ProperTorsions')
    measurements.append('ImproperTorsions')
    norm_mat=np.ones((2,2))
    outdata= {'smiles': [], 'atoms': [], 'conformation': [], 'label': [], 'k': [], 'k_std': [], 'hess_normed': []}
    if(normalize_per_set):
        norm_mat=np.zeros((4,4))
        for mol in d["mol_data"].keys():
            for measureA, measureB in itertools.product(measurements,measurements):
                if( sum([(x not in d["mol_data"][mol]) for x in [measureA, measureB]])):
                    continue
                for termA in d["mol_data"][mol][measureA]['indices'].keys():
                    termA1 = tuple([x+1 for x in termA])
                    kA = d["mol_data"][mol][measureA]['indices'][termA]['oFF'][0] 
                    if(termA1 not in d["mol_data"][mol]['hessian']['prim']):
                        continue
                    for termB in d["mol_data"][mol][measureB]['indices'].keys():
                        termB1 = tuple([x+1 for x in termB])
                        kB = d["mol_data"][mol][measureB]['indices'][termB]['oFF'][0] 
                        if(termB1 in d["mol_data"][mol]['hessian']['prim'][termA1]):
                            val =d["mol_data"][mol]['hessian']['prim'][termA1][termB1]
                            norm_mat[norm_keys[kA],norm_keys[kB]] += val**2
                            #print("val", val**2, "added to", kA, kB, norm_mat[norm_keys[kA],norm_keys[kB]])
                            if(kA != kB):
                                counts[kA+kB] += 1
                                #print(termA, termB, kA, kB, val, norm_mat[norm_keys[kA],norm_keys[kB]])

        norm_mat = np.sqrt(norm_mat)
        fid.write("Per set norms:\n")
        fid.write(str(norm_mat) + '\n')
        outdata['norm_set'] = norm_mat
        diff = np.abs(norm_mat - norm_mat.T) / ((norm_mat + norm_mat)/2)
        for i,j in itertools.product(*[range(len(norm_mat))]*2):
            if i<j: return
            if(diff[i,j] < .05):
                mean = (norm_mat[i,j] + norm_mat[j,i]) / 2.0
                norm_mat[i,j] = mean
                norm_mat[j,i] = mean
        print("Per set norms:\n")
        print(str(norm_mat) + '\n')
        if(np.abs(norm_mat[0,1] - norm_mat[1,0]) > 1e-7):
            print("norm not symmetric!!!")
            return
    for mol in d["mol_data"].keys():
        if(normalize_per_mol):
            norm_mat=np.zeros((4,4))
            for measureA, measureB in itertools.product(measurements,measurements):
                if(sum([(x not in d["mol_data"][mol]) for x in [measureA, measureB]])):
                    continue
                for termA in d["mol_data"][mol][measureA]['indices'].keys():
                    termA1 = tuple([x+1 for x in termA])
                    kA = d["mol_data"][mol][measureA]['indices'][termA]['oFF'][0] 
                    if(termA1 not in d["mol_data"][mol]['hessian']['prim']):
                        continue
                    for termB in d["mol_data"][mol][measureB]['indices'].keys():
                        termB1 = tuple([x+1 for x in termB])
                        kB = d["mol_data"][mol][measureB]['indices'][termB]['oFF'][0] 
                        if(termB1 in d["mol_data"][mol]['hessian']['prim'][termA1]):
                            val =d["mol_data"][mol]['hessian']['prim'][termA1][termB1]
                            norm_mat[norm_keys[kA],norm_keys[kB]] += val**2
                            #print("val", val**2, "added to", kA, kB, norm_mat[norm_keys[kA],norm_keys[kB]])
                            if(kA != kB):
                                counts[kA+kB] += 1
                                #print(termA, termB, kA, kB, val, norm_mat[norm_keys[kA],norm_keys[kB]])

            norm_mat = np.sqrt(norm_mat)
            fid.write("Per frag norms:\n")
            fid.write(str(norm_mat) + '\n')
            diff = np.abs(norm_mat - norm_mat.T) / ((norm_mat + norm_mat)/2)
            for i,j in itertools.product(*[range(len(norm_mat))]*2):
                if i<j: continue
                if(diff[i,j] < .05):
                    mean = (norm_mat[i,j] + norm_mat[j,i]) / 2.0
                    norm_mat[i,j] = mean
                    norm_mat[j,i] = mean
            print("Per frag norms:\n")
            print(str(norm_mat) + '\n')
            if(np.abs(norm_mat[0,1] - norm_mat[1,0]) > 1e-7):
                print("norm not symmetric!!!")
                return
        for measure in measurements:
            if(measure not in d["mol_data"][mol]):
                continue

            for bond in d["mol_data"][mol][measure]['indices'].keys():
                bond1 = bond #tuple([x+1 for x in bond])
                if(bond1 in d["mol_data"][mol]['hessian']['prim']):
                    #if(d["mol_data"][mol][measure]['indices'][bond]['oFF'] != "b5"):
                    #    continue
                    kA = d["mol_data"][mol][measure]['indices'][bond]['oFF']
                    x.append(kA)
                    kA = kA[0]
                    vals = []
                    for key,val in d["mol_data"][mol]['hessian']['prim'][bond1].items():
                        exists = False
                        for idx in bond1:
                            if(idx in key):
                                exists = True

                        if(exists):
                            kB = 'b' if len(key) == 2 else 'a'
                            vals.append(val / norm_mat[norm_keys[kA],norm_keys[kB]])
                    vals = np.array(vals)
                    #vals = np.array(list(d["mol_data"][mol]['hessian']['prim'][bond1].values()))
                    #vals = np.array(d["mol_data"][mol]['hessian']['prim'][bond1][bond1])
                    y.append((vals.sum()))
                    label = d["mol_data"][mol][measure]['indices'][bond]['oFF'] 
                    outstr = "{:50s} {:14s} {:4s} {:12.8e} {:10.2f}".format(mol, str(bond), label, vals.mean(), vals.std())
                    outdata['smiles'].append( mol[:-2] )
                    outdata['conformation'].append(mol.split("-")[-1])
                    outdata['atoms'].append(str(bond))
                    outdata['label'].append(label)
                    outdata['k'].append(vals.sum())
                    outdata['k_std'].append(vals.std())
                    outdata['hess_normed'].append(vals)

                    if(save is not None):
                        fid.write(outstr + '\n')
                    print(outstr)
                else:
                    print("MISS", bond)
    if(save is not None):
        fid.close()
    xletter = np.array([i[0] for i in x])
    xnumber = np.array([int(i[1:]) for i in x])
    s = np.lexsort((y,xnumber,xletter))
    xs = np.array([l+str(n) for l,n in zip(xletter[s],xnumber[s])])
    ys = np.array(y)[s]
    ax.plot(xs,ys,'k.', ms=5)
    #ax.set_yscale('log')
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    if(not normalize_per_mol):
        param_id = [i for i in d['oFF'].keys() if (i[0] not in 'ti' and i[1] != '0' and ("version" not in i))]
        param_k = [d['oFF'][i]['k'] for i in d['oFF'].keys() if (i[0] not in 'ti' and i[1] != '0' and ("version" not in i))]
        param_k = [float(str(i).split()[0]) for i in param_k]
        for i,(param,k) in enumerate(zip(param_id, param_k)):
            if(param[0] == 'b'):
                param_k[i] /= norm_mat[0,0]
            else:
                param_k[i] /= norm_mat[1,1]
        ax.plot(param_id, param_k, 'r.', ms=10)

    return x,y,outdata, fig
#clf()
#for measure in ['Bonds', 'Angles', 'ImproperTorsions']:
#    for label in d['oFF'].keys():
#        y = []
#        for mol in d["mol_data"].keys():
#            print(label)
#            for bond in d["mol_data"][mol][measure]['indices'].keys():
#                bond1 = tuple([x+1 for x in bond])
#                if(bond1 in d["mol_data"][mol]['hessian']['prim']):
#                    if(d["mol_data"][mol][measure]['indices'][bond]['oFF'] != label):
#                        continue
#         
#                    x.append(d["mol_data"][mol][measure]['indices'][bond]['oFF'])
#                    vals = []
#                    for key,val in d["mol_data"][mol]['hessian']['prim'][bond1].items():
#                        if(bond1[0] in key or bond1[1] in key):
#                            vals.append(val)
#                    vals = np.array(vals)
#                #vals = np.array(list(d["mol_data"][mol]['hessian']['prim'][bond1].values()))
#                #vals = np.array(d["mol_data"][mol]['hessian']['prim'][bond1][bond1])
#                    y.append((vals.mean()))
#                    #print("{:50s} {:14s} {:4s} {:10.2f} {:10.2f}".format(mol, str(bond), d["mol_data"][mol][measure]['indices'][bond]['oFF'], vals.mean(), vals.std()))
#        hist(y, bins=50, density=False, histtype='step', label=label)
