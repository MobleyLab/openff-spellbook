
#/usr/bin/env python3

import os 
import numpy as np
import matplotlib.pyplot as plt
import collections

def redo_all(fname='bonds_N-C_deg-2.npz', mindb=None, minfname='bonds_N-C_deg-2.minimum.npz', params={"Mol": {}, "Bonds": ['b1'], "Angles": [], "ImproperTorsions": [], "ProperTorsions": []}):
    import pickle
    d = None
    m = None
    if(os.path.exists(fname)):
        d = np.load(fname, allow_pickle=True)
    if(mindb is not None):
        m = mindb
    elif(os.path.exists(minfname)):
        if(minfname.split('.')[-1] == "npz"):
            m = np.load(minfname, allow_pickle=True)
        else:
            with open(minfname, "rb") as fid:
                m = dict(pickle.load(fid))
    if(m is None):
        return

    params = collections.defaultdict(list)
    if("Mol" not in params):
        params["Mol"] = {}

    measurements = ["Bonds", "Angles", "ImproperTorsions", "ProperTorsions"]
    colors = ['red', 'blue', 'purple', 'green', 'orange', 'yellow']
    rows = 1
    if(len(params["Bonds"]) + len(params["Angles"]) + len(params["ImproperTorsions"]) + len(params["ImproperTorsions"]) == 0):
        rows = 2
    elif(len(params["Bonds"]) > 0 and (len(params["Angles"]) > 0 or len(params["ImproperTorsions"]) > 0)):
        rows = 2

    mol_list = params['Mol']
    if(mol_list == {}):
        vals = list(m["mol_data"].keys())
        mol_list = dict(zip(range(len(vals)), [vals]))

    for jj, (name, smiles_list) in enumerate(mol_list.items()):
        print("{:4d}, Search molecule {:64s}:".format(jj,str(name)), end=" ")
        plt.clf()
        hits=0
        ax = [plt.subplot2grid((rows,3),(0,0), colspan=2)]
        ax2 = [plt.subplot2grid((rows,3),(0,2), sharey=ax[0])]
        if(rows > 1):
            ax.append(plt.subplot2grid((rows,3),(1,0), colspan=2))
            ax2.append(plt.subplot2grid((rows,3),(1,2), sharey=ax[1]))
        if(len(params["Bonds"]) > 0):
            ax[0].set_ylabel(r"Bond length ($\mathrm{\AA}$)")
            if(len(params["Angles"]) > 0):
                ax[1].set_ylabel(r"Angle ($\mathrm{deg}$)")
                ax[1].set_xlabel(r"Torsiondrive angle (deg.)")
            else:
                ax[0].set_xlabel(r"Torsiondrive angle (deg.)")
        elif(len(params["Angles"]) > 0):
            ax[0].set_ylabel(r"Angle ($\mathrm{deg}$)")
            ax[0].set_xlabel(r"Torsiondrive angle (deg.)")

        fig = plt.figure(1)
        fig.subplots_adjust(wspace=.3, hspace=.2)
        ddata = []
        mdata = {}
        mdatamean = {}
        lows = []

        #first is key
        #then is param vs data
        # then is 1xN for param data (choose 0)
        # then is param

        #oFF_labels = [c[0] for c in m.values()]
        used_labels = []
        c_idx = -1
        used_colors = {}
        bond_r0 = {}
        bond_k = {}
        bond_smirks = {}
        smiles_hits = []
        smiles_idx = []
        bond_dict = {}

        for ii,smiles in enumerate(smiles_list):
            plot_idx = -1
            all_params = []
            skip=False
            for measure in measurements:
                try:
                    all_params += [p['oFF'] for p in m["mol_data"][smiles][measure]["indices"].values()]
                except KeyError:
                    print("Mol with smiles", smiles, "empty or corrupted. Skipping")
                    skip=True
                    break
                if(skip): break
                for param in params[measure]:
                    if(not (param in all_params)):
                        #print(smiles, "Does not have", param)
                        skip=True
                        break
            if(skip):
                continue
            #print("HIT!")
            #try:
                #print(m[smiles].shape, end=" ")
                #if(False and (not (d is None)) and smiles in d):
            if(0):
                for i,(j,jmin) in enumerate(zip(d[smiles][1].T[1:],m[smiles][1].T[1:])):
                    label = m[smiles][0][0][i]['id']
                    #print(ii, i, smiles, end=" ")
                    ax = subplot2grid((1,3),(0,0), colspan=2)
                    ax.plot(d[smiles][1][:,0], j,'b.', ms=5)
                    ax.plot(m[smiles][1][:,0], jmin, 'k.-', ms=7)
                    ddata += list(j)
                    mdata.setdefault(m[smiles][0][i], [])
                    mdata[m[smiles][0][i]] += list(jmin)
                    ax2.hist(j,bins=20, color='b', orientation='horizontal')
                    ax2.hist(jmin,bins=20, color='k', orientation='horizontal')
            else:
                for measure_ii, measure in enumerate(measurements):
                    if(measure not in params):
                        continue
                    if(len(params[measure]) == 0):
                        continue
                    plot_idx += 1
                    for i,(index_key,index_dat) in enumerate(m["mol_data"][smiles][measure]["indices"].items()):
                        label = index_dat['oFF']
                        #print(ii, i, smiles, jmin.mean(), end=" ")
                        plot_label=None
                        if(not (label in params[measure])):
                            #print("param", label, " not wanted. skipping")
                            continue
                        hits += 1
                        if(not (smiles in smiles_hits)):
                            smiles_hits.append(smiles)
                            smiles_idx.append(ii)
                        if(not (label in used_labels)):
                            plot_label=label
                            used_labels.append(label)
                            c_idx += 1
                            used_colors[label] = colors[c_idx]
                            if( not ( label in ["a0", "b0", "t0", "i0"] )):
                                if(measure == "Bonds"):
                                    bond_r0[label] = m['oFF'][label]['length']
                                elif(measure == "Angles"):
                                    bond_r0[label] = m['oFF'][label]['angle']
                                bond_smirks[label] = m['oFF'][label]['smirks']
                                bond_dict[label] = m['oFF'][label]
                                if( not (label[0] in 'ti')):
                                    bond_k[label] = m['oFF'][label]['k']
                        
                        color = used_colors[label]
                        td_ang = m["td_ang"][m["mol_data"][smiles]["td_ang"]]
                        measure_data = m["mol_data"][smiles][measure]["values"][:,index_dat["column_idx"]]
                        if(measure == "Angles"):
                            pass
                            #measure_data *= np.pi/180
                        #print(plot_idx, "plotting", label, "td_ang=", td_ang)
                        ax[plot_idx].plot(td_ang, measure_data, lw=.1, ls='-', marker='.', color=color, label=plot_label, ms=2)
                        mdata.setdefault(label, [])
                        mdata[label] += list(measure_data)

                        mdatamean.setdefault(smiles, {})
                        mdatamean[smiles].setdefault(label, [])
                        mdatamean[smiles][label].append(measure_data.mean())
            #elif(jmin.mean() < 1.433):
            #    print("Med:", ii, k)
            #else:
            #    print("High:", ii, k)
            #print()

            #except TypeError:
            #    print("TypeError")
            #except IndexError:
            #    print("IndexError")
        title = str(dict([(k,v) for k,v in params.items() if (v != [] and k != "Mol")]))
        print(title,"HITS=", hits, end=" ")
        param_list = [p for p_list in measurements for p in params[p_list]]
        for p in param_list:
            if( p in ["a0", "b0", "i0", "t0"] ):
                m['oFF'][p]['smirks'] = "None"
        if(hits > 0):
            smiles_idx_str = ("{:s}."*len(smiles_idx)).format(*[str(x) for x in smiles_idx]) 
            param_str = ("{:s}."*len(param_list)).format(*[str(x) for x in param_list]) 
            if(len(ddata) > 0):
                ax2.hist(ddata,bins=50, color='blue', orientation='horizontal')
            kT = (.001987*298.15)
            for ii,(label,dat) in enumerate(mdata.items()):
                plot_idx = 0
                if(label[0] in "ait"):
                    #num *= np.pi/180
                    if(rows == 2):
                        plot_idx = 1
                    else:
                        plot_idx = 0
                color=used_colors[label]
                ax2[plot_idx].hist(dat,bins=50, color=used_colors[label], histtype='step', orientation='horizontal')
                if( label in ["a0", "b0", "t0", "i0"] ):
                    continue
                # TODO: calculate spread of torsions
                if( label[0] in 'it' ):
                    continue
                num = float(str(bond_r0[label]).split()[0])
                force_k = float(str(bond_k[label]).split()[0])
                delta = (2*(kT)/force_k)**.5
                if(label[0] in "ait"):
                    delta *= 180/np.pi
                dat = np.array(dat)
                if((dat < (num - delta)).any() or (dat > (num + delta)).any()):
                    print("RED FLAG (",label, ")", end=" ")
                elif(dat.max() < num or dat.min() > num):
                    print("YELLOW FLAG (",label, ")", end=" ")
                ax[plot_idx].axhline(y=num, ls='-', marker='.', color='black', ms=20, mec='black', mfc=color)
                ax[plot_idx].axhline(y=num + delta, ls='--', marker='.', color='black', ms=10, mec='black', mfc=color)
                ax[plot_idx].axhline(y=num - delta, ls='--', marker='.', color='black', ms=10, mec='black', mfc=color)
                ax2[plot_idx].axhline(y=num, ls='-', marker='.', color='black', ms=20, mec='black', mfc=color)
                ax2[plot_idx].axhline(y=num + delta, ls='--', marker='.', color='black', ms=10, mec='black', mfc=color)
                ax2[plot_idx].axhline(y=num - delta, ls='--', marker='.', color='black', ms=10, mec='black', mfc=color)
            ax[0].legend()
            if(rows > 1):
                ax[1].legend()
            print_header = True
            smiles_out_fname = ("mol." + str(jj) + ".smiles_with."+"{:s}."*len(param_list)+"txt").format(*param_list)
            if(os.path.exists(smiles_out_fname)):
                print_header = False
            with open(smiles_out_fname, 'w') as fid:
                for measure in measurements:
                    for label in params[measure]:
                        if(not (label in bond_r0)):
                            continue
                        r0 = float(str(bond_r0[label]).split()[0])
                        force_k = float(str(bond_k[label]).split()[0])
                        smirks = bond_dict[label]['smirks']
                        if(print_header):
                            fid.write("# idx confs flag smiles lengths ({:s} {:s} r0 = {:6.2f}, k = {:6.2f})\n".format(
                                label, smirks, r0, force_k))
                        delta = (2*(kT)/force_k)**.5
                        if(measure == "Angles"):
                            delta *= 180/np.pi
                            pass
                            #num *= np.pi/180
                        #print([(smiles, mdatamean[smiles]) for smiles in smiles_hits])
                        dat = []
                        outstr = []
                        per_term_flag = "G"
                        flag = "G"
                        bond_indices = []
                        valence_term = {}
                        for idx,smiles in zip(smiles_idx,smiles_hits):
                            for index_key, index_dat in m["mol_data"][smiles][measure]["indices"].items():
                                if(index_dat['oFF'] == label):
                                    bond_indices.append(index_key)
                                    valence_term[index_key] = []
                        for index_key in valence_term:
                            single_terms = []
                            for idx,smiles in zip(smiles_idx,smiles_hits):
                                try:
                                    col_idx = m["mol_data"][smiles][measure]["indices"][index_key]["column_idx"]
                                except KeyError as e:
                                    print("\n** Missing", e, ": Probably different molecules with same smiles. Check the output! ** ")
                                    continue
                                vals = np.atleast_1d(m["mol_data"][smiles][measure]["values"][:,col_idx])
                                valence_term[index_key] = np.append(valence_term[index_key], vals)
                                flag = "G"
                                if((vals < (r0 - delta)).any() or (vals > (r0 + delta)).any()):
                                    flag = "R"
                                elif(vals.max() < r0 or vals.min() > r0):
                                    flag = "Y"
                                avglen = vals.mean()
                                rmslen = np.linalg.norm(vals - r0)
                                rmsene = np.linalg.norm(force_k/2 * (vals - r0)**2)
                                mrmslen = np.linalg.norm(vals - avglen)
                                mrmsene = np.linalg.norm(force_k/2 * (vals - avglen)**2)
                                single_terms.append((" {:4d} {:5d} {:>4s} {:60s} " + "kRMS(L)=" + "{:7.4f} "*1 + "kRMS(E)={:7.4f} " + "meanL={:7.4f} mRMS(L)=" + "{:7.4f} "*1 + "mRMS(E)={:7.4f} " + "Vals=(" + ("{:10.4f} "*(len(vals))) + ")\n").format(jj,1, flag,"--->conformation "+smiles.split("-")[-1],rmslen, rmsene, avglen, mrmslen, mrmsene, *[term for term in vals]))
                            avglen = valence_term[index_key].mean()
                            rmslen = np.linalg.norm(valence_term[index_key] - r0)
                            rmsene = np.linalg.norm(force_k/2 * (valence_term[index_key] - r0)**2)
                            mrmslen = np.linalg.norm(valence_term[index_key] - avglen)
                            mrmsene = np.linalg.norm(force_k/2 * (valence_term[index_key] - avglen)**2)

                            flag = "G"
                            if((valence_term[index_key] < (r0 - delta)).any() or (valence_term[index_key] > (r0 + delta)).any()):
                                flag = "R"
                            elif(valence_term[index_key].max() < r0 or valence_term[index_key].min() > r0):
                                flag = "Y"
                            outstr.append((" {:4d} {:5d} {:>4s} {:60s} " + "kRMS(L)=" + "{:7.4f} "*1 + "kRMS(E)={:7.4f} " + "meanL={:7.4f} mRMS(L)=" + "{:7.4f} "*1 + "mRMS(E)={:7.4f} " + "\n").format(jj,len(smiles_hits), flag,"==>atoms " + str(index_key),rmslen, rmsene, avglen, mrmslen, mrmsene))
                            [outstr.append(term) for term in single_terms]
                            dat = np.append(dat, valence_term[index_key])
                                    
                                    #if(measure == "Angles"):
                                    #    pass
                                        #dat *= np.pi/180
                        if(len(dat) > 0):
                            flag = "G"
                            if((dat < (r0 - delta)).any() or (dat > (r0 + delta)).any()):
                                flag = "R"
                            elif(dat.max() < r0 or dat.min() > r0):
                                flag = "Y"
                    #fid.write(("{:8d} {:1s} {:s}" + "{:6.3f}"*len(mdatamean[smiles][label]) + "\n").format(idx,flag,smiles,*mdatamean[smiles][label])) 
                            #print(params, end="\n\n")
                            #full_name_categories = [p for p_list in params for p in p_list]
                            #full_name_categories = [p for p in params]
                            #print(full_name_categories)
                            #full_name_components = [x for y in [(p, m['oFF'][p]['smirks']) for p in full_name_categories] for x in y]
                            avglen = dat.mean()
                            rmslen = np.linalg.norm(dat - r0)
                            rmsene = np.linalg.norm(force_k/2 * (dat - r0)**2)
                            mrmslen = np.linalg.norm(dat - avglen)
                            mrmsene = np.linalg.norm(force_k/2 * (dat - avglen)**2)

                            #print()
                            #print(dat)
                            #print("label", index_dat['oFF'], "r0", r0, "delta", delta, "max", dat.max(), "kRMS(Len)", rmslen, "kRMS(Ene)", rmsene, flag)
                            fid.write((" {:4d} {:5d} {:>4s} {:60s} " + "kRMS(L)=" + "{:7.4f} "*1 + "kRMS(E)={:7.4f} " + "meanL={:7.4f} mRMS(L)=" + "{:7.4f} "*1 + "mRMS(E)={:7.4f} " + "\n").format(jj,len(smiles_hits), flag,"|>molecule " + "".join(smiles.split("-")[:-1]),rmslen, rmsene, avglen, mrmslen, mrmsene)) 
                            [fid.write(s) for s in outstr]
                #plt.suptitle(("frag={:s} " + ("{:s}: {:s} "*len(param_list))).format(fragstr,*full_name_components))
                fragstr = "all"
            if("fragment" in m):
                fragstr = m["fragment"]
            plt.suptitle(("frag={:s} " + "{:s}").format(fragstr,title))

            savefig("fig." + str(jj) + "." + param_str +"png")
        print()
