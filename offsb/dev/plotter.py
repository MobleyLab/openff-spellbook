
import matplotlib.pyplot as plt

def flag_measurements(fname='bonds_N-C_deg-2.npz', mindb=None, groups=None, minfname='bonds_N-C_deg-2.minimum.npz', params={"Mol": {}, "Bonds": ['b1'], "Angles": [], "ImproperTorsions": [], "ProperTorsions": [], 'Energy': {}} ):
    """
    energy is a list of keys to search for energy, example: {'oFF'; 'vdW'}. Plotted energies are relative to the min value.
    """
    d = None
    m = None
    rms_str="ffRMS(L)= {:9.4e} ffRMS(E)= {:9.4e} measL= {:9.4e} measRMS(L)= {:9.4e} measRMS(E)= {:9.4e} "
    rms_str="{:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} "


    ene_str="{:s}: meanE= {:9.4f} RMS(E)= {:9.4f} maxAngEne= {:9.4f} {:9.4f}"
    ene_maxdel_str="DmeanE= {:9.4f} maxDiffAngEne= {:9.4f} {:9.4f} maxGapAngEne {:9.4f} {:9.4f}"
    ref_ene_key = 'qm'

    index = None
    if os.path.exists("index.txt"):
        with open("index.txt", 'r') as fid:
            index = dict([line.strip('\n').split()[::-1] for line in fid.readlines()])
    elif os.path.exists(os.path.join("..","index.txt")):
        with open(os.path.join("..","index.txt"), 'r') as fid:
            index = dict([line.strip('\n').split()[::-1] for line in fid.readlines()])
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


    if("Mol" not in params):
        params["Mol"] = {}
    params_new = collections.defaultdict(list)
    params_new.update(params)
    params = params_new
    params_new = None
    measurements = ["Bonds", "Angles", "ImproperTorsions", "ProperTorsions", "Energy"]
    colors = ['red', 'blue', 'purple', 'green', 'orange', 'yellow']
    rows = 1
    hasbonds = int(len(params["Bonds"]) > 0)
    hasangles = int(len(params["Angles"]) + len(params["ImproperTorsions"]) + len(params["ProperTorsions"])  > 0)
    hasenergy = int(len(params["Energy"]) > 0)
    rows = hasbonds + hasangles + hasenergy
    logger.debug("VAR: rows= " + str(rows))

    mol_list = params['Mol']
    if(mol_list == {}):
        vals = list(m["mol_data"].keys())
        mol_list = dict(zip(range(len(vals)), vals))

    param_list = [p for p_list in measurements for p in params[p_list]]
    ene_out_fname = ("ene."+"{:s}."*len(param_list)+"txt").format(*param_list)
    fid =  open(ene_out_fname, 'w') ; fid.close()
    # this is looping through each molecule
    for jj, (name, smiles_list) in enumerate(mol_list.items()):
        print("{:4d} {:4d} {:64s}:".format(jj,int(index[name]),name), end=" ")
        hits=0

        fig = plt.figure(figsize=(8,4*rows),dpi=120)
        logger.debug("fig created id " + str(id(fig)))
        ax_grid = [] #[[]]*rows
        for r in range(rows):
            logger.debug("Init row {} for axes\n".format(r))
            ax = [plt.subplot2grid((rows,3),(r,0), colspan=2, fig=fig)]
            ax.append(plt.subplot2grid((rows,3),(r,2), fig=fig, sharey=ax[0]))
            logger.debug("ax left  {} ax right {}\n".format(id(ax[0]), id(ax[1])))
            ax_grid.append(ax)
        logger.debug("axes look like\n{}\n".format(str(ax_grid)))
        checks = [[["Bonds"], hasbonds], \
                  [["Angles", "ProperTorsions", "ImproperTorsions"], hasangles],\
                  [["Energy"], hasenergy]]
        present = 0
        plot_idx = {}
        for ncheck_i, check_i in enumerate(checks):
            if(check_i[1]):
                for param in check_i[0]:
                    plot_idx[param] = present
                present += 1
        logger.debug("Will plot using {} axes\n".format(present))
        logger.debug(str(plot_idx))

        fig.subplots_adjust(wspace=.3, hspace=.2,right=.95)
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

        nonempty_ene = False
        # this is looping through each conformation of the molecule
        for ii,smiles in enumerate(smiles_list):
            all_params = []
            skip=False
            if(len(params["Energy"].keys()) > 0):
                for ene_group in params["Energy"]:
                    if("energy" not in m["mol_data"][smiles]):
                        logger.debug("SKIP 1")
                        skip=True
                        break
                    if(ene_group == 'qm'):
                        if('qm' not in m["mol_data"][smiles]["energy"]):
                            logger.debug("SKIP 2")
                            skip=True
                            break
                    else:
                        for ene_type in params["Energy"][ene_group]:
                            if(ene_type not in m["mol_data"][smiles]["energy"][ene_group]):
                                logger.debug("SKIP 3")
                                skip=True
                                break
                            if(skip): break
                if(skip): break
            if(skip): break
            for measure in measurements:
                if(measure == "Energy"):
                    continue
                try:
                    all_params += [p['oFF'] for p in m["mol_data"][smiles][measure]["indices"].values()]
                except KeyError:
                    print("Mol with smiles", smiles, "empty or corrupted (missing", measure, ". Skipping")
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
                    ax2.hist(j,bins=10, color='b', orientation='horizontal')
                    ax2.hist(jmin,bins=10, color='k', orientation='horizontal')
            else:
                for measure_ii, measure in enumerate(measurements):
                    logger.debug("VAR: measure= " + str(measure))
                    if(measure not in params):
                        logger.debug("Not in params so skipping: " + str(measure))
                        continue
                    if(len(params[measure]) == 0):
                        logger.debug("Nothing in params for: " + str(measure) + " so skipping")
                        continue
                    if(measure != "Energy"):
                        for i,(index_key,index_dat) in enumerate(m["mol_data"][smiles][measure]["indices"].items()):
                            label = index_dat['oFF']
                            #print(ii, i, smiles, jmin.mean(), end=" ")
                            plot_label=None
                            if(not (label in params[measure])):
                                logger.debug("This param not wanted so skipping: " + str(label))
                                #print("param", label, " not wanted. skipping")
                                continue
                            logger.debug("Continuing to plot for : " + str(label))
                            hits += 1
                            #print(index_key)
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
                            #if(measure == "Angles"):
                            #    pass
                                #measure_data *= np.pi/180
                            #print(plot_idx, "plotting", label, "td_ang=", td_ang)

                            if(td_ang[0] is not None):
                                logger.debug("***PLOTTING {:s} to ax {:s} id {}\n".format( str(measure), str(plot_idx[measure]), id(ax_grid[plot_idx[measure]][0])))
                                ax_grid[plot_idx[measure]][0].plot(td_ang, measure_data, lw=.1, ls='-', marker='.' , color=color, label=plot_label, ms=2)
                                ax_grid[plot_idx[measure]][0].legend(loc='upper right')
                            if(label not in mdata):
                                mdata[label] = []
                            mdata[label] += list(measure_data)

                            mdatamean.setdefault(smiles, {})
                            mdatamean[smiles].setdefault(label, [])
                            mdatamean[smiles][label].append(measure_data.mean())
                    else:
                        for ene_group in params['Energy']:
                            logger.debug("VAR: ene_group=" + str(ene_group))
                            if(ene_group == 'qm'):
                                c_idx += 1
                                used_colors[ene_group] = colors[c_idx]
                                label = ene_group
                                color = used_colors[label]
                                ene = np.array(m["mol_data"][smiles]['energy'][ene_group]) * hartree2kcalmol
                                ene -= ene.min()
                                td_ang = m["td_ang"][m["mol_data"][smiles]["td_ang"]]
                                if(td_ang[0] is not None):
                                    logger.debug("plotting to idx" + str(plot_idx[measure]) + " for measure " + str(measure) )
                                    logger.debug("***PLOTTING {:s} to ax {:s} id {}\n".format( str(measure), str(plot_idx[measure]), id(ax_grid[plot_idx[measure]][0])))
                                    ax_grid[plot_idx[measure]][0].plot(td_ang, ene, lw=1.5, ls='-', marker='.', ms=4, color=color, label=label)
                                    ax_grid[plot_idx[measure]][0].set_ylabel("Energy (kcal/mol)")
                                    nonempty_ene = True

                                if(label not in mdata):
                                    mdata[label] = []
                                mdata[label] += list(ene)

                                mdatamean.setdefault(smiles, {})
                                mdatamean[smiles].setdefault(label, [])
                                mdatamean[smiles][label].append(ene.mean())
                            else:
                                logger.debug("VAR: ene_types=" + str(list(params["Energy"].keys())))
                                for ene_type in params["Energy"][ene_group]:
                                    logger.debug("VAR: ene_type=" + str(ene_type))
                                    ene = m["mol_data"][smiles]['energy'][ene_group][ene_type]
                                    if(len(ene) > 0 and isinstance(ene[0], simtk.unit.Quantity)):
                                        ene = np.array([x.value_in_unit(x.unit) for x in ene])
                                        ene -= ene.min()
                                    else:
                                        ene = np.array(ene)
                                        ene -= ene.min()
                                    label = ".".join((ene_group, ene_type))
                                    c_idx += 1
                                    used_colors[label] = colors[c_idx % len(colors)]
                                    color = used_colors[label]
                                    td_ang = m["td_ang"][m["mol_data"][smiles]["td_ang"]]
                                    if(td_ang[0] is not None):
                                        logger.debug("plotting to idx " + str(plot_idx[measure]) + " for measure " + str(measure) )
                                        ax_grid[plot_idx[measure]][0].plot(td_ang, ene, lw=1.5, ls='-', marker='.', ms=4, color=color, label=label)
                                        ax_grid[plot_idx[measure]][0].set_ylabel("Energy (kcal/mol)")
                                        nonempty_ene = True
                                    if(label not in mdata):
                                        mdata[label] = []
                                    mdata[label] += list(ene)

                                    mdatamean.setdefault(smiles, {})
                                    mdatamean[smiles].setdefault(label, [])
                                    mdatamean[smiles][label].append(ene.mean())
                        if(nonempty_ene):
                            ax_grid[plot_idx[measure]][0].legend(loc='upper left')
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
        print("HITS= {:7.1f}".format(hits/len(smiles_list)), end=" ")
        for p in param_list:
            if( p in ["a0", "b0", "i0", "t0"] ):
                m['oFF'][p]['smirks'] = "None"
        logger.debug("hits or nonempty? " + str(hits > 0 or nonempty_ene)) 
        if(hits > 0 or nonempty_ene):
            smiles_idx_str = ("{:s}."*len(smiles_idx)).format(*[str(x) for x in smiles_idx]) 
            param_str = ("{:s}."*len(param_list)).format(*[str(x) for x in param_list]) 
            if(len(ddata) > 0):
                ax2.hist(ddata,bins=20, color='blue', orientation='horizontal')
            kT = (.001987*298.15)
            for ii,(label,dat) in enumerate(mdata.items()):
                #if(label[0] in "ait"):
                #    #num *= np.pi/180
                #    if(rows == 2):
                #        plot_idx = 1
                #    else:
                #        plot_idx = 0
                plot_row = -1
                if(label[0] in "ait"):
                    plot_row = plot_idx["Angles"]
                elif(label[0] == "b"):
                    plot_row = plot_idx["Bonds"]
                else:
                    plot_row = plot_idx["Energy"]
                logger.debug("VAR: plot_row=" + str(plot_row))
                color=used_colors[label]
                ax_grid[plot_row][1].hist(dat,bins=20, color=used_colors[label], histtype='step', orientation='horizontal')
                if( label in ["a0", "b0", "t0", "i0"] ):
                    continue
                # TODO: calculate spread of torsions
                if(label[0] not in 'ab'):
                    continue
                num = float(str(bond_r0[label]).split()[0])
                force_k = float(str(bond_k[label]).split()[0])
                delta = (2*(kT)/force_k)**.5
                if(label[0] in "ait"):
                    delta *= 180/np.pi
                dat = np.array(dat)
                if((dat < (num - delta)).any() or (dat > (num + delta)).any()):
                    print(label + "= R", end=" ")
                elif(dat.max() < num or dat.min() > num):
                    print(label + "= Y", end=" ")
                else:
                    print(label + "= G", end=" ")
                ax_grid[plot_row][0].axhline(y=num, ls='-', marker='.', color='black', ms=20, mec='black', mfc=color)
                ax_grid[plot_row][0].axhline(y=num + delta, ls='--', marker='.', color='black', ms=10, mec='black', mfc=color)
                ax_grid[plot_row][0].axhline(y=num - delta, ls='--', marker='.', color='black', ms=10, mec='black', mfc=color)
                ax_grid[plot_row][1].axhline(y=num, ls='-', marker='.', color='black', ms=20, mec='black', mfc=color)
                ax_grid[plot_row][1].axhline(y=num + delta, ls='--', marker='.', color='black', ms=10, mec='black', mfc=color)
                ax_grid[plot_row][1].axhline(y=num - delta, ls='--', marker='.', color='black', ms=10, mec='black', mfc=color)
                
            #ax[0].legend()
            #if(rows > 1):
            #    ax[1].legend()
            print_header = True
            smiles_out_fname = ("mol." + str(index[smiles]) + ".smiles_with."+"{:s}."*len(param_list)+"txt").format(*param_list)
            #if(os.path.exists(smiles_out_fname)):
            #    print_header = False
            with open(smiles_out_fname, 'w') as fid:
                for measure in measurements:
                    mol_label_count = 0
                    for label in params[measure]:
                        if((label not in bond_r0) ):
                            continue
                        label_count = 0
                        #r0 = float(str(bond_r0[label]).split()[0])
                        r0 = bond_r0[label] / bond_r0[label].unit
                        force_k = bond_k[label] / bond_k[label].unit
                        delta = 2*kT/force_k**.5
                        smirks = bond_dict[label]['smirks']
                            #delta *= 180/np.pi
                            #pass
                        if(measure in ["Angles", "ImproperTorsions", "ProperTorsions"] ):
                            delta = 2*kT/(force_k * (np.pi/180)**2 )**.5
                        if(print_header):
                            fid.write("# {:3s} {:24s} \n".format(label, smirks))
                            fid.write("# r0 = {:6.2f}, k = {:6.2f} kT-> {:6.2f}\n".format(r0, force_k, delta))
                            fid.write("#{:4s} {:5s} {:4s} {:60s} {:10s} {:10s} {:10s} {:10s} {:10s}\n".format("idx", "count", "flag", "category", "ffRMS(L)", "ffRMS(E)", "measL", "measRMS(L)", "measRMS(E)"))

                        if(measure in ["Angles", "ImproperTorsions", "ProperTorsions"] ):
                            force_k = force_k * (np.pi/180)**2 # put into degrees

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
                            index_count = 0
                            for idx,smiles in zip(smiles_idx,smiles_hits):
                                try:
                                    col_idx = m["mol_data"][smiles][measure]["indices"][index_key]["column_idx"]
                                except KeyError as e:
                                    logger.warning("\n** Missing" + str(e) + ": Probably different molecules with same smiles. Check the output! ** ")
                                    continue
                                vals = np.atleast_1d(m["mol_data"][smiles][measure]["values"][:,col_idx])
                                valence_term[index_key] = np.append(valence_term[index_key], vals)
                                flag = "G"
                                if((vals < (r0 - delta)).any() or (vals > (r0 + delta)).any()):
                                    flag = "R"
                                elif(vals.max() < r0 or vals.min() > r0):
                                    flag = "Y"
                                avglen = vals.mean()
                                rmslen = rms(vals - r0)
                                rmsene = rms(force_k/2 * (vals - r0)**2)
                                mrmslen = rms(vals - avglen)
                                mrmsene = rms(force_k/2 * (vals - avglen)**2)
                                single_terms.append((" {:4d} {:5d} {:>4s} {:60s} " + rms_str + "\n").format(jj,len(vals), flag,"--->conformation "+ get_conformation_number(smiles), rmslen, rmsene, avglen, mrmslen, mrmsene))
                                if(vals.size > 1):
                                    for kk,val in enumerate(vals):
                                        flag = "G"
                                        if((val < (r0 - delta)) or (val > (r0 + delta))):
                                            flag = "R"
                                        elif(val < r0 or val > r0):
                                            flag = "X"
                                        avglen = val
                                        rmslen = rms(val - r0)
                                        rmsene = rms(force_k/2 * (val - r0)**2)
                                        mrmslen = rms(val - avglen)
                                        mrmsene = rms(force_k/2 * (val - avglen)**2)
                                        single_terms.append((" {:4d} {:5d} {:>4s} {:60s} " + rms_str + "\n").format(jj,1, flag,".....>intermediate "+str(kk),rmslen, rmsene, avglen, mrmslen, mrmsene))
                                index_count += len(vals)
                            avglen = valence_term[index_key].mean()
                            rmslen = rms(valence_term[index_key] - r0)
                            rmsene = rms(force_k/2 * (valence_term[index_key] - r0)**2)
                            mrmslen = rms(valence_term[index_key] - avglen)
                            mrmsene = rms(force_k/2 * (valence_term[index_key] - avglen)**2)

                            flag = "G"
                            if((valence_term[index_key] < (r0 - delta)).any() or (valence_term[index_key] > (r0 + delta)).any()):
                                flag = "R"
                            elif(valence_term[index_key].max() < r0 or valence_term[index_key].min() > r0):
                                flag = "Y"
                            outstr.append((" {:4d} {:5d} {:>4s} {:60s} " + rms_str + "\n").format(jj, index_count, flag,"==>atoms " + str(index_key),rmslen, rmsene, avglen, mrmslen, mrmsene))
                            [outstr.append(term) for term in single_terms]
                            dat = np.append(dat, valence_term[index_key])
                                    
                                    #if(measure == "Angles"):
                                    #    pass
                                        #dat *= np.pi/180

                        mol_label_count += len(dat)
                        if(len(dat) > 0):
                            flag = "G"
                            if((dat < (r0 - delta)).any() or (dat > (r0 + delta)).any()):
                                flag = "R"
                            elif(dat.max() < r0 or dat.min() > r0):
                                flag = "Y"
                            avglen = dat.mean()
                            rmslen = rms(dat - r0)
                            rmsene = rms(force_k/2 * (dat - r0)**2)
                            mrmslen = rms(dat - avglen)
                            mrmsene = rms(force_k/2 * (dat - avglen)**2)

                            fid.write((" {:4d} {:5d} {:>4s} {:60s} " + rms_str + "\n").format(jj,mol_label_count, flag,"|>molecule " + strip_conformation_number(smiles), rmslen, rmsene, avglen, mrmslen, mrmsene)) 
                            [fid.write(s) for s in outstr]
                            print(rms_str.format(rmslen, rmsene, avglen, mrmslen, mrmsene), end=" ")
                fragstr = "all"





#    ene_str="{:s}: meanE= {:9.4f} RMS(E)= {:9.4} maxAngEne= {:9.4f} {:9.4f}"
#    ene_maxdel_str="DmeanE= {:9.4f} maxDiffAngEne= {:9.4f} {:9.4f} maxGapAngEne {:9.4f} {:9.4f}"
#    ref_ene_key = 'qm'
#           
            if len(params["Energy"]) > 1:
                with open(ene_out_fname, 'a') as fid:
                    measure = "Energy"
                    mol_label_count = 0
                    ref_ene = np.array(mdata[ref_ene_key])
                    logger.debug("VAR: ref_ene= " + str(ref_ene))
                    ref_ene_max_idx = ref_ene.argmax()
                    td_ang = m["td_ang"][m["mol_data"][smiles]["td_ang"]]
                    if(td_ang[0] == None):
                        continue
                    fid.write((" {:4d} {:5d} {:>4s} {:60s} " + ene_str + "\n").format(jj,len(ref_ene), "REF","|>molecule " + strip_conformation_number(smiles),ref_ene_key, ref_ene.mean(), rms(ref_ene - ref_ene.mean()), td_ang[ref_ene_max_idx], ref_ene[ref_ene_max_idx])) 
                    ene_list = {x:y for x,y in params["Energy"].items() if x != "qm"}
                    for ene_group in ene_list:
                        for ene_type in ene_list[ene_group]:
                            label = ".".join((ene_group, ene_type))
                            ene = np.array(mdata[label])
                            ene_max_idx = ene.argmax()
                            delta = ene - ref_ene
                            delta_max_idx = np.abs(delta).argmax()
                            fid.write((" {:4d} {:5d} {:>4s} {:60s} " + ene_str + " " + ene_maxdel_str +"\n").format(jj,len(ene), "","==> " + label, "", ene.mean(), rms(ene - ene.mean()), td_ang[ene_max_idx], ene[ene_max_idx], ene.mean() - ref_ene.mean(), td_ang[ene_max_idx] - td_ang[ref_ene_max_idx], ene[ene_max_idx] - ref_ene[ref_ene_max_idx], td_ang[delta_max_idx], delta[delta_max_idx])) 


                    # need argmax of ref for angle and ene 
                    # need mean angle
    #                for label in params[measure]:
    #                    if(("qm" not in label) or ("oFF" not in label)):
    #                        continue
    #                    label_count = 0
    #                    #r0 = float(str(bond_r0[label]).split()[0])
    #                    
    #                        #delta *= 180/np.pi
    #                        #pass
    #                    if(print_header):
    #                        fid.write("#{:4s} {:5s} {:4s} {:50s} {:10s} {:10s} {:10s} {:10s} {:10s}\n".format("idx", "count", "flag", "category", "ffRMS(L)", "ffRMS(E)", "measL", "measRMS(L)", "measRMS(E)"))
    #                    
    #                    #need argmax of angle and ene
    #                    # need argmax of different between data and ref
    #
    #                    # have a reference ene (the qm)
    #                    dat = []
    #                    outstr = []
    #                    per_term_flag = "X"
    #                    flag = "X"
    #                    bond_indices = []
    #                    valence_term = {}
    #
    #
    #                        single_terms = []
    #                        index_count = 0
    #                        for idx,smiles in zip(smiles_idx,smiles_hits):
    #                            try:
    #                                col_idx = m["mol_data"][smiles][measure]["indices"][index_key]["column_idx"]
    #                            except KeyError as e:
    #                                logger.warning("\n** Missing" + str(e) + ": Probably different molecules with same smiles. Check the output! ** ")
    #                                continue
    #                            vals = np.atleast_1d(m["mol_data"][smiles][measure]["values"][:,col_idx])
    #                            valence_term[index_key] = np.append(valence_term[index_key], vals)
    #                            flag = "G"
    #                            if((vals < (r0 - delta)).any() or (vals > (r0 + delta)).any()):
    #                                flag = "R"
    #                            elif(vals.max() < r0 or vals.min() > r0):
    #                                flag = "Y"
    #                            avglen = vals.mean()
    #                            rmslen = rms(vals - r0)
    #                            rmsene = rms(force_k/2 * (vals - r0)**2)
    #                            mrmslen = rms(vals - avglen)
    #                            mrmsene = rms(force_k/2 * (vals - avglen)**2)
    #                            single_terms.append((" {:4d} {:5d} {:>4s} {:50s} " + rms_str + "\n").format(jj,len(vals), flag,"--->conformation "+ get_conformation_number(smiles), rmslen, rmsene, avglen, mrmslen, mrmsene))
    #                            if(vals.size > 1):
    #                                for kk,val in enumerate(vals):
    #                                    flag = "G"
    #                                    if((val < (r0 - delta)) or (val > (r0 + delta))):
    #                                        flag = "R"
    #                                    elif(val < r0 or val > r0):
    #                                        flag = "X"
    #                                    avglen = val
    #                                    rmslen = rms(val - r0)
    #                                    rmsene = rms(force_k/2 * (val - r0)**2)
    #                                    mrmslen = rms(val - avglen)
    #                                    mrmsene = rms(force_k/2 * (val - avglen)**2)
    #                                    single_terms.append((" {:4d} {:5d} {:>4s} {:50s} " + rms_str + "\n").format(jj,1, flag,".....>intermediate "+str(kk),rmslen, rmsene, avglen, mrmslen, mrmsene))
    #                            index_count += len(vals)
    #                        avglen = valence_term[index_key].mean()
    #                        rmslen = rms(valence_term[index_key] - r0)
    #                        rmsene = rms(force_k/2 * (valence_term[index_key] - r0)**2)
    #                        mrmslen = rms(valence_term[index_key] - avglen)
    #                        mrmsene = rms(force_k/2 * (valence_term[index_key] - avglen)**2)
    #
    #                        flag = "G"
    #                        if((valence_term[index_key] < (r0 - delta)).any() or (valence_term[index_key] > (r0 + delta)).any()):
    #                            flag = "R"
    #                        elif(valence_term[index_key].max() < r0 or valence_term[index_key].min() > r0):
    #                            flag = "Y"
    #                        outstr.append((" {:4d} {:5d} {:>4s} {:50s} " + rms_str + "\n").format(jj, index_count, flag,"==>atoms " + str(index_key),rmslen, rmsene, avglen, mrmslen, mrmsene))
    #                        [outstr.append(term) for term in single_terms]
    #                        dat = np.append(dat, valence_term[index_key])
    #                                
    #                                #if(measure == "Angles"):
    #                                #    pass
    #                                    #dat *= np.pi/180
    #
    #                    mol_label_count += len(dat)
    #                    if(len(dat) > 0):
    #                        flag = "G"
    #                        if((dat < (r0 - delta)).any() or (dat > (r0 + delta)).any()):
    #                            flag = "R"
    #                        elif(dat.max() < r0 or dat.min() > r0):
    #                            flag = "Y"
    #                        avglen = dat.mean()
    #                        rmslen = rms(dat - r0)
    #                        rmsene = rms(force_k/2 * (dat - r0)**2)
    #                        mrmslen = rms(dat - avglen)
    #                        mrmsene = rms(force_k/2 * (dat - avglen)**2)
    #
    #                        fid.write((" {:4d} {:5d} {:>4s} {:50s} " + rms_str + "\n").format(jj,mol_label_count, flag,"|>molecule " + strip_conformation_number(smiles), rmslen, rmsene, avglen, mrmslen, mrmsene)) 
    #                        [fid.write(s) for s in outstr]
    #                        print(rms_str.format(rmslen, rmsene, avglen, mrmslen, mrmsene), end=" ")




            if("fragment" in m):
                fragstr = m["fragment"]
            fig.suptitle(("frag={:s} " + "{:s}").format(fragstr,smiles))

            fig.savefig("fig.mol_" + str(index[smiles]) + "." + param_str +"png")
        plt.close(fig)
        print()
