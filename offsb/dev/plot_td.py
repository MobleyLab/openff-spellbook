import pickle
with open('test.oFF-1.1.0.p', 'rb') as fid:
    oFF10 = pickle.load(fid)
import interfoot.tools.node as Node
import interfoot.tools.const as const
with open('QCA.p', 'rb') as fid:
    QCA = pickle.load(fid)
for entry in QCA.node_iter_entry(QCA.root):
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = {}
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, angle)
        print( {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()})
        print()
with open('bonds.p', 'rb') as fid:
    bonds = pickle.load(fid)
for entry in QCA.node_iter_entry(QCA.root):
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = {}
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, angle)
        print( {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()})
        print()
for entry in QCA.node_iter_entry(QCA.root):
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = {}
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, node.ID, angle)
        print( {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()})
        print()
for entry in QCA.node_iter_entry(QCA.root):
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = {}
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, node.ID, angle)
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        print( vals[(14,1)] )
        print()
for entry in QCA.node_iter_entry(QCA.root):
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = {}
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, end=" ")
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        print( angle[0], vals[(14,1)] )
        print()
for entry in QCA.node_iter_entry(QCA.root):
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = {}
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, end=" ")
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        print( angle[e.ID][0], vals[(14,1)] )
        print()
for entry in QCA.node_iter_entry(QCA.root):
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = {}
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, end=" ")
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        print( angle, vals[(14,1)] )
        print()
fg
for entry in QCA.node_iter_entry(QCA.root):
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = {}
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, end=" ")
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        print( list(angle.values()), vals[(14,1)] )
        print()
for entry in QCA.node_iter_entry(QCA.root):
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = {}
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, end=" ")
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        print( list(angle.values())[0][0], vals[(14,1)] )
        print()
for entry in QCA.node_iter_entry(QCA.root):
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = {}
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, end=" ")
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        print( list(angle.values())[0][0], vals[(14,1)][0] )
        print()
dat = []
for entry in QCA.node_iter_entry(QCA.root):
    dat.append( [] )
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = {}
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, end=" ")
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        dat[-1].append([list(angle.values())[0][0], vals[(14,1)][0]])
        print()
dat
dat[0]
dat[1]
dat[0]
dat[1]
dat = []
for entry in QCA.node_iter_entry(QCA.root):
    dat.append( [] )
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = {}
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, end=" ")
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        dat[-1].append([list(angle.values())[0][0], vals[(14,1)][0]])
        print()
dat = []
for entry in QCA.node_iter_entry(QCA.root):
    dat.append( [] )
    angle = {}
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle[e.ID] = eval(e.payload)
        print( entry.payload.get( "meta").name, end=" ")
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        dat[-1].append([list(angle.values())[0][0], vals[(14,1)][0]])
        print()
dat
dat[0]
dat[1]
dat[0]
dat = []
for entry in QCA.node_iter_entry(QCA.root):
    dat.append( [] )
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = -1
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle = eval(e.payload)
        print( entry.payload.get( "meta").name, end=" ")
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        dat[-1].append([angle[0], vals[(14,1)][0]])
        print()
dat
dat[0]
dat[1]
fg
dat = []
for i,entry in enumerate(QCA.node_iter_entry(QCA.root)):
    dat.append( [] )
    for node in Node.node_iter_depth_first( QCA.root, select="Molecule"):
        #entry = None
        angle = -1
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle = eval(e.payload)
        print( entry.payload.get( "meta").name, end=" ")
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        dat[i].append([angle[0], vals[(14,1)][0]])
        print()
dat[0]
dat[1]
dat[0]
dat
dat = []
for i,entry in enumerate(QCA.node_iter_entry(QCA.root)):
    dat.append( [] )
    for node in Node.node_iter_depth_first( entry, select="Molecule"):
        #entry = None
        angle = -1
        for e in Node.node_iter_to_root( node):
            if e.ID == entry.ID:
                break
            #if "meta" in e.payload:
                #entry = e
                #break
            if e.name == "Constraint":
                angle = eval(e.payload)
        print( entry.payload.get( "meta").name, end=" ")
        vals =  {x[0]: x[1]*const.bohr2angstrom for x in bonds.node_index.get( node.index).payload.items()}
        dat[i].append([angle[0], vals[(14,1)][0]])
        print()
dat
dat[0]
dat[1]
import numpy as np
import matplotlib.pyplot as plt
dat = np.array(dat)
plt.scatter(dat[0].T)
dat
plt.scatter(*np.array(dat[0]).T)
plt.clf()
plt.scatter(*np.array(dat[0]).T, 'x')
plt.scatter(*(np.array(dat[0]).T), 'x')
plt.scatter(*(np.array(dat[0]).T), ms='x')
plt.scatter(*(np.array(dat[0]).T), marker='x')
plt.scatter(*(np.array(dat[1]).T), marker='x')
%history -f plot_td.py


#################################################################################

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
