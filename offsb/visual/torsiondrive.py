#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.gridspec import GridSpec
import matplotlib
import sys
import re
import os

from offsb.tools import const

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import DrawingOptions

import pickle
import offsb.rdutil as rdutil

PREFIX="."

HEADER_LEN=4


class TorsionPlotComparitor():

    """
    Convenience plotter that compares two values over a given domain

    >>> tdplot = TorsionPlotComparitor()
    >>> x1,y1 = [[0.0,1.0],[3.0,4.5]]
    >>> tdplot.add_set_reference(x1, y1, label="QM")
    >>> x2,y2 = [[0.0,1.0],[3.2,4.1]]
    >>> tdplot.add_set_comparison(x2, y2, label="MM")
    >>> tdplot.show()
    >>> tdplot.savefig("td.png")

    """
    
    def __init__(self):
        self._datasets = {}
        self._labels = {}
        self._dataset_name = ""
        self._entry_name = ""
        self._finalized = False
        self._mol_image = None
        self._mol_charge = 0.0
        self._fig = None
        self._reference_dataset_name = None
        self._comparison_dataset_name = None

        self._reference_dataset_spec = "unknown"
        self._comparison_dataset_spec = "unknown"

        self._infolines = None

    def show(self):
        if self.finalize():
            self.plot()
            return self._fig.show()

    @property
    def dataset_name(self):
        return self._dataset_name
    
    @dataset_name.setter
    def dataset_name(self, dataset_name):
        self._dataset_name = dataset_name

    @property
    def entry_name(self):
        return self._dataset_name
    
    @entry_name.setter
    def entry_name(self, entry_name):
        self._entry_name = entry_name

    @property
    def mol_image(self, img):
        return self._mol_image

    @mol_image.setter
    def mol_image(self, img):
        self._mol_image = img

    @property
    def reference_dataset_name(self):
        return self._reference_dataset_name

    @reference_dataset_name.setter
    def reference_dataset_name(self, name):
        self._reference_dataset_name = name

    @property
    def comparison_dataset_name(self):
        return self._comparison_dataset_name

    @comparison_dataset_name.setter
    def comparison_dataset_name(self, name):
        self._comparison_dataset_name = name

    @property
    def mol_charge(self):
        return self._mol_charge

    @mol_charge.setter
    def mol_charge(self, name):
        self._mol_charge = name

    def add_set_comparison(self, x, y, name=None, label=None):

        key = self.add(x,y,name,label)
        self._comparison_dataset_name = key
        return key

    def add_set_reference(self, x, y, name=None, label=None):

        key = self.add(x,y,name,label)
        self._reference_dataset_name = key
        return key

    def add(self, x,y, name=None, label=None):

        if len(x) != len(y):
            raise Exception("x and y are different lengths")

        if name is None:
            name = str(len(self._datasets))

        self._datasets[name] = [x,y]

        if label is not None:
            self._labels[name] = label

        self._finalized = False

        return name

    def pop(self, key):
        ret = self._datasets.pop(key, None)
        self._labels.pop(key, None)
        self._finalized = ret is None
        return ret

    def finalize(self):
        """
        Calculate plot properties using the data added
        """
        if not self._finalized:
            self._finalized = self._finalize()

        return self._finalized

    def _plot_one_dataset(self, name, ax):

        label=self._labels[name]
        (x, y) = self._datasets[name]
        print("plotting", name)
        ax.plot(x, y, lw=1, ms=2, label=label)


    def plot(self):
        self.finalize()

        ax1, ax2, ax3 = self._fig.axes
        assert ax1 is not None
        

        name = self._reference_dataset_name
        self._plot_one_dataset(name, ax1)

        name = self._comparison_dataset_name
        self._plot_one_dataset(name, ax1)

        ax1.legend(loc='best')

        if self._mol_image is None:
            print("Molecule image was not set, skipping")
        else:
            assert ax2 is not None
            ax2.imshow(self._mol_image)

    def savefig(self, fnm):
        self._fig.savefig(fnm)

    def _add_text(self, ax, x, y, line, opts):
        ax.text(x, y, line, **opts)

    def _initialize_figure(self):

        if self._fig is None:
            self._fig = plt.figure(
                figsize=(8,3.5),
                dpi=120,
                constrained_layout=True
            )
            print(self._fig)
        else:
            self._fig.clf()

        gs = GridSpec(2, 2, figure=self._fig)
        ax1 = self._fig.add_subplot(gs[:2,0])
        ax2 = self._fig.add_subplot(gs[0,1])
        ax3 = self._fig.add_subplot(gs[1,1])

        ax1.set_ylabel("Energy (kcal/mol)")
        ax1.set_xlabel("Scan value")
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax2.axis('off')
        shared_opts = dict(fontsize=8, fontname="Monospace")

        x = 0

        ref_abs_min = min(self._ref_dataset[1])

        ref_min_pt = min(self._ref_dataset[0])
        cmp_min_pt = min(self._cmp_dataset[0])

        ds_name = self._dataset_name
        entry_name = self._entry_name

        total_charge = self._mol_charge
        spec = self._reference_dataset_spec

        if self._infolines is not None:

        line_fmt = "{:<16s}{:>39s}"
        for y, pair in zip(np.arange(0,1,0.1)[::-1], lines):
            line = line_fmt.format(*pair)
            self._add_text(ax3, x, y, line, shared_opts)

        ax3.axis('off')

    def _set_data_limits(self, datasets):

        dat = [x_i for X in datasets[0] for x_i in X]
        self.xlims = [np.min(dat), np.max(dat)]

        dat = [x_i for X in datasets[1] for x_i in X]
        self.ylims = [np.min(dat), np.max(dat)]
        
    def _finalize(self):

        print("Finalizing plot")

        ref_name = self._reference_dataset_name
        cmp_name = self._comparison_dataset_name

        keys = [ref_name, cmp_name]
        print("ref/cmp keys are", keys)
        datasets = list(self._datasets[k] for k in keys)
        print("datasets are", datasets)
        self._set_data_limits(datasets)

        self._ref_dataset = self._datasets[ref_name]
        self._cmp_dataset = self._datasets[cmp_name]

        self._initialize_figure()

        return True

def QCArchiveTorsionPlotComparitor(TorsionPlotComparitor):

    def __init__(self):
        super().__init__()
        self.min_only = True

        self._infolines =  (
            ("QCA Dataset:", ds_name),
            ("QCA Entry:", entry_name),
            ("Cmp spec:", "Parsley unconstrained"),
            ("Cmp charges:", "ANTECHAMBER AM1-BCC"),
            ("Total charge", "{:6.2f}".format(total_charge)),
            ("Ref spec:", "{:s}".format(spec)),
            ("Ref Absolute min:", "{:16.13e} au".format( ref_abs_min)),
            ("Ref Min point:", "{:8.2f}".format(ref_min_pt)),
            ("Cmp Min point:", "{:8.2f}".format(cmp_min_pt)),
        )
            #("OFFSB ver", "1.0")

        def add_set_reference_procedure(self, td):
            
            pass

        def add_set_comparison_procedure(self, td):

            pass




def gen_plot( QCA, fnm):

    try:
        d = np.loadtxt( fnm, usecols=(5,2,6), skiprows=3)
        if d.size == 0:
            return
    except StopIteration:
        print("No data for", fnm)
        return

    entry   = np.genfromtxt( fnm, usecols=(0,),  skip_header=HEADER_LEN, dtype=np.dtype("str"))[0]
    smiles  = np.genfromtxt( fnm, usecols=(7,),  skip_header=HEADER_LEN, dtype=np.dtype("str"), comments=None)[0]
    scanned = np.genfromtxt( fnm, usecols=( 4,), skip_header=HEADER_LEN, dtype=np.dtype("str"))[0]
    scanned = eval('[' + scanned.replace('-',',') + ']')
    scanned = [x+1 for x in scanned]
    #print(scanned)
    print( fnm, smiles)
    mol_id = np.genfromtxt( fnm, usecols=(1,), skip_header=HEADER_LEN, dtype=np.dtype("str"))

    with open( fnm, 'r') as fd:
        ds_name = fd.readline().strip("\n")
        method  = fd.readline().strip("\n")
        basis   = fd.readline().strip("\n")

    scan = d[:,0]
    qm_abs_min = d[:,1].min()
    qmene = d[:,1] * const.hartree2kcalmol
    qmene -= qmene.min()
    mmene = d[:,2] 
    mmene -= mmene.min()

    mm_min_pt = scan[ np.argsort(mmene)[0]]
    qm_min_pt = scan[ np.argsort(qmene)[0]]

    #AllChem.GenerateDepictionMatching3DStructure()

    qcmol = QCA.db[mol_id[0]]["data"]
    try:
        mol = rdutil.mol.build_from_smiles( smiles)
        molnoindex = rdutil.mol.build_from_smiles( re.sub(":[1-9][0-9]*", "", smiles))
    except Exception as e:
        print(e)
        return

    AllChem.ComputeGasteigerCharges( mol)
    totalcharge=0.0
    for i, atom in enumerate( mol.GetAtoms()):
        totalcharge += float( atom.GetProp('_GasteigerCharge'))

    atom_map = rdutil.mol.atom_map( mol)
    inv = {val:key for key,val in atom_map.items()}
    #print(atom_map)
    scanned = [inv[i] for i in scanned]
    #scanned = [rdutil.mol.atom_map_invert( atom_map)[i] for i in scanned]
    #print(scanned)
    # molflat = rdutil.mol.build_from_smiles( smiles)
    try:
        rdutil.mol.embed_qcmol_3d( mol, qcmol)
        #print("EMBED WAS RET:", ret)
        #AllChem.GenerateDepictionMatching3DStructure(molflat, mol)
        AllChem.Compute2DCoords(molnoindex)
    except Exception as e:
        print( "Could not generate conformation:")
        print( e)
        return
    options = DrawingOptions()
    options.atomLabelFontSize = 110
    options.atomLabelFontFace = "sans"
    options.dotsPerAngstrom = 400
    options.bondLineWidth = 8
    options.coordScale = 1
    png = Draw.MolToImage(molnoindex, highlightAtoms=scanned, highlightColor=[0,.8,0], size=(500*12,500*4), fitImage=True, options=options)

    fig = plt.figure(figsize=(8,3.5), dpi=300, constrained_layout=True)
    matplotlib.rc("font", **{"size": 10})
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot( gs[:2,0])
    ax1.plot( scan, qmene, 'bo-', lw=1, ms=2, label="QM")
    ax1.plot( scan, mmene, 'ro-', lw=1, ms=2, label="MM")
    ax1.set_ylabel("Energy (kcal/mol)")
    ax1.set_xlabel("Scan value")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='best')
    ax2 = fig.add_subplot(gs[0,1])
    ax2.imshow(png)
    ax2.axis( 'off')
    ax3 = fig.add_subplot(gs[1,1])
    #ax3.text(.5,1.1,"{:^60s}".format(re.sub(":[1-9][0-9]*","",smiles)), fontsize=4, fontname="Monospace", ha='center')

    shared_opts = dict(fontsize=8, fontname="Monospace")


    def add_text(ax, x,y,line,opts):
        ax.text(x, y, line, **opts)

    x = 0
    line_fmt="{:<16s}{:>39s}"

    lines = (
        ("QCA Dataset:", ds_name),
        ("QCA Entry:", entry.split("-")[1]),
        ("MM spec:", "Parsley unconstrained"),
        ("MM charges:", "ANTECHAMBER AM1-BCC"),
        ("Total charge", "{:6.2f}".format(totalcharge)),
        ("QM spec:", "{}/{}".format(method, basis)),
        ("QM Absolute min:", "{:16.13e} au".format( qm_abs_min)),
        ("QM Min point:", "{:8.2f}".format( qm_min_pt)),
        ("MM Min point:", "{:8.2f}".format( mm_min_pt)),
        ("OFFSB ver", "1.0")
    )
    for y, pair in zip(np.arange(0,1,0.1), lines):
        add_text(ax3, x, y, line_fmt.format(*pair), shared_opts)

    ax3.axis('off')
    fig.savefig( re.sub( "\.*$", ".png", fnm))
    fig = None
    plt.close("all")

    return

def main():

    with open( os.path.join(PREFIX, 'QCA.p'), 'rb') as fid:
        QCA = pickle.load(fid)
    if QCA.db is None:
        with open( os.path.join( PREFIX, 'QCA.db.p'), 'rb') as fid:
            QCA.db = pickle.load(fid).db
    for fnm in sys.argv[1:]:
        gen_plot( QCA, os.path.join( PREFIX, fnm) )

if __name__ == "__main__":
    main()
