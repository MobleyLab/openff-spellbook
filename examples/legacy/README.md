
# Starting out

The aim of the example will show how to access data in QCArchive in a way that makes it easy to navigate and compose additional analysis on top of the molecules. Specifically, the example will show how to download optimizations and torsion drives, and iterate over the data intelligently. By intelligent, I mean in ways that help research, such as collecting all optimizations and torsion drives on a unique molecule.

To start, either look at the Jupyter notebooks, or run the py versions yourself locally. They should be run as follows:

1. loading_and_browsing (Note that the end, where OpenMM energy is calculated, can take a few mins; the example loads 4 torsions and 2 optimizations. The time is spent generating generating am1-bcc charges for each molecule)
2. compare_QCA_and_anaylsis
3. plot_torsiondrive.py \*.dat

Note that several files will be saved on your machine if the python files are run. These also saved in the repo, and are as follows:

1. QCA.p (pickle): Contains the QCArchive data downloaded from the the server, as well as the index system built to navigate it
2. oMM.oFF-Parsley-uncons.p: Contains MM energy of the QCArchive molecules as determined by the applied forcefield. Here, Parsley 1.0 was applied. It holds a similar index to QCA.p, and can be mapped to the QCArchive data easily as shown in the examples. This pickle was produced by applying offsb/op/openmm.OpenMMEnergy on QCA.p.
3. \.dat files: A list of the the QCA energies paired alongside the Parsley energy calculated above.
4. \.png: Images representing the data files, produced by plot_torsiondrive.py

# Gridopt 

This is an older example, and has a few more bits, such as saving xyz trajectories of all scans. Most of it has been updated in the example laid out above.
