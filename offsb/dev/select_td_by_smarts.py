import json
import os
import pickle

import offsb.qcarchive
import offsb.qcarchive.qcatree as qca
import offsb.rdutil.mol
import offsb.tools.const
import qcfractal.interface as ptl
import simtk.openmm.openmm
import simtk.unit
from offsb.op import geometry, openforcefield, openmm
from offsb.search import smiles

# The openff forcefield version to use for calculations below
version = "1.2.0"

# The unique identifier for this job.

# We are going to do an OpenMM minimization on the TD, so lets call this opt
name = "opt"


def match_td_smiles(QCA, smitree, indices):
    """
    From a SMARTS search, pull out the torsiondrives that match

    Parameters
    ----------
    QCA : offsb.qcarchive.qcatree.QCATree
        The object holding the index and data of QCArchive objects

    smitree : offsb.op.smiles.SmilesSearchTree
        The object that performed the SMARTS searching on the QCA object

    indices : List[int]
        A 1-index list of indices used to determine the match. If a mapped
        SMARTS pattern was used and the exact pattern is desired, set this
        to [1,2,3,4]. If any torsiondrive matching the inner rotatable bond
        is desired, set this to [2,3], indicating that only the inner indices
        must match.

    Returns
    -------
    match_entries : List[treedi.node.Node]
        A list of nodes corresponding to QCArchive TDEntry objects
    """

    match_entries = list()
    for node in QCA.node_iter_depth_first(QCA.root(), select="Entry"):

        if node.payload not in smitree.db:
            continue

        # these are the indices of everything that matched in the smiles operation
        group_smi_list = smitree.db[node.payload]["data"]

        CIEHMS = "canonical_isomeric_explicit_hydrogen_mapped_smiles"

        smi = QCA.db[node.payload]["data"].attributes[CIEHMS]

        mol = offsb.rdutil.mol.build_from_smiles(smi)

        map_idx = offsb.rdutil.mol.atom_map(mol)
        map_inv = {v - 1: k for k, v in map_idx.items()}

        dihedral = QCA.db[node.payload]["data"].td_keywords.dihedrals[0]

        # Try to detect the special case where we want the rotatable bond
        if len(indices) == 2:
            dihedral = tuple(dihedral[1:3])
        elif len(indices) != 4:
            raise Exception(
                "Only able to search a torsiondrive with 4 indices. If two indices are given, we assume it is the inner rotatable bond"
            )

        dihedral = tuple([map_inv[i] for i in dihedral])

        if dihedral[-1] < dihedral[0]:
            dihedral = dihedral[::-1]

        found = False
        for pattern, matches in group_smi_list.items():
            if len(matches) == 0:
                continue

            for match in matches:

                # The supplied indices are 1-indexed, since they are coupled
                # to the CMILES map e.g. [*:2][*:1][*:3]
                smi = tuple([match[i - 1] for i in indices])
                if smi[-1] < smi[0]:
                    smi = smi[::-1]
                smi = tuple(smi)

                # This is a match
                if smi == dihedral:
                    match_entries.append(node)
                    found = True
                    break

            if found:
                break

    return match_entries


def save(tree):
    name = os.path.join(".", tree.name + ".p")
    print("Saving: ", tree.ID, "as", name, end=" ... ")
    tree.to_pickle(db=True, name=name)
    print("{:12.1f} MB".format(os.path.getsize(name) / 1024 ** 2))


if False:

    # TODO: needs work still as ui can't be imported 
    # Load a list of datasets and build the index
    # "datasets" is a file of datasets to load
    datasets = offsb.ui.qcasb.load_dataset_input("datasets")

    QCA = offsb.ui.qcasb.QCArchiveSpellBook(datasets=datasets).QCA

else:
    # Just a quick shortcut to get something going
    client = ptl.FractalClient()
    ds = client.get_collection(
        "TorsionDriveDataset", "OpenFF Gen 2 Torsion Set 1 Roche"
    )
    QCA = QCA = qca.QCATree("QCA", root_payload=client, node_index=dict(), db=dict())
    drop = ["Intermediates", "Hessian"]
    
    QCA.build_index(ds, drop=drop, keep_specs=["default"])
    save(QCA)


entries = list(QCA.node_iter_depth_first(QCA.root(), select="Entry"))
print("There are {:d} entries total".format(len(entries)))

# The general query on the dataset.
# The results respect the key, and will only return those indices if specified
# If no key is specified, then the indices are in order as they appear in the string
query = smiles.SmilesSearchTree(
    "[#7X3:2](~[#1:1])(~[#6])~[#6:3]=[O:4]", QCA, name="smiles"
)
query.apply(targets=entries)


if True:
    # This is the torsion drive we are looking for
    # For whatever reason, the mapped indices are not respected, so need the map
    #
    # It is possible to select just a subset of the above search.
    # For example matching [2,3] will match any torsiondrive with the same
    # rotatable bond
    #
    # This is 1-indexing
    #
    # This will search for the exact torsion specified above
    # tmap = [1, 2, 3, 4]

    # This will search for *any* torsion that was driven which matches the
    # rotatable bound find in the above search. Useful when we found the pattern,
    # but a slightly different torsion was driven (but on the same bond)
    tmap = [2, 3]

    # Use this to only focus on the TDs that match the query
    entries = match_td_smiles(QCA, query, tmap)
else:

    # This will retreive all entries that matched the query

    # First, pull the entries that the SMARTS matcher recorded
    keys = [n for n in query.db if len(query.db[n]["data"]) > 0]

    # Then pull those entries from the dataset
    entries = [e for e in entries if e.payload in keys]

    print("There are {:d} entries that match".format(len(entries)))

# Download the optimized geometries of the entries of interest

# If torsiondrives, this will download all optimization results
# QCA.cache_optimization_minimum_molecules(nodes=entries)

# This downloads just the "best" optimization per angle for TDs
QCA.cache_torsiondriverecord_minimum_molecules(nodes=entries)

# Save for use later with the newly cached molecules
save(QCA)

###############################################################################
# Perform an OpenMM Energy evaluation

oMM_name = "oFF-" + name + "-" + version
if os.path.exists(oMM_name + ".p"):
    with open(oMM_name + ".p", "rb") as fid:
        oMM = pickle.load(fid)

else:
    # Calculate MM energies of the TD snapshots
    oMM = openmm.OpenMMEnergy(
        "openff_unconstrained-" + version + ".offxml", QCA, oMM_name
    )

    # This will (not) try to do a minimization starting from the TD optimized geometries
    oMM.minimize = True
    oMM.processes = 8

    # Whether to use geometric for the minimization (True), or the native OpenMM minimizer (False)
    oMM.use_geometric = False

    # This will also add the same contraints used in the TD optimizations
    # Turn this minimization and this on if you want the MM-minimized TD
    oMM.constrain = True

    # Perform the calculation; this could take some time depending on the number
    # of entries
    oMM.apply(targets=entries)
    save(oMM)

###############################################################################
# Apply the force field labeler

oFF_name = "labels-" + name + "-" + version
if os.path.exists(oFF_name + ".p"):
    with open(oFF_name + ".p", "rb") as fid:
        oFF = pickle.load(fid)

else:
    oFF = openforcefield.OpenForceFieldTree(
        QCA,
        oFF_name,
        "openff_unconstrained-" + version + ".offxml",
    )

    oFF.apply(targets=entries)
    save(oFF)

# This is the regular output
fid = open("test" + name + version + ".dat", "w")

# This is raw data with just numbers
fid2 = open("test" + name + version, "w")

smidb = query

# Now go through the QM, MM, and SMARTS data and aggregate

for entry_nr, entry in enumerate(entries):

    if entry.payload not in smidb.db:
        # Shouldn't happen, but check just in case
        print("This entry not in the SMARTS match list")
        continue
    idx = smidb.db[entry.payload]["data"]
    if idx == []:
        # Probably also shouldn't happen..
        print("Not a match")
        continue

    ###########################################################################
    # Need to recompute the QCA dihedral map -> CMILES for labeler mapping
    CIEHMS = "canonical_isomeric_explicit_hydrogen_mapped_smiles"

    smi = QCA.db[entry.payload]["data"].attributes[CIEHMS]

    mmol = offsb.rdutil.mol.build_from_smiles(smi)

    map_idx = offsb.rdutil.mol.atom_map(mmol)

    # This inverse map takes QCA indices and sends to CMILES indices
    map_inv = {v - 1: k for k, v in map_idx.items()}

    ###########################################################################

    molecules = list(QCA.node_iter_depth_first(entry, select="Molecule"))

    qmenes = {}
    mmenes = {}

    min_id = None
    qmmin = None
    mmmin = None

    molecules_with_angle = []

    # First pass through the molecules to get the minimimum energy for
    # calculating a reference, and getting the constraint angles for sorting
    for molecule_node in molecules:
        if molecule_node.payload not in oMM.db:
            print("case 1")
            continue
        molecule = oMM.db[molecule_node.payload]["data"]
        if molecule == []:
            print("case 2")
            continue
        optimization = QCA.db[
            next(QCA.node_iter_to_root(molecule_node, select="Optimization")).payload
        ]
        ene = optimization["data"]["energies"][-1]
        if (
            oMM.db[molecule_node.payload]["data"]["energy"] is not None
            and len(optimization["data"]["energies"]) > 0
        ):
            if qmmin is None or ene < qmmin:
                qmmin = ene
                min_id = molecule_node.payload

            qmenes[molecule_node.payload] = ene
            mmenes[molecule_node.payload] = oMM.db[molecule_node.payload]["data"][
                "energy"
            ]

        constr = []
        constraints = list(QCA.node_iter_to_root(molecule_node, select="Constraint"))
        if len(constraints) > 0:
            constr = list(constraints[0].payload[1])
            val = constraints[0].payload[2]
            molecules_with_angle.append([val, molecule_node])

    mmmin = oMM.db[min_id]["data"]["energy"]

    molecules = sorted(molecules_with_angle, key=lambda x: x[0])

    # for molecule_node in QCA.node_iter_depth_first(entry, select="Molecule"):
    for mol_nr, molecule_node in enumerate(molecules):
        constr_val, molecule_node = molecule_node
        if molecule_node.payload not in oMM.db:
            continue

        # This can be just an energy, or a whole new molecule if the OpenMM
        # energy calculation used minimize=True
        omm_result = oMM.db[molecule_node.payload]["data"]
        if omm_result == []:
            continue

        if oMM.db[molecule_node.payload]["data"]["energy"] is None:
            continue

        optimization = QCA.db[
            next(QCA.node_iter_to_root(molecule_node, select="Optimization")).payload
        ]
        qene = optimization["data"]["energies"][-1] - qmmin

        constr = []
        constraints = list(QCA.node_iter_to_root(molecule_node, select="Constraint"))
        if len(constraints) > 0:
            constr = list(constraints[0].payload[1])

        # Depending on what the reference spec is, this could be a plain float
        # (from e.g. QM calculations) or a fancy object with units (OpenMM)
        if issubclass(type(qene), simtk.unit.quantity.Quantity):
            qene /= simtk.unit.kilocalories_per_mole
        else:
            # Assumes unitless from QCA are in au
            qene *= offsb.tools.const.hartree2kcalmol
        mene = oMM.db[molecule_node.payload]["data"]["energy"] - mmmin

        if issubclass(type(mene), simtk.unit.quantity.Quantity):
            mene /= simtk.unit.kilocalories_per_mole

        qcmol = QCA.db[molecule_node.payload]["data"]
        print(entry, molecule_node, end="\n")
        fid.write("{:s} {:s}\n".format(entry.__repr__(), molecule_node.__repr__()))

        # Only caring about 1D torsions for now
        for i in [constr]:

            # Get the indices of the driven torsion that will correspond to
            # what CMILES-ordered molecules use (e.g. the FF labeler)
            mapped_dihedral = tuple([map_inv[j] for j in i])
            if mapped_dihedral[0] > mapped_dihedral[-1]:
                mapped_dihedral = tuple(mapped_dihedral[::-1])

            print("Complete key is", i)
            if len(i) != 4:
                continue

            if True:
                # If we want to save the molecule to
                tdr = next(QCA.node_iter_to_root(molecule_node, select="TorsionDrive"))
                mode = "w" if mol_nr == 0 else "a"
                with open(
                    "TD-{:s}.QMMin.xyz".format(tdr.payload),
                    mode,
                ) as fd:
                    offsb.qcarchive.qcmol_to_xyz(
                        qcmol,
                        fd=fd,
                        comment=json.dumps(i)
                        + str(constr_val)
                        + " {:s} {:s}".format(
                            molecule_node.payload, molecule_node.index
                        ),
                    )
                if (
                    oMM.db[molecule_node.payload]["data"].get("schema_name", "")
                    == "qcschema_molecule"
                ):
                    qcmmmol = oMM.db[molecule_node.payload]["data"]
                with open(
                    "TD-{:s}.MMMin.xyz".format(tdr.payload),
                    mode,
                ) as fd:
                    offsb.qcarchive.qcmol_to_xyz(
                        qcmmmol,
                        fd=fd,
                        comment=json.dumps(i)
                        + str(constr_val)
                        + "MM energy: "
                        + str(oMM.db[molecule_node.payload]["data"].get("energy", None))
                        + " {:s} {:s}".format(
                            molecule_node.payload, molecule_node.index
                        ),
                    )

            angle = geometry.TorsionOperation.measure_praxeolitic_single(
                qcmol["geometry"], i
            )

            if "geometry" in omm_result:
                # Measure the angle directly from the MM minimization (mostly to ensure everything went OK)
                anglemin = geometry.TorsionOperation.measure_praxeolitic_single(
                    omm_result["geometry"], i
                )
            else:
                # No minimization, so the QM and MM angles are the same
                anglemin = angle

            # Labels are stored at the entry level since all entry molecules have the same labels
            label = oFF.db[entry.payload]["data"]["ProperTorsions"][mapped_dihedral]

            out_str = "{:3d}-{:3d}-{:3d}-{:3d} TD_Angle= {:10.2f} AngleMin= {:10.2f} QM_Energy= {:10.2f} MM_Energy= {:10.2f} Label= {}\n".format(
                *i, angle, anglemin, qene, mene, label
            )
            print(out_str, end="")
            fid.write(out_str)

            # This is just the data, no strings attached
            fid2.write(
                "{:4d} {:f} {:f} {:f} {:f}\n".format(
                    entry_nr, angle, anglemin, qene, mene
                )
            )
        print()
        fid.write("\n")

fid.close()
fid2.close()
