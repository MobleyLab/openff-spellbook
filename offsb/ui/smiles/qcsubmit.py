#!/usr/bin/env python3
import sys
from openforcefield.topology import Molecule
from qcsubmit import \
    workflow_components  # load in a list of workflow_components
from qcsubmit.factories import OptimizationDatasetFactory

if __name__ == "__main__":
    breakpoint()

    qcsds = OptimizationDatasetFactory()

    component = workflow_components.StandardConformerGenerator()
    component.max_conformers = 100
    component.toolkit = "rdkit"
    qcsds.add_workflow_component(component)

    filter = workflow_components.RMSDCutoffConformerFilter()
    filter.rmsd_cutoff = 4.0
    qcsds.add_workflow_component(filter)

    smi_file = sys.argv[1]
    mols = [Molecule.from_smiles(smi.split()[0], allow_undefined_stereochemistry=True) for smi in open(smi_file).readlines()]

    dataset = qcsds.create_dataset(dataset_name='my_dataset', molecules=mols, description="my test dataset.")
    dataset.export_dataset("dataset.json")


    # now lets save the workflow to json
    qcsds.export_settings('output.yaml')
