openff-spellbook
================
   <!-- image:: https://img.shields.io/travis/trevorgokey/openff-spellbook.svg -->
   <!-- :target: https://travis-ci.org/trevorgokey/openff-spellbook -->
   <!-- image:: https://circleci.com/gh/trevorgokey/openff-spellbook.svg?style=svg -->
   <!-- :target: https://circleci.com/gh/trevorgokey/openff-spellbook -->
   <!-- image:: https://codecov.io/gh/trevorgokey/openff-spellbook/branch/master/graph/badge.svg -->
   <!-- :target: https://codecov.io/gh/trevorgokey/openff-spellbook -->

Handy functionality for working with OpenFF data

# Installation

Available on PyPi, so a pip install should work:

``` bash

  $ pip install openff-spellbook
```

Preferably in a preconfigured virtual environment e.g. conda. Append --user if
such an environment is not being used.

Currently no dependency checking is performed... depending on the functionality,
openforcefield (RDKit), OpenBabel, CMILES, OpenMM, QCElemental, QCPortal, geomeTRIC, and Psi4 are needed.

# Functionality

Bottled functionality resides in the `ui` submodule. So far:

<details>
<summary>The OpenForceField Spellbook TorsionDrive parser and plotter</summary>

This useful utility is an automated pipeline to save and plot torsiondrive data and figures.
```

  $ python3 -m offsb.ui.qca.torsiondrive -h

  usage: torsiondrive.py [-h] [--out_file_name OUT_FILE_NAME]
               [--datasets DATASETS] [--qm-energy]
               [--mm-energy {None,all,vdw,bonds,angles,dihedrals,outofplanes}]
               [--openff-name OPENFF_NAME]
               [--openff-parameter OPENFF_PARAMETER]
               [--openff-previous OPENFF_PREVIOUS]
               {torsiondrive_groupby_openff}

  The OpenForceField Spellbook TorsionDrive parser

  positional arguments:
    {torsiondrive_groupby_openff}

  optional arguments:
    -h, --help      show this help message and exit
    --out_file_name OUT_FILE_NAME
    --datasets DATASETS
    --qm-energy
    --mm-energy {None,all,vdw,bonds,angles,dihedrals,outofplanes}
    --openff-name OPENFF_NAME
    --openff-parameter OPENFF_PARAMETER
    --openff-previous OPENFF_PREVIOUS
```
</details> 

<details>
<summary>The OpenForceField Spellbook error scanner for QCArchive</summary>

```
  $ python3 -m offsb.ui.qca.errors -h
  usage: errors.py [-h] [--save-xyz] [--report-out REPORT_OUT] [--full-report]
  
  The OpenForceField Spellbook error scanner for QCArchive
  
  optional arguments:
    -h, --help      show this help message and exit
    --save-xyz
    --report-out REPORT_OUT
    --full-report
  
  $ python3 -m offsb.ui.qca.run-optimization
  usage: run-optimization.py [-h] [-o OUT_JSON] [-i] [-m MEMORY] [-n NTHREADS]
                 optimization_id molecule_id

  positional arguments:
    optimization_id   QCA ID of the optimization to run
    molecule_id     QCA ID of the molecule to use

  optional arguments:
    -h, --help      show this help message and exit
    -o OUT_JSON, --out_json OUT_JSON
              Output json file name
    -i, --inputs-only   just generate input json; do not run
    -m MEMORY, --memory MEMORY
              amount of memory to give to psi4 eg '10GB'
    -n NTHREADS, --nthreads NTHREADS
              number of processors to give to psi4
```
</details>

<details>
<summary>The OpenForceField Spellbook energy extractor from QCArchive</summary>

```
  $ python3 -m offsb.ui.qca.energy-per-molecule
  usage: energy-per-molecule.py [-h] [--report-out REPORT_OUT]
  
  The OpenForceField Spellbook energy extractor from QCArchive
  
  optional arguments:
    -h, --help      show this help message and exit
    --report-out REPORT_OUT
```

</details>

<details>
<summary>Transform a SMILES string into a QCSchema</summary>

```
  $ python3 -m offsb.ui.smiles.load -h

  usage: load.py [-h] [-c CUTOFF] [-n MAX_CONFORMERS] [-s LINE_START]
         [-e LINE_END] [-H HEADER_LINES] [-u] [-i ISOMERS]
         [-o OUTPUT_FILE] [-f FORMATTED_OUT] [-j] [-m] [--ncpus NCPUS]
         input

  A tool to transform a SMILES string into a QCSchema. Enumerates stereoisomers
  if the SMILES is ambiguous, and generates conformers.

  positional arguments:
    input         Input file containing smiles strings. Assumes that the
              file is CSV-like, splits on spaces, and the SMILES is
              the first column

  optional arguments:
    -h, --help      show this help message and exit
    -c CUTOFF, --cutoff CUTOFF
              Prune conformers less than this cutoff using all
              pairwise RMSD comparisons (in Angstroms)
    -n MAX_CONFORMERS, --max-conformers MAX_CONFORMERS
              The number of conformations to attempt generating
    -s LINE_START, --line-start LINE_START
              The line in the input file to start processing
    -e LINE_END, --line-end LINE_END
              The line in the input file to stop processing (not
              inclusive)
    -H HEADER_LINES, --header-lines HEADER_LINES
              The number of lines at the top of the file to skip
              before data begins
    -u, --unique-smiles If stereoisomers are generated, organize molecules by
              their unambiguous SMILES string
    -i ISOMERS, --isomers ISOMERS
              The number of stereoisomers to keep if multiple are
              found
    -o OUTPUT_FILE, --output-file OUTPUT_FILE
              The file to write the output log to
    -f FORMATTED_OUT, --formatted-out FORMATTED_OUT
              Write all molecules to a formatted output as qc_schema
              molecules. Assumes singlets! Choose either --json or
              --msgpack as the the format
    -j, --json      Write the formatted output to qc_schema (json) format.
    -m, --msgpack     Write the formatted output to qc_schema binary message
              pack (msgpack).
    --ncpus NCPUS     Number of processes to use.
```
An example output if the SMILES input file is just `C` (methane) would be the following:
```
{   
  "C": [
    {
      "schema_name": "qcschema_molecule",
      "schema_version": 2,
      "validated": true,
      "symbols": [
        "C",
        "H",
        "H",
        "H",
        "H"
      ],
      "geometry": [
        0.00967296,
        -0.02006983,
        0.01136534,
        1.0387219,
        1.42757171,
        -1.12813096,
        1.41684881,
        -1.11105294,
        1.10602765,
        -1.10880164,
        -1.23235809,
        -1.277628,
        -1.35644204,
        0.93590916,
        1.28836596
      ],
      "name": "CH4",
      "molecular_charge": 0.0,
      "molecular_multiplicity": 1,
      "connectivity": [
        [
          0,
          1,
          1.0
        ],
        [
          0,
          2,
          1.0
        ],
        [
          0,
          3,
          1.0
        ],
        [
          0,
          4,
          1.0
        ]
      ],
      "fix_com": false,
      "fix_orientation": false,
      "provenance": {
        "creator": "QCElemental",
        "version": "v0.15.1",
        "routine": "qcelemental.molparse.from_schema"
      },
      "extras": null
    }
  ]
}
```

</details>

<details>
<summary>Submit an Optimization Dataset based on SMILES</summary>

First, generate the the JSON for --input-molecules from `python3 -m offsb.ui.smiles.load`. This will
be the direct input for `--input-molecules`. Then call the following:

```
  $ python3 -m offsb.ui.smiles.load -h

  usage: submit-optimizations.py [-h] [--input-molecules INPUT_MOLECULES]
                   [--metadata METADATA]
                   [--compute-spec COMPUTE_SPEC]
                   [--threads THREADS]
                   [--dataset-name DATASET_NAME] [--server SERVER]
                   [--priority {low,normal,high}]
                   [--compute-tag COMPUTE_TAG] [--verbose]

  The OpenFF Spellbook QCArchive Optimization dataset submitter

  optional arguments:
    -h, --help      show this help message and exit
    --input-molecules INPUT_MOLECULES
              A JSON file which contains the QCSchema ready for
              submission. The json should be a list at the top-
              level, containing dictionaries with a name as a key,
              and the value a list of QCMolecules representing the
              different conformations of the same molecule. Note
              that entry data, e.g. the CMILES info, should not be
              specified here as it is generated automatically from
              this input.
    --metadata METADATA The JSON file containing the metadata of the dataset.
    --compute-spec COMPUTE_SPEC
              A JSON file containing the new compute specification
              to add to the dataset
    --threads THREADS   Number of threads to use to communicate with the
              server
    --dataset-name DATASET_NAME
              The name of the dataset. This is needed if the dataset
              already exists and no metadata is supplied. Useful
              when e.g. adding computes or molecules to an existing
              dataset.
    --server SERVER   The server to connect to. The special value
              'from_file' will read from the default server
              connection config file for e.g. authentication
    --priority {low,normal,high}
              The priority of the calculations to submit.
    --compute-tag COMPUTE_TAG
              The compute tag used to match computations with
              compute managers. For OpenFF calculations, this should
              be 'openff'
    --verbose       Show the progress in the output.
```

Here, an example `--metadata metadata.json` could be:
```
{
  "submitter": "trevorgokey",
  "creation_date": "2020-09-18",
  "collection_type": "OptimizationDataset",
  "dataset_name": "OpenFF Sandbox CHO PhAlkEthOH v1.0", 
  "short_description": "A diverse set of CHO molecules",
  "long_description_url": "https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2020-09-18-OpenFF-Sandbox-CHO-PhAlkEthOH",
  "long_description": "This dataset contains an expanded set of the AlkEthOH and PhEthOH datasets, which were used in the original derivation of the frosst specification.",
  "elements": [
    "C",
    "H",
    "O"
  ],
  "change_log": [
    {"author": "trevorgokey",
     "date": "2020-09-18",
     "version": "1.0",
     "description": "A diverse set of CHO molecules. The molecules in this set were generated to include all stereoisomers if chirality was ambiguous from the SMILES input. Conformations were generated which had an RMSD of at least 4 Angstroms from all other conformers"
    }
  ]
}
```

And if we want to perform both optimizations using B3LYP-D3BJ/DZVP and MM OpenFF 1.0.0, then the JSON file to give to `--compute-spec compute.json` could be the following:

```
{"default":
  {"opt_spec":
    {"program": "geometric",
     "keywords":
      {"coordsys": "tric",
       "enforce": 0.1,
       "reset": true,
       "qccnv": true,
       "epsilon": 0.0}
    },
    "qc_spec": {
      "driver": "gradient",
      "method": "b3lyp-d3bj",
      "basis": "dzvp",
      "program": "psi4",
      "keywords": {
        "maxiter": 200,
        "scf_properties": [
          "dipole",
          "quadrupole",
          "wiberg_lowdin_indices",
          "mayer_indices"
        ]
      }
    }
  },
 "openff-1.0.0":
  {"opt_spec":
    {"program": "geometric",
     "keywords":
      {"coordsys": "tric",
       "enforce": 0.1,
       "reset": true,
       "qccnv": true,
       "epsilon": 0.0}
    },
    "qc_spec": {
      "driver": "gradient",
      "method": "openff-1.0.0",
      "basis": "smirnoff",
      "program": "openmm",
      "keywords": { }
    }
  }
}
```
Note that the `default` specification is the standard for fitting new versions of the SMIRNOFF OpenForceField.

Running the command will will produce the following output if `--verbose` is used. First, create the input molecules:
```
$ python3 -m offsb.ui.smiles.load methane.smi -n 10 -c 2 -f methane.json -j
     1 /      1 ENTRY: C
            ISOMER   1/  1 CONFS: 1 SMILES: C
            Inputs:      1 Isomers:      1 Conformations:      1
100%|█████████████████████████████████████████████| 1/1 [00:00<00:00,  2.71it/s]
Totals:
  Inputs:   1
  Isomers:     1
  Conformations: 1
```
Now submit the optimizations (here a private server using `localhost:7777`):

```

$ python3 -m offsb.ui.qca.submit-optimizations --verbose --metadata metadata.json --compute-spec compute.json --server localhost:7777 --priority normal --compute-tag openff --input-molecules methane.json

Arguments given:
{'compute_spec': 'compute.json',
 'compute_tag': 'openff',
 'dataset_name': None,
 'input_molecules': 'methane.json',
 'metadata': 'metadata.json',
 'priority': 'normal',
 'server': 'localhost:7777',
 'threads': None}

Dataset created with the following metadata:
{'change_log': [{'author': 'trevorgokey',
         'date': '2020-09-18',
         'description': 'A diverse set of CHO molecules. The molecules '
                'in this set were generated to include all '
                'stereoisomers if chirality was ambiguous from '
                'the SMILES input. Conformations were '
                'generated which had an RMSD of at least 4 '
                'Angstroms from all other conformers',
         'version': '1.0'}],
 'collection_type': 'OptimizationDataset',
 'creation_date': '2020-09-18',
 'dataset_name': 'OpenFF Sandbox CHO PhAlkEthOH v1.0',
 'elements': ['C', 'H', 'O'],
 'long_description': 'This dataset contains an expanded set of the AlkEthOH '
           'and PhEthOH datasets, which were used in the original '
           'derivation of the frosst specification.',
 'long_description_url': 'https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2020-09-18-OpenFF-Sandbox-CHO-PhAlkEthOH',
 'short_description': 'A diverse set of CHO molecules',
 'submitter': 'trevorgokey'}

Successfully added specification default:
{'opt_spec': {'keywords': {'coordsys': 'tric',
               'enforce': 0.1,
               'epsilon': 0.0,
               'qccnv': True,
               'reset': True},
        'program': 'geometric'},
 'qc_spec': {'basis': 'dzvp',
       'driver': 'gradient',
       'keywords': {'maxiter': 200,
              'scf_properties': ['dipole',
                       'quadrupole',
                       'wiberg_lowdin_indices',
                       'mayer_indices']},
       'method': 'b3lyp-d3bj',
       'program': 'psi4'}}

Successfully added specification openff-1.0.0:
{'opt_spec': {'keywords': {'coordsys': 'tric',
               'enforce': 0.1,
               'epsilon': 0.0,
               'qccnv': True,
               'reset': True},
        'program': 'geometric'},
 'qc_spec': {'basis': 'smirnoff',
       'driver': 'gradient',
       'keywords': {},
       'method': 'openff-1.0.0',
       'program': 'openmm'}}

Loading methane.json into QCArchive...
Number of unique molecules: 1
Entries: 100%|████████████████████████████████████| 1/1 [00:00<00:00, 39.24it/s]

Number of new entries: 1/1

Submitting calculations in batches of 20 for specification default
Tasks: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 16.18it/s]

Submitting calculations in batches of 20 for specification openff-1.0.0
Tasks: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 20.08it/s]

Number of new tasks: 2
```

</details>

The format of the file required for `--datasets` in all commands is the following:
`TYPE NAME WITH SPACES / SPEC1 SPEC2`
Where we could specify, using the above dataset submission example, as:
`OptimizationDataset OpenFF Sandbox CHO PhAlkEthOH v1.0 / default openff-1.0.0`
