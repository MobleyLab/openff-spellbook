===============================
openff-spellbook
===============================
.. 
    image:: https://img.shields.io/travis/trevorgokey/openff-spellbook.svg
    :target: https://travis-ci.org/trevorgokey/openff-spellbook
..
    image:: https://circleci.com/gh/trevorgokey/openff-spellbook.svg?style=svg
    :target: https://circleci.com/gh/trevorgokey/openff-spellbook
..
    image:: https://codecov.io/gh/trevorgokey/openff-spellbook/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/trevorgokey/openff-spellbook

Handy functionality for working with OpenFF data

Installation
############

Available on PyPi, so a pip install should work:

.. code-block::
    pip install openff-spellbook

Preferably in a preconfigured virtual environment e.g. conda. Append --user if
such an environment is not being used.

Currently no dependency checking is performed... depending on the functionality,
openforcefield, OpenMM, QCElemental, QCPortal, geomeTRIC, and Psi4 are needed.

Functionality
#############

Bottled functionality resides in the `ui` submodule. So far:

* QCArchive error scanning
* Run an optimization locally using inputs pulled from QCArchive using
  `geometric+psi4`
* Aggregate molecules and print the energy from each optimization (including
  torsion drives and grid scans) 
* Take a list of SMILES from a file and enumerate isomers and conformers, with
  the result saved as a QCSchema representation. Designed to be cluster
  friendly for large dataset processing (block processing of the input, and
  saving as binary msgpack data)

.. code-block:: bash

    $ python3 -m offsb.ui.qca.errors -h
    usage: errors.py [-h] [--save-xyz] [--report-out REPORT_OUT] [--full-report]
    
    The OpenForceField Spellbook error scanner for QCArchive
    
    optional arguments:
      -h, --help            show this help message and exit
      --save-xyz
      --report-out REPORT_OUT
      --full-report
    
    $ python3 -m offsb.ui.qca.run-optimization
    usage: run-optimization.py [-h] [-o OUT_JSON] [-i] [-m MEMORY] [-n NTHREADS]
                               optimization_id molecule_id

    positional arguments:
      optimization_id       QCA ID of the optimization to run
      molecule_id           QCA ID of the molecule to use

    optional arguments:
      -h, --help            show this help message and exit
      -o OUT_JSON, --out_json OUT_JSON
                            Output json file name
      -i, --inputs-only     just generate input json; do not run
      -m MEMORY, --memory MEMORY
                            amount of memory to give to psi4 eg '10GB'
      -n NTHREADS, --nthreads NTHREADS
                            number of processors to give to psi4
    
    $ python3 -m offsb.ui.qca.energy-per-molecule
    usage: energy-per-molecule.py [-h] [--report-out REPORT_OUT]
    
    The OpenForceField Spellbook energy extractor from QCArchive
    
    optional arguments:
      -h, --help            show this help message and exit
      --report-out REPORT_OUT


    $ python3 -m offsb.ui.smiles.load -h

    usage: load.py [-h] [-c CUTOFF] [-n MAX_CONFORMERS] [-s LINE_START]
                   [-e LINE_END] [-H HEADER_LINES] [-u] [-i ISOMERS]
                   [-o OUTPUT_FILE] [-f FORMATTED_OUT] [-j] [-m]
                   input

    A tool to transform a SMILES string into a QCSchema. Enumerates stereoisomers
    if the SMILES is ambiguous, and generates conformers.

    positional arguments:
      input                 Input file containing smiles strings. Assumes that the
                            file is CSV-like, splits on spaces, and the SMILES is
                            the first column

    optional arguments:
      -h, --help            show this help message and exit
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
      -u, --unique-smiles   If stereoisomers are generated, organize molecules by
                            their unambiguous SMILES string
      -i ISOMERS, --isomers ISOMERS
                            The number of stereoisomers to keep if multiple are
                            found
      -o OUTPUT_FILE, --output-file OUTPUT_FILE
                            The file to write the output log to
      -f FORMATTED_OUT, --formatted-out FORMATTED_OUT
                            Write all molecules to a formatted output as qc_schema
                            molecules. Assumes singlets! Only choose one option:
                            --json or --msgpack
      -j, --json            Write the formatted output to qc_schema (json) format.
      -m, --msgpack         Write the formatted output to qc_schema binary message
                            pack (msgpack)


The default behavior is to scan every dataset that OpenFF has curated (see
`QCArchiveSpellbook.openff_qcarchive_datasets_default` in `offsb.ui.qcasb`).
This takes about 25 minutes to prepare when run for the first time, but the
results will save to disk after completion, and reloading takes a second. Each
command will look for the same cache. The search functionality to combine
similar molecules across all dataset set takes long; near an hour. This is also
cached by default. Both caches are saved as pickle objects for now.
