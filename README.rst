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


Bottled functionality resides in the `ui` submodule. So far:

* QCArchive error scanning
* Run an optimization locally using inputs pulled from QCArchive using `geometric+psi4`
* Aggregate molecules and print the energy from each optimization (including torsion drives and grid scans) 

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


The default behavior is to scan every dataset that OpenFF has curated (see `QCArchiveSpellbook.openff_qcarchive_datasets_default` in `offsb.ui.qcasb`). This takes about 25 minutes to prepare when run for the first time, but the results will save to disk after completion, and reloading takes a second. Each command will look for the same cache.
