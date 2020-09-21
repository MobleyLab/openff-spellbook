#!/usr/bin/env python3
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import zip_longest
from pprint import pformat

import cmiles
import qcfractal.interface as ptl
import tqdm

logger = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()

        except (KeyboardInterrupt, SystemExit):
            raise

        except:
            self.handleError(record)


logger.addHandler(TqdmLoggingHandler())


def chunk(iterable, chunk_size):
    """
    Generate a chunked version of an iterable for batch processing

    Parameters
    ----------
    iterable : iterable
        The object that has an iterator
    chunk_size : int
        The numble of elements to place into each chunk.

    Returns
    -------
    result : list
        A list of lists where the sublists from the iterable have a length
        specified by chunk_size. The last chunk may be smaller if the chunk_size
        does not evenly divide the length of the iterable.
    """
    chunks = [iter(iterable)] * chunk_size
    result = [
        list(filter(lambda x: x is not None, sublst)) for sublst in zip_longest(*chunks)
    ]
    return result


def submit(ds, entry_name, molecule, index):
    """
    Submit an optimization job to a QCArchive server.

    Parameters
    ----------
    ds : qcportal.collections.OptimizationDataset
        The QCArchive OptimizationDataset object that this calculation
        belongs to
    entry_name : str
        The base entry name that the conformation belongs to. Usually,
        this is a canonical SMILES, but can be anything as it is represents
        a key in a dictionary-like datastructure. This will be used as an
        entry name in the dataset
    molecule : QCMolecule
        The JSON representation of a QCMolecule, which has geometry
        and connectivity present, among others
    index : int
        The conformation identifier of the molecule. This is used to make
        the entry names unique, since each conformation must have its own
        unique entry in the dataset in the dataset

    Returns
    -------
    (unique_id, success): tuple
    unique_id : str
        The unique_id that was submitted to the dataset. This is the name
        of the new entry in the dataset.
    success : bool
        Whether the dataset was able to successfully add the entry. If this
        is False, then the entry with the name corresponding to unique_id
        was already present in the dataset.
    """

    # This workaround prevents cmiles from crashing if OE is installed but has
    # no license. Even though rdkit is specified, protomer enumeration is OE-
    # specific and still attempted.
    oe_flag = cmiles.utils.has_openeye
    cmiles.utils.has_openeye = False

    attrs = cmiles.generator.get_molecule_ids(molecule, toolkit="rdkit")

    cmiles.utils.has_openeye = oe_flag

    CIEHMS = "canonical_isomeric_explicit_hydrogen_mapped_smiles"
    molecule["extras"] = {CIEHMS: attrs[CIEHMS]}

    unique_id = entry_name + f"-{index}"

    success = False
    try:
        ds.add_entry(unique_id, molecule, attributes=attrs, save=False)
        success = True
    except KeyError:
        pass

    return unique_id, success


def add_compute_specs(ds, specifications):
    """
    Add one or more completed compute specifications to a dataset. These
    correspond to QCEntrySpecs

    Parameters
    ----------
    ds : qcportal.collections.OptimizationDataset
        The QCArchive OptimizationDataset object that this calculation
        belongs to
    specifications : dict(name=values)
        name : str
            The name of the specification.
        values : dict
            The compute specification defined by the given name. An example
            specification resembles the following:

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
                }
            }

            The keys "opt_spec" and "qc_spec" are required by this function,
            but are not stored in QCA. All of the other key-value pairs, however,
            are QCArchive specific and must use appropriate values.

            Note that this specification in QCA will replace the keyword dictionary
            in the "qc_spec" and replace it with an identifier from the internal
            table of keyword stores.

    Returns
    -------
    success : dict(name=value)
        name : str
            The specification name from the input.
        value : bool
            Whether the specification was successfully added. If this is False,
            the specification had a problem, or was already present.
    """

    success = {}
    for spec_name, spec in specifications.items():

        opt_spec = spec["opt_spec"]
        qc_spec = spec["qc_spec"]
        kw_vals = qc_spec["keywords"]

        if type(kw_vals) == str:
            kw_id = kw_vals
        else:
            kw = ptl.models.KeywordSet(values=kw_vals)
            kw_id = ds.client.add_keywords([kw])[0]

        qc_spec["keywords"] = kw_id

        try:

            ds.add_specification(spec_name, opt_spec, qc_spec)

            spec["qc_spec"]["keywords"] = kw.dict()["values"]
            success[spec_name] = True

            logger.info("\nSuccessfully added specification {}:".format(spec_name))
            logger.info(pformat(spec))

        except KeyError:

            success[spec_name] = False

    return success


def submit_qca_optimization_dataset(
    dataset_name=None,
    metadata=None,
    compute_spec=None,
    input_molecules=None,
    server="from_file",
    threads=None,
    compute_tag=None,
    priority="normal",
):
    """
    Create or update an optimization dataset
    """

    ds_type = "OptimizationDataset"
    ds_name = dataset_name

    if server == "from_file":
        # Connect to a server that needs authentication
        client = ptl.FractalClient().from_file()

    elif server is not None:
        # Use a custom server, possibly a local, private server
        client = ptl.FractalClient(server, verify=False)

    else:
        # Use the default public MOLSSI server
        client = ptl.FractalClient()

    try:

        ds = client.get_collection(ds_type, ds_name)

    except KeyError:

        assert metadata is not None
        metadata = json.load(open(metadata))
        metadata["collection_type"] = ds_type

        if ds_name is not None:
            metadata["dataset_name"] = ds_name
        else:
            ds_name = metadata["dataset_name"]

        ds = getattr(ptl.collections, ds_type)(
            ds_name,
            client=client,
            metadata=metadata,
            description=metadata["short_description"],
            tags=["openff"],
            tagline=metadata["short_description"],
        )
        logger.info("\nDataset created with the following metadata:")
        logger.info(pformat(metadata))

    if compute_spec is not None:
        specs = json.load(open(compute_spec))

        add_compute_specs(ds, specs)

    if input_molecules is not None:
        pool = ThreadPoolExecutor(max_workers=threads)

        new_mols = 0
        new_calcs = 0
        total_calcs = 0

        logger.info("\nLoading {} into QCArchive...".format(input_molecules))
        input_ds = json.load(open(input_molecules))
        logger.info("Number of unique molecules: {}".format(len(input_ds)))

        work = []
        for j, index in enumerate(input_ds):
            for i, mol in enumerate(input_ds[index], 1):
                work_unit = pool.submit(submit, *(ds, index, mol, i))
                work.append(work_unit)

        ds.save()

        ids = []
        new_entries = 0

        iterable = enumerate(as_completed(work))
        if logger.getEffectiveLevel() >= logging.INFO:
            iterable = tqdm.tqdm(iterable, total=len(work), ncols=80, desc="Entries")

        for j, unit in iterable:
            unique_id, success = unit.result()
            new_entries += int(success)
            ids.append(unique_id)

        new_mols += len(input_ds)
        new_calcs += new_entries
        total_calcs += len(ids)

        logger.info("\nNumber of new entries: {}/{}".format(new_entries, len(ids)))

        stride = 20

        if compute_spec is not None:
            new_tasks = 0
            for qc_spec_name in specs:
                out_str = (
                    "\nSubmitting calculations in batches of {} for specification {}"
                )
                logger.info(out_str.format(stride, qc_spec_name))

                work = []
                args = (qc_spec_name,)
                kwargs = dict(priority=priority, tag=compute_tag)

                for entry_list in chunk(ids, stride):
                    kwargs["subset"] = entry_list
                    work_unit = pool.submit(ds.compute, *args, **kwargs)
                    work.append(work_unit)

                iterable = as_completed(work)
                if logger.getEffectiveLevel() >= logging.INFO:
                    iterable = tqdm.tqdm(
                        iterable, total=len(work), ncols=80, desc="Tasks"
                    )

                for unit in iterable:
                    submitted = unit.result()
                    new_tasks += submitted

            logger.info("\nNumber of new tasks: {}".format(new_tasks))

        pool.shutdown(wait=True)


def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="The OpenFF Spellbook QCArchive Optimization dataset submitter"
    )

    parser.add_argument(
        "--input-molecules",
        type=str,
        help="A JSON file which contains the QCSchema ready for submission. The json should be a list at the top-level, containing dictionaries with a name as a key, and the value a list of QCMolecules representing the different conformations of the same molecule. Note that entry data, e.g. the CMILES info, should not be specified here as it is generated automatically from this input.",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        help="The JSON file containing the metadata of the dataset.",
    )

    parser.add_argument(
        "--compute-spec",
        type=str,
        help="A JSON file containing the new compute specification to add to the dataset",
    )

    parser.add_argument(
        "--threads",
        type=int,
        help="Number of threads to use to communicate with the server",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        help="The name of the dataset. This is needed if the dataset already exists and no metadata is supplied. Useful when e.g. adding computes or molecules to an existing dataset.",
    )
    parser.add_argument(
        "--server",
        type=str,
        help="The server to connect to. The special value 'from_file' will read from the default server connection config file for e.g. authentication",
    )

    parser.add_argument(
        "--priority",
        type=str,
        default="normal",
        choices=["low", "normal", "high"],
        help="The priority of the calculations to submit.",
    )
    parser.add_argument(
        "--compute-tag",
        type=str,
        default=None,
        help="The compute tag used to match computations with compute managers. For OpenFF calculations, this should be 'openff'",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the progress in the output.",
    )

    args = vars(parser.parse_args())

    logger.handlers = []
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.WARNING)

    if args["verbose"]:
        logger.setLevel(logging.INFO)

    args.pop("verbose", None)

    logger.info("\nArguments given:")
    logger.info(pformat(args))

    submit_qca_optimization_dataset(**args)


if __name__ == "__main__":
    main()
