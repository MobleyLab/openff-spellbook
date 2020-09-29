#!/usr/bin/env python3
import os
import sys

import qcfractal.interface as ptl
import tqdm


def restart(input_filename, id_type="task"):
    client = ptl.FractalClient().from_file()

    if os.path.exists(input_filename):
        proc_ids = [x.rstrip("\n") for x in open(input_filename, "r").readlines()]
    else:
        # try interpreting as an ID
        proc_ids = [input_filename]
    n_updated = 0
    if id_type == "task":
        ret = client.modify_tasks(
            operation="restart", base_result=proc_ids, full_return=True
        )
    elif id_type == "service":
        for pid in proc_ids:
            ret = client.modify_services(
                operation="restart", procedure_id=pid, full_return=True
            )
            n_updated += ret.data.n_updated
    print(
        "{:20s} Total updated: {:8d}/{:8d}".format(
            input_filename, n_updated, len(proc_ids)
        )
    )


if __name__ == "__main__":
    id_type = "task"
    if len(sys.argv) > 1:
        id_type = sys.argv[2]
    restart(sys.argv[1], id_type)
