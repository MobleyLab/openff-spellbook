#!/usr/bin/env python3
import json

def generate_json(opt, mol):
    js = {}

    js["keywords"] = opt.keywords

    # need to strip non-builtins e.g. numpy arrays
    js["initial_molecule"] = json.loads(mol.json())    

    js["input_specification"] = opt.qc_spec.dict()
    js["input_specification"]['keywords'] = {
        "maxiter": 200, 
        "properties": [
            "dipole", "quadrupole", "wiberg_lowdin_indices", "mayer_indices"
        ]
    }

    js["input_specification"]["model"] = {
        "basis": opt.qc_spec.basis,
        "method": opt.qc_spec.method
    }
    js["input_specification"].pop("basis")
    js["input_specification"].pop("method")

    return js

def qca_query(oid, mid):
    import qcfractal.interface as ptl
    client = ptl.FractalClient()

    opt = client.query_procedures(oid)[0]
    mid = client.query_molecules(mid)[0]

    return opt,mid


def qca_generate_input(oid, mid, memory="2GB", nthreads=1):
    opt, mol = qca_query(oid, mid)
    js = generate_json(opt, mol)
    js['memory'] = memory
    js['nthreads'] = nthreads
    return js

def qca_run_geometric_opt(js):
    import geometric
    out = geometric.run_json.geometric_run_json(js)
    return out 

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('optimization_id', help='QCA ID of the optimization to run')
    parser.add_argument('molecule_id', help='QCA ID of the molecule to use')
    parser.add_argument('-o', '--out_json', default=None, help='Output json file name')
    parser.add_argument('-i', '--inputs-only', action="store_true",
        help='just generate input json; do not run')
    parser.add_argument('-m', '--memory', type=str, default="2GB", help="amount of memory to give to psi4 eg '10GB'")
    parser.add_argument('-n', '--nthreads', type=int, default=1, help="number of processors to give to psi4")

    args = parser.parse_args()

    js = qca_generate_input(args.optimization_id, args.molecule_id, args.memory, args.nthreads)

    if args.inputs_only:
        ret = js
    else:
        ret = qca_run_geometric_opt(js)
    if args.out_json is None:
        if len(ret) > 0:
            print(json.dumps(ret, indent=2))
    else:
        with open(args.out_json,'w') as fid:
            json.dump(ret, fid, indent=2)

if __name__ == '__main__':
    main()
