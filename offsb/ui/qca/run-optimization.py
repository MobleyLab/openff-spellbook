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
        "scf_properties": [
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

def run(js, fnm=None):
    import geometric
    out = geometric.run_json.geometric_run_json(js)
    if fnm is None:
        return json.dumps(out, indent=2)
    else:
        with open(fnm,'w') as fid:
            json.dump(out, fid, indent=2)
        return None

def qca_run_geometric_opt(oid, mid, fnm=None):
    opt, mol = qca_query(oid, mid)
    js = generate_json(opt, mol)
    ret = run(js, fnm)
    return ret

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('optimization_id', help='QCA ID of the optimization to run')
    parser.add_argument('molecule_id', help='QCA ID of the molecule to use')
    parser.add_argument('-o', '--out_json', default=None, help='Output json file name')
    args = parser.parse_args()

    ret = qca_run_geometric_opt(args.optimization_id, args.molecule_id, args.out_json)
    if not ret is None:
        print(ret)

if __name__ == '__main__':
    main()
