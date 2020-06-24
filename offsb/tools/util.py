#!/usr/bin/env python3 

import itertools

def flatten_list(l, times=1):
    if times == 0: return l
    if times == -1:
        if isinstance(l[0], list):
            ll = [a for b in l if hasattr(b,"__iter__") for a in b]
            return flatten_list(ll, times)
        else:
            return l
    else:
        return flatten_list(
                 [a for b in l if hasattr(b,"__iter__") for a in b], times-1)

def argsort_labels(l):
    letter = [i[0] for i in l]
    num = [int(i[1:]) for i in l]
    return np.lexsort((num, letter))

def strip_conformation_number(mol):
    ret = mol.split("-")
    if(len(ret) == 1):
        ret = mol
    else:
        try:
            isnumber = int(ret[-1])
            ret = "".join(ret[:-1])
        except ValueError:
            ret = mol
    return ret

def get_conformation_number(mol):
    ret = mol.split("-")
    if(len(ret) == 1):
        ret = "1"
    else:
        try:
            isnumber = int(ret[-1])
            ret = isnumber
        except ValueError:
            ret = "1"
    return ret

def combine(db):
    mols = list(db['mol_data'].keys())
    mols_combined = {}
    logger.debug("Mol list: ") ; [logger.debug(m) for m in mols]
    for i in range(len(mols)):
        query = strip_conformation_number(mols[i])
        logger.debug("Query is: " + query)
        if(query not in mols_combined):
            hits = []
            for j in range(len(mols)):
                mol = strip_conformation_number(mols[j])
                if(mol == query):
                    hits.append(mol)
            logger.debug("Matches are " + str(hits))
            mols_combined[query] = hits
    return mols_combined
