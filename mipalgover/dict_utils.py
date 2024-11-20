import numpy as np
import scipy.sparse as spa


def merge_dict(dict1, dict2):
    # the union of the 2 inputs with added values
    merged_dict = dict1.copy()
    for key in dict2.keys():
        if key in dict1.keys():
            merged_dict[key] += dict2[key]
        else:
            merged_dict[key] = dict2[key]
    return merged_dict


def prune_dict(my_dict):
    pruned_dict = dict()
    for key in my_dict.keys():
        val = my_dict[key]
        if isinstance(val, np.ndarray):
            if not np.allclose(val, 0, atol=1e-9):
                pruned_dict[key] = val
        elif spa.issparse(val):
            if val.count_nonzero():  # condition passes as long as at least 1 nonzero element
                pruned_dict[key] = val
    return pruned_dict
