import numpy as np


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
        if not np.allclose(my_dict[key], 0, atol=1e-9):
            pruned_dict[key] = my_dict[key]
    return pruned_dict
