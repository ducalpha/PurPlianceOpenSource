"""Utilities for lists."""


def get_dict_permutations_recur(value_dict, value_dict_keys, i, comb, combs):
    """DFS, a combination = a path from root to a leaf.
    Generate all combinations at value_tuple[i], given the combination at
    value_tuple[i-1], add to combinations if at the next of the leaf (i == len(value_tuple))."""
    if i == len(value_dict_keys):
        # Assume combination contains no reference values.
        combs.append(comb.copy())
        return

    dict_key = value_dict_keys[i]
    dict_val = value_dict[dict_key]
    if not isinstance(dict_val, list):
        dict_val = [dict_val]

    for val in dict_val:
        comb[dict_key] = val
        get_dict_permutations_recur(value_dict, value_dict_keys, i + 1, comb, combs)


def get_dict_permutations(value_dict):
    """Return a "flattened" version of a_dict."""
    combs = []
    value_dict_keys = list(value_dict.keys())
    get_dict_permutations_recur(value_dict, value_dict_keys, 0, {}, combs)
    return combs
