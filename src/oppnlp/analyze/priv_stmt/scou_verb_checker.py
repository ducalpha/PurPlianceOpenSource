"""Find SCoU (SoC + Use) verbs."""

from typing import Dict, List

from spacy.tokens import Token


_raw_scou_verbs = {
    'USE': 'access, analyze, check, combine, connect, keep, know, process, save, use, utilize',
    'COLLECT': 'collect, gather, obtain, receive, record, store, solicit',
    'SHARE': 'disclose, distribute, exchange, give, lease, provide, rent, release, report, sell, send, share, trade, transfer, transmit'
}


def split_verb_dict(verb_dict) -> Dict[str, List[str]]:
    """Split verb dict from str->str to str->list."""
    for e_type, verbs in verb_dict.items():
        verb_dict[e_type] = [v.strip().lower() for v in verbs.split(',')]
    return verb_dict


def get_combined_dict_values(adict):
    """Return combined lists of dict value list."""
    all_values = []
    for values in adict.values():
        all_values.extend(values)
    return all_values


scou_type_to_verbs: Dict[str, List[str]] = split_verb_dict(_raw_scou_verbs)
scou_verbs: List[str] = get_combined_dict_values(scou_type_to_verbs)

def is_scou_verb(token: Token):
    """Check whether a token is a scou verb."""
    return token.lemma_ in scou_verbs
