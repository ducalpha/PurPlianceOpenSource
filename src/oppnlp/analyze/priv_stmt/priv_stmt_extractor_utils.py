"""Utilities for extracting CI parameters."""

from typing import List, Union
from colorama import Fore
from spacy.tokens import Doc, Span


DocOrSpan = Union[Doc, Span]

def _get_end_of_tag(tags, tag_name, start_idx):
    """End of tag is either the end of array or the start of another tag."""
    begin_tag = 'B-' + tag_name
    assert tags[start_idx] == begin_tag, f'Should start at {begin_tag}'

    idx = start_idx + 1
    while idx < len(tags):
        cur_tag = tags[idx]
        if cur_tag.startswith('B-') or cur_tag == 'O':
            break

        idx += 1

    return idx


def get_multi_tag_ranges(tags, tag_name):
    """Get all the ranges from B-tag_name to I-tag_name."""
    start_end_pairs = []
    start = -1
    while True:
        try:
            start = tags.index('B-' + tag_name, start + 1)
            end = _get_end_of_tag(tags, tag_name, start)
            start_end_pairs.append((start, end))
        except ValueError:
            break
    return start_end_pairs


def get_single_tag_range(tags, tag_name, doc=None, verbose=0):
    """Get the range from B-tag_name to I-tag_name.
    Check there is at most 1 range in the tags."""
    ranges = get_multi_tag_ranges(tags, tag_name)
    if len(ranges) > 1:
        print(Fore.YELLOW + f'WARNING: Got more than 1 range: {len(ranges)} of {tag_name}: {tags=} {doc=}' + Fore.RESET)
    if len(ranges) == 0:
        if verbose >= 2:
            print(f'No clause of tag {tag_name} found.')
        return None

    start, end = ranges[0]

    if verbose >= 3:
        print(f'get_single_tag_range(): Found range {start=} {end=} for {tag_name=} {tags=}')

    return start, end


def get_entities_of_type(entity_type, span, verbose=0):
    """Get entity of type entity_type in the span."""
    # This does not include entities partially in the span.
    entities: List[Span] = []
    entity_start = -1  # whether we are in an entity or not.
    for i, t in enumerate(span):
        if t.ent_type_ == entity_type:
            if entity_start == -1: # enter an entity, else: continue an entity
                entity_start = i
        else:
            if entity_start != -1: # leave an entity
                # Extend to include 'your' to distinguish "your information" with "such information".
                if entity_type == 'DATA' and entity_start >= 1 and span[entity_start - 1].lower_ == 'your':
                    entity_start -= 1
                #### End extension
                entities.append(span[entity_start:i])
                entity_start = -1

    # Add the last one if still in an entity.
    if entity_start != -1:
        # Extend to include 'your' to distinguish "your information" with "such information".
        if entity_type == 'DATA' and entity_start >= 1 and span[entity_start - 1].lower_ == 'your':
            entity_start -= 1
        #### End extension
        entities.append(span[entity_start:])

    if verbose >= 2:
        print(f'get_entities_of_type(), span {span}')
        print(f'get_entities_of_type(), entity_type {entity_type}')
        print(f'get_entities_of_type(), entities {entities}')
        if verbose >= 3:
            for t in span:
                print(t, t.lemma_, t.ent_type_)
    return entities


def get_entities(ps_param, spans, verbose=0):
    """Extract entities from the ps_param spans."""
    if verbose >= 2:
        print('get_entities(), ps_param', ps_param)
        print('get_entities(), spans', spans)
        if verbose >= 3:
            for span in spans:
                for t in span:
                    print(t, t.tag_, t.ent_type_)

    if ps_param == 'data':
        entity_type = 'DATA'
    elif ps_param in ['sender', 'receiver']:
        entity_type = 'ORG'
    else:
        raise ValueError(f'Unsupported ci param {ps_param}')

    entities = []
    for span in spans:
        # Exception whey 'they' is a receiver, as 'they' is normally not recognized as an entity.
        if ps_param == 'receiver' and str(span).lower() in ['they', 'with whom']:
            entities.append(span)
            continue
        entities.extend(get_entities_of_type(entity_type, span))

    # Replace with noun chunks.
    noun_chunks = entities

    # In case of wrong POS like 'We may use *feedback* (should be noun but recognized as VB) you provide to improve our products and services.'
    # This is a compromise to get the span which contains some DATA entity.
    if len(noun_chunks) == 0:
        # Likely the name of the company in form of domain name or 2-token name.
        if len(spans) == 1 and sum(1 for t in spans[0]) == 1 and ps_param in ['sender', 'receiver']:
            return spans
        return entities

    if verbose >= 2:
        print('get_entities(), noun_chunks', noun_chunks)

    return noun_chunks


def align_ps_param_to_ent_spans(ps_param_to_spans):
    """Align the spans to the entities of the doc for certain types of param
    such as DATA."""
    ps_param_to_entities = {}

    for ps_param, spans in ps_param_to_spans.items():
        # , 'sender', 'receiver']: # sender/recv normally alread aligned and want to know which stmt has both null sender and receiver
        if ps_param in ['data', 'sender', 'receiver']: # without alignment: "to company locate in other country"
            ps_param_to_entities[ps_param] = get_entities(ps_param, spans)
        else:
            ps_param_to_entities[ps_param] = spans

    return ps_param_to_entities


def get_param_clauses(srl_verb, pred_idx, doc, get_param_func, verbose=0):
    """Get spans for text for each param."""
    param_clauses = []

    if verbose >= 3:
        print('verb obj', srl_verb)

    # Find ARGM-PRP in SRL parsing which have the verb as its predicate.
    param_range_or_ranges = get_param_func(srl_verb, pred_idx, doc)

    if param_range_or_ranges is not None:
        if not isinstance(param_range_or_ranges, list):
            param_ranges = [param_range_or_ranges]
        else:
            param_ranges = param_range_or_ranges

        for param_range in param_ranges:
            if verbose >= 3:
                print('param range', param_range)

            prp_start, prp_end = param_range
            param_clauses.append(doc[prp_start:prp_end])

    return param_clauses
