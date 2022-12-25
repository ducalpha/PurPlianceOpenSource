"""Extract privacy statement params from sentences using SRL."""

from typing import Dict, List, Union
import re

from colorama import Fore
from loguru import logger
from overrides import overrides
from spacy.tokens import Doc, Span, Token
import spacy

from oppnlp.analyze.priv_stmt.priv_stmt_extractor_utils import (
    DocOrSpan, align_ps_param_to_ent_spans, get_param_clauses, get_multi_tag_ranges)
from oppnlp.analyze.priv_stmt.prp_clause_preprocessor import PreprocPrpClause
from oppnlp.analyze.policheck.PolicyExtractor import DependencyGraphConstructor
from oppnlp.analyze.priv_stmt.priv_stmt_extractor import PrivStmt, PrivStmtExtractor
from oppnlp.analyze.priv_stmt.priv_stmt_extractor_utils import get_single_tag_range
from oppnlp.analyze.priv_stmt.semantic_role_model import SemanticRoleModel, SrlVerb
from oppnlp.analyze.priv_stmt.priv_stmt_post_processor import post_process_priv_stmts
from oppnlp.analyze.priv_stmt.scou_verb_checker import is_scou_verb, scou_type_to_verbs
from oppnlp.common.list_utils import get_dict_permutations
from oppnlp.common.nlp_utils import get_single_sent, get_ent_ranges, parse_doc


logger.add("skipped_priv_stmt.log")

prp_start_words = {'to', 'for', 'as'}


def has_data_entity_in_obj(srl_verb: SrlVerb, doc: Doc, verbose=0):
    """Check whether the verb contain any data entity in its object."""
    if verbose >= 2:
        print(f'has_data_entity_in_obj {srl_verb=}')

    if verbose >= 3:
        print('has_data_entity_in_obj(): entities in sentence:')
        print(f'{doc=}')
        for ent in doc.ents:
            print(ent, ent.label_)

    arg1_ranges = get_multi_tag_ranges(srl_verb.tags, 'ARG1')
    for arg1_range in arg1_ranges:
        if verbose >= 2:
            print('has_data_entity_in_obj(): predicate arg1_range', arg1_range)

        for ent_range in get_ent_ranges(doc, 'DATA'):
            # Check whether the arg1_range contain any ent_range.
            if (arg1_range[0] <= ent_range[0]
                    and ent_range[1] <= arg1_range[1]):
                return True

            if verbose >= 2:
                print('DATA ent_range:', doc[ent_range[0]:ent_range[1] + 1])

    return False


def arg2_is_prp(tokens: List[Token], srl_verb: SrlVerb, verbose=0):
    arange = get_single_tag_range(srl_verb.tags, 'ARG2')
    if arange is None:
        return False

    arg2_span = tokens[arange[0]:arange[1]]
    if verbose >= 2:
        print(f'arg2_is_prp() {arg2_span=}')

    return PreprocPrpClause.check_prp_prefix_valid(str(arg2_span), arg2_span)


def get_data_clause_ranges(srl_verb, pred_idx, doc):
    return get_multi_tag_ranges(srl_verb.tags, 'ARG1')


def get_adv_clause_range(srl_verb, pred_idx, doc):
    return get_multi_tag_ranges(srl_verb.tags, 'ARGM-ADV')


def has_purpose(srl_verb, pred_idx, doc):
    """Check whether the doc has purpose or not."""
    valid_prp_ranges = get_purpose_clause_ranges(srl_verb, pred_idx, doc)

    return len(valid_prp_ranges) > 0


def get_purpose_clause_ranges(srl_verb, pred_idx, doc):
    """Return the range of ARGM-PRP from B- to end of I-"""
    verb = doc[pred_idx].lemma_
    purpose_roles = ['ARGM-PRP', 'ARGM-PNC']

    # Some verbs name the argument with a special argument.
    if verb in ['use', 'save', 'check', 'utilize']:
        purpose_roles.extend(['ARG2'])
    if verb in ['analyze']:
        purpose_roles.extend(['ARGM-ADV'])
    if verb in ['save', 'receive', 'solicit', 'record']:
        purpose_roles.extend(['ARG3'])
    if verb in ['receive']:
        purpose_roles.extend(['ARG4'])
    if verb in ['disclose', 'give', 'sell', 'send', 'transmit', 'provide']:
        purpose_roles.extend(['C-ARG1'])

    purpose_ranges = []
    for purpose_role in purpose_roles:
        purpose_ranges.extend(
            get_multi_tag_ranges(srl_verb.tags, purpose_role))

    cau_purpose_ranges = get_multi_tag_ranges(srl_verb.tags, 'ARGM-CAU')
    if len(cau_purpose_ranges) > 0:
        print(f'****************** Try ARGM-CAU, found: {cau_purpose_ranges}')
        purpose_ranges.extend(cau_purpose_ranges)

    valid_prp_ranges = []
    for prp_range in purpose_ranges:
        prp_span = doc[prp_range[0]:prp_range[1]]
        if PreprocPrpClause.check_prp_prefix_valid(str(prp_span), prp_span):
            valid_prp_ranges.append(prp_range)

    return valid_prp_ranges


def get_sender_receiver_clause_ranges(srl_verb, pred_idx, doc):
    """There can be multiple senders/receivers."""
    verb = doc[pred_idx].lemma_

    if verb in scou_type_to_verbs['USE']:
        # No sender for use.
        sender_arg, receiver_arg = 'NONE', 'ARG0'
    elif verb in scou_type_to_verbs['COLLECT']:
        sender_arg, receiver_arg = 'ARG2', 'ARG0'
    elif verb in scou_type_to_verbs['SHARE']:
        sender_arg, receiver_arg = 'ARG0', 'ARG2'
    else:
        raise ValueError(f'Unsupported verb {verb}.')

    # Assume there is only 1 sender/receiver, so use get single range.
    tags = srl_verb.tags
    sender_ranges = get_multi_tag_ranges(tags, sender_arg)
    receiver_ranges = get_multi_tag_ranges(tags, receiver_arg)

    return sender_ranges, receiver_ranges


def get_sender_clause_ranges(srl_verb, pred_idx, doc):
    return get_sender_receiver_clause_ranges(srl_verb, pred_idx, doc)[0]


def get_receiver_clause_ranges(srl_verb, pred_idx, doc):
    return get_sender_receiver_clause_ranges(srl_verb, pred_idx, doc)[1]


def get_ps_param_to_spans_internal(doc, srl_verb, pred_idx, verbose=0):
    ps_param_to_get_clause_func = {
        'purpose': get_purpose_clause_ranges,
        'sender': get_sender_clause_ranges,
        'receiver': get_receiver_clause_ranges,
        'data': get_data_clause_ranges,
        'adv': get_adv_clause_range
    }

    ps_param_to_spans = {}
    for param, get_clause_func in ps_param_to_get_clause_func.items():
        ps_param_to_spans[param] = get_param_clauses(srl_verb, pred_idx, doc, get_clause_func)

    if verbose >= 3:
        print('Add action')

    add_action(ps_param_to_spans, doc, pred_idx)

    add_neg(ps_param_to_spans, doc, pred_idx, srl_verb)

    if verbose >= 3:
        print(f'Predicate: {srl_verb.verb}, '
              f'purpose: {ps_param_to_spans["purpose"]}, '
              f'data: {ps_param_to_spans["data"]}, ')

    if ps_param_to_spans['neg'] == [True] and is_xcomp_allow(ps_param_to_spans['action'][0][0]):
        ps_param_to_spans  = reduce_general_info(ps_param_to_spans)
        if verbose >= 3:
            print(f'After reducing general info:\nPredicate: {srl_verb.verb}, '
                f'purpose: {ps_param_to_spans["purpose"]}, '
                f'data: {ps_param_to_spans["data"]}, ')

    # Align the spans to the entities of the doc.
    ps_param_to_spans = align_ps_param_to_ent_spans(ps_param_to_spans)

    ps_param_to_spans['pred_idx'] = [pred_idx]
    return ps_param_to_spans

def is_xcomp_allow(token: Token):
    return token.dep == spacy.symbols.xcomp and token.head.lower_ in ['allow', 'permit']

def create_priv_stmt(ps_param_to_span, verbose=0):
    """Create a PrivStmt from a flattened ps_param_to_span dict."""
    priv_stmt = PrivStmt(
        ps_param_to_span.get('sent', 'NULL'),
        ps_param_to_span.get('receiver', 'NULL'),
        ps_param_to_span.get('neg', 'NULL'),
        ps_param_to_span.get('action', 'NULL'),
        ps_param_to_span.get('action_lemma', 'NULL'),
        ps_param_to_span.get('sender', 'NULL'),
        ps_param_to_span.get('data', 'NULL'),
        ps_param_to_span.get('purpose', 'NULL'),
        ps_param_to_span.get('pred_idx', 'NULL'),
        ps_param_to_span.get('adv', 'NULL')
    )

    if verbose >= 2:
        msg = ' | '.join(str(getattr(priv_stmt, key)) for key in ['neg', 'sender', 'receiver', 'action', 'data', 'purpose', 'adv'])
        print(msg)

    return priv_stmt


def flatten_ps_param_to_spans(ps_param_to_spans):
    """Return only 1-to-1 privacy-statement param to 1 span of text."""
    return get_dict_permutations(ps_param_to_spans)


def reduce_general_info(ps_param_to_spans):
    if ps_param_to_spans['data'] == 'NULL':
        return ps_param_to_spans
    new_spans = []
    for span in ps_param_to_spans['data']:
        # keyword = 'information about'
        new_span = None
        for i, word in enumerate(span):
            if word.lower_ == 'information' and i < len(span) - 1 and span[i + 1].lower_ == 'about':
                new_span = span[i + 2:]
        if new_span is not None:
            new_spans.append(new_span)
        else:
            new_spans.append(span)
    ps_param_to_spans['data'] = new_spans
    return ps_param_to_spans


def add_null(ps_param_to_spans):
    """Add NULL for empty span lists."""
    for spans in ps_param_to_spans.values():
        assert isinstance(spans, list), (
            f'spans should be a list, but is {type(spans)}')
        if len(spans) == 0:
            spans.append('NULL')


def add_neg(ps_param_to_spans, doc: DocOrSpan, pred_idx: int, srl_verb: SrlVerb):
    verb = doc[pred_idx]
    if isinstance(doc, Doc):
        sent = get_single_sent(doc)
    else:
        assert isinstance(doc, Span)
        sent = doc

    # Try to use SRL to determine negation.
    if get_single_tag_range(srl_verb.tags, 'ARGM-NEG', doc) is not None:
        ps_param_to_spans['neg'] = [True]
    else: # fall back to dependency tree.
        ps_param_to_spans['neg'] = [is_verb_negated(verb, sent)]


def add_action(ps_param_to_spans, doc, pred_idx):
    """Add action param."""
    ps_param_to_spans['action'] = [doc[pred_idx:pred_idx + 1]]
    ps_param_to_spans['action_lemma'] = [doc[pred_idx].lemma_]


def add_prp_class(ps_param_to_span):
    prp_class_key = 'prp_classes'
    # This step now is done in post-processing, after prp clauses are extracted.
    # prp_clause = ps_param_to_span['purpose']
    # ps_param_to_span[prp_class_key] = PrpClassifier.predict(prp_clause)
    ps_param_to_span[prp_class_key] = ''


def is_verb_negated(verb, sent, verbose=0):
    """Check whether the verb is negated or not."""
    negated = DependencyGraphConstructor.isVerbNegated(verb, sent)
    if verbose >= 2:
        print(f'{verb=} {negated=}')
        for t in verb.children:
            print(t, t.dep_)

    # Ignore: "we do not use ... to collect ..." does not mean not collect.
    if negated:
        if verb.dep == spacy.symbols.xcomp and verb.head.lower_ in ['use'] and is_verb_negated(verb.head, sent):
            return False

    return negated


def get_ps_param_to_spans(sent, srl_verb, pred_idx, verbose=0):
    """Return privacy statements for a given predicate at pred_idx."""
    if verbose >= 3:
        suffix = sent[pred_idx + 1:] if pred_idx < len(sent) - 1 else ''
        print(Fore.RED + 'get_ps_param_to_spans()' + Fore.MAGENTA, 'sent:', sent[:pred_idx], Fore.BLUE, sent[pred_idx], Fore.MAGENTA, suffix, Fore.RESET)

    ps_param_to_spans = get_ps_param_to_spans_internal(sent, srl_verb, pred_idx)

    if verbose >= 2:
        print(Fore.YELLOW + '--> Got spans:' + Fore.RESET, f'{ps_param_to_spans=}')

    ps_param_to_spans['sent'] = [sent]

    if verbose >= 3:
        print(Fore.YELLOW + 'get_ps_param_to_spans()' + Fore.RESET, 'Add NULL')

    add_null(ps_param_to_spans)

    if verbose >= 3:
        print(Fore.YELLOW + 'get_ps_param_to_spans()' + Fore.RESET, 'Flatten list')

    return ps_param_to_spans


def flatten_to_priv_stmts(ps_param_to_spans, verbose=0):
    assert isinstance(ps_param_to_spans, dict), 'ps_param_to_spans should be a dict.'

    ps_param_to_span_list = flatten_ps_param_to_spans(ps_param_to_spans)
    assert len(ps_param_to_span_list) >= 1, (
        f'Flatten list should have more elements than the unflattened one'
        f'{len(ps_param_to_span_list)} < 1')

    if verbose >= 2:
        print(Fore.GREEN + 'flatten_to_priv_stmts()' + Fore.RESET, f'create priv_stmts:')

    priv_stmts = [create_priv_stmt(ps_param_to_span) for ps_param_to_span in ps_param_to_span_list]

    for priv_stmt in priv_stmts:
        if str(priv_stmt.data) == 'NULL':
            logger.debug(f'Data is null: {priv_stmt=}')

    return priv_stmts


def get_srl_verb(doc: Union[Doc, Span], pred_idx: int, verbose=0):
    """Get SrlVerb for the predicate in the tokens.
    doc can be Doc or Span (sent) as long as pred_idx is valid for that."""
    if verbose >= 2:
        print(f'get_srl_verb() {pred_idx=} {doc[pred_idx]=}')

    srl_verb = SemanticRoleModel([t for t in doc], pred_idx=pred_idx).get_pred_idx_to_srl_verb()[pred_idx]

    if verbose >= 2:
        print('get_srl_verb(): original srl_verb:', srl_verb)

    return srl_verb


class PrivStmtExtractorImpl(PrivStmtExtractor):
    """Extract privacy statement parameters using SRL."""

    @classmethod
    def get_ps_param_to_spans(cls, sent: Span, verbose=0):
        doc_ps_param_to_spans: List[Dict] = []

        if len(sent) > 150 or len(str(sent)) > 1200:
            print('WARNING: Skip too long sentence:', sent)
            return doc_ps_param_to_spans

        tokens = [t for t in sent]

        if verbose >= 2:
            print(Fore.GREEN + '*** Start get_ps_param_to_spans() for doc:', Fore.CYAN, str(sent), Fore.RESET)
            if verbose >= 3:
                for t in tokens:
                    print(t, t.dep_, t.ent_type_)

        for i, t in enumerate(tokens):
            # Consider only arguments of SCoU verbs.
            if not t.tag_.startswith('VB') or not is_scou_verb(t):
                continue

            if verbose >= 2:
                prefix = sent[:i] if i > 0 else ''
                suffix = sent[i + 1:] if i < len(sent) - 1 else ''
                print(Fore.GREEN + 'get_ps_param_to_spans() for verb:' + Fore.MAGENTA, prefix, Fore.BLUE, sent[i], Fore.MAGENTA, suffix, Fore.RESET)

            srl_verb = get_srl_verb(sent, i)

            if not has_data_entity_in_obj(srl_verb, sent):
                if verbose >= 2:
                    print(Fore.RED + f'Skip this verb, not have entity in obj:' + Fore.RESET, f'{srl_verb=}')
                continue

            if verbose >= 3:
                print(f'{sent[i]} is a valid verb, sent root: {sent.root}')

            # Predicate's privacy statements.
            ps_param_to_spans = get_ps_param_to_spans(sent, srl_verb, i)
            doc_ps_param_to_spans.append(ps_param_to_spans)

        return doc_ps_param_to_spans

    @classmethod
    def should_ignore(cls, sent: Span):
        def doesSentenceStartWithInterrogitive(sentence): # From PolicyLint
            return any(child.lemma_ in [u'who', u'what', u'when', u'where', u'why', u'how', u'do'] and child.dep == spacy.symbols.advmod for child in sent.root.children)

        sentence = str(sent)
        if "You have the right" in sentence:
            return True

        if re.search("(this|our)?\s(policy|document|privacy\s(policy|notice)?)\s.*(address|describe|explain|outline|cover|set|applies)", sentence, re.IGNORECASE):
            return True

        if re.search("(described|set out)\sin\s((the|this)\s)?(section|(privacy\s)policy)", sentence, re.IGNORECASE):
            return True

        # if FirstThirdPartyPredicate.is_title_case(sentence): return True  # WE DO NOT SHARE ...

        if doesSentenceStartWithInterrogitive(sentence):
            return True

        return False

    @classmethod
    def get_priv_stmts_from_sent(cls, sent: Span, verbose=0):
        if cls.should_ignore(sent):
            if verbose >= 2:
                print(Fore.CYAN + f'get_priv_stmts_from_sent() Ignore: {sent=}' + Fore.RESET)
            return []

        ps_param_to_spans = cls.get_ps_param_to_spans(sent)

        return [priv_stmt for amap in ps_param_to_spans for priv_stmt in flatten_to_priv_stmts(amap)]

    @classmethod
    @overrides
    def get_priv_stmts_from_sent_doc(cls, doc: Doc, post_process=True):
        """Extract and return a list of priv_stmts. test_no_prp: turn on/off for testing effect of purposes."""
        # Assume the doc has only 1 sentence.
        sent = get_single_sent(doc)
        priv_stmts = cls.get_priv_stmts_from_sent(sent)
        priv_stmts = [s._asdict() for s in priv_stmts]

        if post_process: 
            priv_stmts = post_process_priv_stmts(priv_stmts)

        return priv_stmts

    @classmethod
    def get_priv_stmts_from_sent_str(cls, doc_str: str, post_process=True):
        return cls.get_priv_stmts_from_sent_doc(cls._nlp(doc_str), post_process)

    @classmethod
    def get_ps_param_to_spans_from_sent_doc(cls, doc: Doc):
        if doc[0].tag_ == '':  # not parsed
            doc = parse_doc(cls._nlp._get_nlp(), doc)

        sent = get_single_sent(doc)

        return cls.get_ps_param_to_spans(sent)
