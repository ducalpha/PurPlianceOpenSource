"""Post-process extracted privacy statements: extract prp classes from purpose
    clauses and handle restricted prp clauses."""

from copy import copy
from typing import Dict, List
import re

from colorama import Fore
from spacy.tokens import Span, Token
import pandas as pd
import spacy

from oppnlp.analyze.priv_stmt.prp_claz.policy_prp_featurizer import classify_prp_clause
from oppnlp.common.nlp_utils import get_single_sent, lemmatize
from oppnlp.analyze.priv_stmt.scou_verb_checker import scou_type_to_verbs
from oppnlp.analyze.policheck.NlpUtils.ExclusionPhraseMerger import mergeExcludePhrases
from oppnlp.analyze.policheck.ExclusionDetector import checkException
from oppnlp.common.nlp_utils import lemmatize_pron


def shouldIgnoreSentence(s):
    mentionsChildRegex = re.compile(r'\b(children|kids|from\sminor(s)?|under\s1[0-9]+|under\s(thirteen|fourteen|fifteen|sixteen|seventeen|eighteen)|age(s)?(\sof)?\s1[0-9]+|age(s)?(\sof)?\s(thirteen|fourteen|fifteen|sixteen|seventeen|eighteen))\b', flags=re.IGNORECASE)
    mentionsUserChoiceRegex = re.compile(r'\b(you|user)\s(.*\s)?(choose|do|decide|prefer)\s.*\s(provide|send|share|disclose)\b', flags=re.IGNORECASE)
    mentionsUserChoiceRegex2 = re.compile(r'\b((your\schoice)|((withhold|withdraw)\s(your\s)?consent)|(you\s.*decline)|(you\sdo\snot\shave\sto\sgive)|((you|we)\s.*ask)|please|opt-(out|in))\b', flags=re.IGNORECASE)
    mentionsUserChoiceRegex3 = re.compile(r'\byou\s(may)?\shave\sthe\srights?\b', flags=re.IGNORECASE)
    mentionsExceptInPrivacyPol1 = re.compile(r'\b(except\sas(\sotherwise)?\s(stated|described|noted))\b', flags=re.IGNORECASE)
    mentionsExceptInPrivacyPol2 = re.compile(r'\b(except\sin(\sthose\slimited)?\s(cases))\b', flags=re.IGNORECASE)
    mentionsLaw = re.compile(r'(california|coppa)', flags=re.IGNORECASE)

    regexes = [mentionsChildRegex,
               mentionsUserChoiceRegex,
               mentionsUserChoiceRegex2,
               mentionsUserChoiceRegex3,
               mentionsExceptInPrivacyPol1,
               mentionsExceptInPrivacyPol2,
               mentionsLaw]
    if any(regex.search(s) is not None for regex in regexes):
        return True

    return False


def create_new_stmt(stmt, new_dict):
    new_stmt = stmt.copy()
    new_stmt.update(new_dict)
    return new_stmt


def handle_other_than_prp_clause(stmt, verbose=0):
    if verbose >= 2:
        print(f'Handle other than prp clause: {stmt["prp_class"]=}')

    if not stmt['neg']:
        print(f'WARNING: Expect negated clause {stmt=}')
        return [stmt]

    if stmt['prp_class'] != 'NULL':
        stmt['for'] = 'for'
    stmt['neg'] = False
    results = [stmt]

    if 'Production_provide_service' == stmt['prp_class']:  # not for other than providing services.
        # not for + other high prps, same party.
        new_stmt = create_new_stmt(stmt, {'for': 'not_for', 'prp_class': 'Marketing_advertising', 'party': 'third party'})
        results.append(new_stmt)

    return results

def handle_use_for_money(stmt):
    if stmt['action_lemma'] in ['share', 'use'] and stmt['prp_class'] == 'NULL':
        if re.search('(profit|money|monetary)', str(stmt['purpose'])):
            stmt['for'] = 'not_for' if stmt['neg'] else 'for'
            stmt['action_lemma'] = 'share'
            stmt['party'] = 'third party'
            stmt['prp_class'] = 'Marketing_advertising'
            return [stmt]

    return None

def handle_exceptions(stmt, verbose=0):
    if not stmt['neg']: # do not reverse: "we will share without your consent"
        # Handle special case for positive statements.
        prp_clause_str = str(stmt['purpose'])
        if (result := handle_only_prp_clause(stmt)) is not None:
            return result

        return [stmt]
    else:
        if re.search(r'except to', str(stmt['sent'])):
            if (result := handle_except_to_prp(stmt)) is not None:
                return result

    if (result := handle_use_for_money(stmt)) is not None:
        return result

    sent = stmt['sent']
    doc = copy(sent.doc)
    mergeExcludePhrases(doc, doc.vocab)
    msent = get_single_sent(doc)
    exceptions = checkException(msent)

    if verbose >= 0:
        print('** Exceptions :', exceptions)

    # TODO: handle other cases which include entities and data objects.
    prp_clause_str = str(stmt['purpose'])
    if prp_clause_str != 'NULL' and re.search(r'^.*\b((other\s)?than)\b.*$', prp_clause_str):
        return handle_other_than_prp_clause(stmt)

    for v, e in exceptions:
        elemma = lemmatize(e)
        if verbose >= 2:
            print('## Exceptions elemma:', elemma)

        if re.search(r'^.*\b(consent|you\sagree|your\s(express\s)?permission|you\sprovide|opt([\s\-](out|in))?|respond\sto\syou(r)?|disclose\sin\sprivacy\spolicy|follow(ing)?\scircumstance|permit\sby\schildren\'s\sonline\sprivacy\sprotection\sact)\b.*$', elemma):
            stmt['neg'] = False
            if stmt['prp_class'] != 'NULL':
                stmt['for'] = 'for'
        elif elemma in ['require by law', 'we receive subpoena', 'law']:
            stmt['neg'] = False
            if stmt['prp_class'] != 'NULL':
                stmt['for'] = 'for'
            stmt['receiver'] = 'government agency'
        else: # do not what it is, still flip anyway.
            stmt['neg'] = False
            if stmt['prp_class'] != 'NULL':
                stmt['for'] = 'for'

    if verbose >= 3:
        print('After handle exceptions', stmt)

    return [stmt]


def handle_exceptions_stmts(post_stmts):
    """Mimics PolicyLint's handleException(), but we just flip the negation, and only create when necessary."""
    return [s for stmt in post_stmts for s in handle_exceptions(stmt)]


def handle_only_prp_clause(priv_stmt):
    prp_clause_str = str(priv_stmt['purpose'])
    if not priv_stmt['neg'] and priv_stmt['action_lemma'] in ['use']:
       if (re.search(r'for internal( (business|legitimate))? purpose(s)? only', prp_clause_str)
           or (priv_stmt['prp_class'] == 'Production_provide_service' and re.search(r'only for', prp_clause_str))
       ):
        return [ priv_stmt, create_new_stmt(priv_stmt, {'receiver': 'third party', 'prp_class': 'Marketing_advertising', 'for': 'not_for', 'party': 'third party'}) ]

    return None


def handle_except_to_prp(priv_stmt, verbose=0):
    if verbose >= 2:
        print("handle_except_to_prp(): To handle 'except to'.")

    assert priv_stmt['neg']
    adv = str(priv_stmt.get('adv'))
    if adv is None or adv == 'NULL' or not re.search(r'except to', adv):
        return None

    except_prps = classify_prp_clause(adv)
    print(f'{except_prps=}')
    # If the prp class is to provide service, add a statement about not for third party's marketing purposes.
    if ['Production_provide_service'] == except_prps:
        return [ priv_stmt, create_new_stmt(priv_stmt, {'receiver': 'third party', 'prp_class': 'Marketing_advertising', 'for': 'not_for', 'party': 'third party', 'neg': False, 'action': 'collect', 'action_lemma': 'collect'}) ]

    return None


def remove_subsumed_prps(post_stmts):
    """Remove subsumed priv stmts."""
    def is_similar(stmt1, stmt2):
        """Almost-the-same stmts except on some keys."""
        assert set(stmt1.keys()) == set(stmt2.keys()), f'Expect same key: {stmt1=} {stmt2=}'
        keys = filter(lambda key: key != 'prp_class', stmt1.keys())
        return all(stmt1[key] == stmt2[key] for key in keys)

    # Remove anything subsumed by Any_purpose-purpose statements.
    any_prp_stmts = filter(lambda stmt: stmt['prp_class'] == 'Any_purpose', post_stmts)
    i = 0
    for any_prp_stmt in any_prp_stmts:
        while i < len(post_stmts):
            if post_stmts[i] != any_prp_stmt and is_similar(post_stmts[i], any_prp_stmt):
                del post_stmts[i]
            i += 1


""" implicit sender/receiver
# share and receiver is None:
#    sender: you -> receiver: we_implicit  (you share)
#    else        -> receiver: third_party_implicit (we share, is shared, they share)
# share and sender is None -> sender: we_implicit (is shared)
# collect and receiver is None -> receiver: we_implicit (is collected)
"""
def transform_share(stmt, verbose=0):
    """Perform the 4 simplification rules for sharing."""
    action_lemma = stmt['action_lemma']
    assert action_lemma in scou_type_to_verbs['SHARE']

    sender = stmt.get('sender')
    receiver = stmt.get('receiver')
    sentence = str(stmt['sent'])
    prp = stmt.get('prp_class')
    result = []

    if str(sender) == 'NULL':
        sender = 'we_implicit'

    stmt_neg = stmt['neg']

    # Part 1: Add collect statements, purpose: NULL
    # Except when sentence says: we do not collect or share data.
    if stmt_neg and re.search(r'not collect (and|or) share', sentence):
        # Do not return here because we will add a sharing stmt in Part 2.
        if verbose >= 2:
            print(f'Ignore: {sentence}')
    else:
        result.append(create_new_stmt(stmt, {'receiver': sender, 'action': 'collect', 'prp_class': 'NULL', 'for': 'NULL'}))

    if str(receiver) == 'NULL' and str(sender).lower() in ['we'] and stmt['action_lemma'] in ['share', 'disclose', 'sell', 'rent', 'trade', 'transfer', 'release']:
        receiver = 'third_party_implicit'

    # Part 2: Add share statements
    if str(prp) == 'NULL': # Rules T1, T2: receiver collects/collect data, purpose: NULL
        action = 'not_collect' if stmt_neg else 'collect'

        result.append(create_new_stmt(stmt, {'receiver': receiver, 'action': action, 'for': 'NULL'}))
    else: # Rules T3, T4: receiver collects data, for/not_for purpose
        action = 'collect'
        if not stmt_neg:
            result.append(create_new_stmt(stmt, {'receiver': receiver, 'action': action, 'for': 'for'}))
        else:
            assert stmt_neg
            result.append(create_new_stmt(stmt, {'receiver': receiver, 'action': action, 'for': 'not_for'}))

    return result

def transform_collect(stmt):
    action_lemma = stmt['action_lemma']
    assert action_lemma in scou_type_to_verbs['COLLECT'] or action_lemma in scou_type_to_verbs['USE'], f'Unrecognized action {action_lemma=}'

    # We do not use/combine/analyze does not mean they do not collect the info. TODO: change to 'USE' verbs.
    if action_lemma in ['analyze', 'check', 'combine', 'connect', 'process', 'use'] and stmt['neg'] and str(stmt['receiver']).lower() in ['we']:
        return []
    if stmt['neg'] and str(stmt['receiver']).lower() in ['we'] and str(stmt['sender']) not in ['you', 'NULL']:  # we do not <collect> A from B does not mean "we do not collect A"
        return []

    receiver = stmt.get('receiver')
    if str(receiver) == 'NULL':
        stmt['receiver'] = 'we_implicit'
    elif stmt['action_lemma'] in ['use']:
        if str(receiver).lower() in ['they'] and stmt['party'] == 'third party':
            stmt['receiver'] = 'third_party_implicit'

    if stmt['prp_class'] == 'NULL': # Instead of str(stmt['purpose']) == 'NULL', prp-class may still be set.
        assert stmt.get('for') is None, f'for Should not be set if there is no purpose {stmt=}'
        if stmt['neg']:
            stmt['action'] = 'not_collect'
            stmt['for'] = 'NULL'
        else:
            stmt['action'] = 'collect'
            stmt['for'] = 'NULL'
    else:
        stmt['action'] = 'collect'
        if stmt.get('for') is None: # not set yet. by handle exceptions.
            if stmt['neg']:
                stmt['for'] = 'not_for'
            else:
                stmt['for'] = 'for'

    return [stmt]


def transform_stmt(stmt):
    """Simplify full priv_stmt to 4-tuple stmt:
    1. Convert action to collect or not_collect.
    2. Remove sender.
    For sharing actions: Perform 4 simplification rules.
    """
    action_lemma = stmt['action_lemma']

    if action_lemma not in scou_type_to_verbs['SHARE']:
        return transform_collect(stmt)

    return transform_share(stmt)


def transform_stmts(post_stmts):
    return [stmt for post_stmt in post_stmts for stmt in transform_stmt(post_stmt)]


def assert_simplified_stmt(priv_stmt):
    assert 'neg' not in priv_stmt and priv_stmt['action'] in {'collect', 'not_collect'}, (
        f'{priv_stmt} should be already simplified.')


def set_prp_class(priv_stmt):
    """Predict purpose classes, handle special cases first.
    Generate negated priv_stmts with other prps for only-prp clause."""
    if str(priv_stmt['purpose']) == 'NULL':
        prp_class = priv_stmt.get('prp_class')
        assert prp_class is None or prp_class == 'NULL', f'{priv_stmt=} should not have prp class set to not null'
        priv_stmt['prp_class'] = 'NULL'
        return [priv_stmt]

    if priv_stmt.get('prp_class') is not None: # set by handle exceptions or elsewhere
        return [priv_stmt]

    # General case.
    return handle_prp_simple_clause(priv_stmt)


def handle_prp_simple_clause(stmt):
    """Return one or more stmts, each with a single prp class."""
    prp_clause = stmt.get('purpose')
    assert prp_clause is not None and str(prp_clause) != 'NULL', 'Should be checked in the outer scope'

    prp_classes = classify_prp_clause(str(prp_clause))

    if len(prp_classes) > 0:
        new_stmts = []
        for prp_class in prp_classes:
            if prp_class is not None:
                stmt_with_class = stmt.copy() # all fields are strings/bools.
                stmt_with_class['prp_class'] = prp_class
                stmt_with_class['for'] = 'not_for' if stmt['neg'] else 'for'
                new_stmts.append(stmt_with_class)
        return new_stmts

    # Cannot recognize.
    stmt['prp_class'] = 'NULL'
    return [stmt]

def set_prp_class_stmts(stmts):
    result = []
    for stmt in stmts:
        result.extend(set_prp_class(stmt))
    return result

def dict_equal(adict, bdict):
    for key, adict_val in adict.items():
        if key not in bdict:
            return False
        bdict_val = bdict[key]
        if type(adict_val) != type(bdict_val) or adict_val != bdict_val:
            return False
    return True

def priv_stmt_in(stmt, others):
    for other in others:
        try:
            if dict_equal(stmt, other):
                return True
        except TypeError as e:
            print(e)
            print(f'Error {stmt=} {other=}')
            raise e
    return False


def dedup(stmts):
    results = []
    for stmt in stmts:
        if not priv_stmt_in(stmt, results):
            results.append(stmt)
    return results


def post_process_priv_stmts(priv_stmts: List):
    """Postprocess privacy statements as list of dicts."""
    posts = [post_stmt for priv_stmt in priv_stmts for post_stmt in post_process_priv_stmt(priv_stmt)]
    # Need to deduplicate because of the addition at the transformation step,
    # e.g., we will disclose your personal info for A, B -> will generate multiple (collect, A) for different (disclose, A), (disclose, B)
    return dedup(posts)


def should_ignore(priv_stmt):
    """Filter out false positive. SRL is flexible so good filtering is needed."""
    action = priv_stmt['action']
    receiver = priv_stmt['receiver']
    data = priv_stmt['data']
    sent = priv_stmt['sent']

    if shouldIgnoreSentence(str(sent)):
        return True

    # Case: the data we collect.
    assert isinstance(action, Span), f'Action should exist, {action=} {priv_stmt=}'
    if not isinstance(data, Span):
        print(f'Warning: data is NULL {data=} {priv_stmt=}')
        return True

    # Data is used for ...
    if priv_stmt['action'].lower_ in ['used'] and str(priv_stmt['purpose']) != 'NULL' and str(receiver) == 'NULL' and str(priv_stmt['sender']) == 'NULL':
        return False

    if action[0].i > data[0].i and action[0].tag_ in ['VB', 'VBP']:
        return True

    # Case: we use encryption [Data] technology.
    if data[-1].i < len(sent) - 1 and sent[data[-1].i + 1].lower_ == 'technology':
        return True

    if str(receiver) == 'NULL' and str(priv_stmt['sender']) == 'NULL':
        if is_verb_be_passive(priv_stmt['action'][0]):
            if priv_stmt['action_lemma'] in scou_type_to_verbs['SHARE']:
                priv_stmt['receiver'] = 'third_party_implicit'
            elif priv_stmt['action_lemma'] in ['collect']:
                priv_stmt['receiver'] = 'we'
            elif (priv_stmt['action_lemma'] in scou_type_to_verbs['USE']) and any('advertising' in prp_class for prp_class in classify_prp_clause(str(priv_stmt['purpose']))):
                priv_stmt['receiver'] = 'advertising network'
                priv_stmt['party'] = 'third party'
                priv_stmt['prp_class'] = 'Marketing_advertising'
            else:
                return True
            return False
        return True
    if str(receiver).lower() == str(priv_stmt['sender']).lower():
        return True
    if str(receiver).lower() == 'you' or (str(priv_stmt['sender']).lower() == 'you' and str(receiver).lower() not in ['us']):
        return True

    # More filter for negated verbs to reduce false positives.
    if priv_stmt['neg']:
        # We cannot disclose ...
        if is_verb_cannot(priv_stmt):
            return True

        if is_receiver_coref(priv_stmt):
            return True

    # If we have not shared ...
    if is_action_in_condition_clause(priv_stmt):
        return True

    return False


def is_verb_be_passive(t):
    return any(child.dep == spacy.symbols.auxpass for child in t.children)


def is_verb_cannot(priv_stmt):
    def is_aux_can(token: Token):
            return any(t.dep == spacy.symbols.aux and t.lower_ in ['ca', 'can'] for t in token.children)
    if priv_stmt['neg'] and is_aux_can(priv_stmt['action'][0]):
        return True
    return False


def is_action_in_condition_clause(priv_stmt):
    return is_verb_in_condition_clause(priv_stmt['action'][0])


def is_verb_in_condition_clause(verb: Token):
    """Search up to find any word with conditional token."""
    def is_mark_condition(token: Token):
        if verb.head == verb: # or t.dep == spacy.symbols.advcl
            return False
        return any(t.dep == spacy.symbols.mark and t.lower_ in ['if', 'when'] for t in token.children) or is_verb_in_condition_clause(verb.head)
    return is_mark_condition(verb)


def is_receiver_coref(priv_stmt):
    receiver = priv_stmt['receiver']
    if not isinstance(receiver, Span) or receiver[0].i == 0:
        return False
    return priv_stmt['sent'][receiver[0].i - 1].lower_ in ['this', 'that', 'these', 'those', 'such']


def set_party_single(stmt, verbose=0):
    # Party already set.
    if stmt.get('party') is not None:
        return

    stmt['party'] = 'anyone'

    prp_clause = stmt.get('purpose')
    if prp_clause is not None and str(prp_clause) != 'NULL':
        their_list = ['their', 'their own', 'third party \'s', 'third party \'']
        prp_lemmas = lemmatize_pron(prp_clause)
        if verbose >= 2:
            print(f'{prp_lemmas=}')

        for gram in their_list:
            if gram in prp_lemmas:
                stmt['party'] = 'third party'
                break

        for gram in ['our', 'us']:
            if gram in str(prp_clause):
                stmt['party'] = 'we'
                break

    receiver_str = str(stmt['receiver'])
    if receiver_str != 'NULL' and isinstance(stmt['receiver'], Span):
        receiver_lemmas = lemmatize_pron(stmt['receiver']).lower()
        if 'Marketing' in stmt['prp_class'] and 'third party' in receiver_lemmas:
            stmt['party'] = 'third party'
        elif any(e in receiver_lemmas for e in ['advertiser', 'advertising']):  # third party receive for marketing purposes...
            stmt['receiver'] = 'advertising network'
            stmt['party'] = 'third party'
            stmt['prp_class'] = 'Marketing_advertising'

    if str(stmt['sender']).lower() in ['we'] and stmt['action_lemma'] in ['lease', 'sell', 'rent', 'trade']:
        stmt['party'] = str(stmt['receiver']) if str(stmt['receiver']) != 'NULL' else 'third party'
        # Conservatively set one of the purposes
        if stmt['prp_class'] == 'NULL':
            stmt['for'] = 'not_for' if stmt['neg'] else 'for'
            stmt['prp_class'] = 'Marketing_advertising'


def set_party(stmts):
    for stmt in stmts:
        set_party_single(stmt)
    return stmts


def post_process_priv_stmt(priv_stmt: Dict, verbose=0):
    """Return a list of priv_stmt dicts, containing only [sender, receiver, neg, action, prp_class]."""
    print('post_process_priv_stmt:', priv_stmt)

    if should_ignore(priv_stmt):
        if verbose >= 2:
            print(Fore.CYAN + f'post_process_priv_stmt(): Ignore: {priv_stmt=}' + Fore.RESET)
        return []

    post_stmts = [priv_stmt]

    if str(priv_stmt.get('data')) == 'NULL':
        print(f'WARNING: No data: {priv_stmt=}')

    post_stmts = set_prp_class_stmts(post_stmts)

    if verbose >= 3: print(f'post_process_priv_stmt(): After set prp class: {post_stmts=}')

    post_stmts = set_party(post_stmts)

    if verbose >= 3: print(f'{Fore.YELLOW}post_process_priv_stmt(): After set party: {Fore.RESET}{post_stmts=}')

    post_stmts = handle_exceptions_stmts(post_stmts)

    for stmt in post_stmts: del stmt['adv'] # Not needed after handling exception.

    if verbose >= 3: print(f'post_process_priv_stmt(): After handle exception: {post_stmts=}')

    # Moved to previous step to avoid duplicate stmts when there are many entities.
    post_stmts = transform_stmts(post_stmts)

    if verbose >= 3: print(f'post_process_priv_stmt(): After handle transform: {post_stmts=}')

    # Remove unnecessary action and sender in the final.
    for new_stmt in post_stmts:
        for attr in ['sender', 'neg']:
            if attr in new_stmt:
                del new_stmt[attr]

    # TODO: should be fixed in the priv stmt extractor.
    post_stmts = [post_stmt for post_stmt in post_stmts if post_stmt.get('data') is not None]
    post_stmts = [post_stmt for post_stmt in post_stmts if str(post_stmt['receiver']).lower() not in ['null', 'you', 'user', 'users', 'that', 'this']]

    for post_stmt in post_stmts:
        assert (post_stmt.get('receiver') is not None and str(post_stmt.get('receiver')) != ''
            and post_stmt.get('data') is not None
            and str(post_stmt['action']) in ['collect', 'not_collect']
            and post_stmt.get('for') is not None), (
            f'Invalid post stmt {post_stmt=} {priv_stmt=}')

        assert post_stmt.get('prp_class') is not None
        if post_stmt['prp_class'] != 'NULL':
            assert post_stmt['for'] != 'NULL', f'{post_stmt=}'

    if verbose >= 3:
        print(f'Final stmts')
        df = pd.DataFrame(post_stmts)
        cols = [col for col in df.columns if col != 'sent']
        print(df[cols].to_string(index=False))

    return post_stmts
