"""Classification of purpose clauses by pattern matching."""

import inspect
import re

from oppnlp.analyze.priv_stmt.prp_clause_preprocessor import PreprocPrpClause


def contains_any_word_text(text, word_list):
    """Search words at word boundaries. Note: does not handle plural forms."""
    # Excluding the hyphen prepending like non-marketing, but not excluding the hyphen following.
    # Check https://bit.ly/306iELM
    # word_regex = r'(?<!-)\b({})s?\b(?!-)'.format('|'.join(word_list))
    word_regex = r'(?<!-)\b({})s?\b'.format('|'.join(word_list))
    search = re.search(word_regex, text)
    return search is not None

def contains_any(words, word_list):
    """Check whether words intersect with word_list not empty."""
    return len(set(words).intersection(word_list)) > 0

ad_terms = {"advertisement", "advertising", "ad"}

class PrpClassMatcher:
    high_name = None


class AnyMatcher(PrpClassMatcher):
    high_name = 'Any'

    @classmethod
    def is_purpose(cls, clause, verb, obj):
        if re.search(r'internal( (business|legitimate))? purpose(s)?', clause):
            return True

        return False


class ProductionMatcher(PrpClassMatcher):
    high_name = 'Production'

    @classmethod
    def is_provide_service(cls, clause, verb, obj):
        """Return one of the classes."""
        verb_list = {'provide', 'deliver', 'offer', 'execute'}
        obj_list = {'service', 'app', 'product', 'feature', 'game', 'application'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        verb_list = {'provide', 'deliver', 'offer', 'execute'}
        obj_list = {'service', 'app', 'product', 'feature'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        return False

    @classmethod
    def is_improve_service(cls, clause, verb, obj):
        verb_list = {'improve'}
        obj_list = {'service', 'app', 'product', 'feature', 'application'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        verb_list = {'provide'}
        obj_list = {'better'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        return False

    @classmethod
    def is_develop_service(cls, clause, verb, obj):
        verb_list = {'track', 'detect', 'debug'}
        obj_list = {'issue', 'bug', 'error'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        verb_list = {'troubleshoot'}
        if contains_any(verb, verb_list):
            return True

        obj_list = {'troubleshooting'}
        if contains_any(obj, obj_list):
            return True

        return False

    @classmethod
    def is_personalize_service(cls, clause, verb, obj):
        verb_list = {'personalize', 'target'}
        obj_list = {'service', 'app', 'product', 'feature', 'content'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        verb_list = {'base'}
        nc_list = {'location'}
        if contains_any(obj, nc_list):
            return True

        return False

    @classmethod
    def is_manage_account(cls, clause, verb, obj):
        verb_list = {'create', 'register', 'manage', 'terminate'}
        obj_list = {'account'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        return False

    @classmethod
    def is_manage_service(cls, clause, verb, obj):
        verb_list = {'administer', 'manage'}
        obj_list = {'service', 'app', 'product', 'feature'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        obj_list = {'administrative'}
        if contains_any(obj, obj_list):
            return True

        return False

    @classmethod
    def is_process_payment(cls, clause, verb, obj):
        verb_list = {'process', 'complete', 'verify'}
        obj_list = {'transaction', 'payment', 'order'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        return False

    @classmethod
    def is_security(cls, clause, verb, obj):
        verb_list = {'detect', 'investigate', 'prevent'}
        obj_list = {'breach', 'fraud'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        obj_list = {'security', 'fraud prevention'}
        if contains_any(obj, obj_list) or contains_any_word_text(clause, obj_list):
            return True

        verb_list = {'authenticate', 'verify'}
        obj_list = {'user', 'identity'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        obj_list = {'authentication'}
        if contains_any(obj, obj_list) or contains_any_word_text(clause, obj_list):
            return True

        return False

class MarketingMatcher(PrpClassMatcher):
    high_name = 'Marketing'

    @classmethod
    def is_customer_comm(cls, clause, verb, obj):
        verb_list = {'notify'}
        obj_list = {'user'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        verb_list = {'send'}
        obj_list = {'update', 'message'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        verb_list = {'resolve', 'respond'}
        obj_list = {'inquiry', 'request'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        return False

    @classmethod
    def is_marketing_analytics(cls, clause, verb, obj):
        verb_list = {'analyze', 'measure'}
        obj_list = {'usage', 'trend'}.union(ad_terms)
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        obj_list = {'analytical', 'analytic', 'analytics', 'analysis', 'research', 'metrics'}
        if contains_any(obj, obj_list) or contains_any_word_text(clause, obj_list):
            return True

        return False

    @classmethod
    def is_promotion(cls, clause, verb, obj):
        verb_list = {'send'}
        obj_list = {'offer', 'reward', 'promotion'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        return False

    @classmethod
    def is_advertising(cls, clause, verb, obj):
        verb_list = {'provide', 'deliver', 'show', 'serve'}
        obj_list = ad_terms
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        if 'purpose of advertisement' in ' '.join(obj):
            return True

        if 'advertising' in clause:
            return True

        return False

    @classmethod
    def is_personalize_ad(cls, clause, verb, obj):
        verb_list = {"personalise", "personalize", "customise", "customize", "tailor", "identify"}
        obj_list = ad_terms
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        return False

    @classmethod
    def is_general_marketing(cls, clause, verb, obj):
        nc_list = {'marketing'}
        if contains_any(obj, nc_list) or any(nc in clause for nc in nc_list):
            return True

        return False


class LegalityMatcher(PrpClassMatcher):
    high_name = 'Legality'

    @classmethod
    def is_general_legality(cls, clause, verb, obj):
        verb_list = {'enfoce'}
        obj_list = {'term', 'right'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        verb_list = {'comply'}
        obj_list = {'law'}
        if contains_any(verb, verb_list) and contains_any(obj, obj_list):
            return True

        nc_list = {"legal", 'law'} #, 'regulation'} too few.
        if contains_any(obj, nc_list):
            return True

        return False

class OtherMatcher(PrpClassMatcher):
    high_name = 'Other'

    @classmethod
    def is_general(cls, clause, verb, obj):
        if re.search('(various|other|any)\spurpose(s)?', clause.lower()):
            return True

        obj_list = {'various purpose', 'other purpose', 'any purpose'}
        if contains_any(obj, obj_list) or contains_any_word_text(clause, obj_list):
            return True
        return False

def get_low_funcs(acls):
    """Get all low-level matching functions."""
    def get_low_class_name(func_name):
        prefix = 'is_'
        return func_name[len(prefix):] if func_name.startswith(prefix) else None

    # f is a pair of (func_name, func)
    low_funcs = []
    for f in inspect.getmembers(acls, predicate=inspect.isroutine):
        low_name = get_low_class_name(f[0])
        if low_name is None:
            continue
        low_funcs.append((low_name, f[1]))
    return low_funcs

matcher_to_funcs = {
    matcher: get_low_funcs(matcher)
    for matcher in [AnyMatcher, ProductionMatcher, MarketingMatcher, LegalityMatcher, OtherMatcher]
}

def classify_prp_clause(prp_clause, verbose=0):
    cvo_pairs = PreprocPrpClause(prp_clause).get_cvo_pairs()

    if len(cvo_pairs) == 0:
        cvo_pairs += [('', '')]

    prp_classes = []

    for v, o in cvo_pairs:
        if verbose >= 2: print('v|o:', v, '|', o)
        for matcher, low_funcs in matcher_to_funcs.items():
            for low_name, low_func in low_funcs:
                if low_func(prp_clause, [v], o):
                    class_name = '_'.join([matcher.high_name, low_name])
                    prp_classes.append(class_name)

    return sorted(list(set(prp_classes)))
