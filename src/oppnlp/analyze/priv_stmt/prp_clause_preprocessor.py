"""Prp clause preprocessor and extractor."""

from spacy.lemmatizer import VERB
from spacy.tokens import Span, Token
from spacy.util import filter_spans
import spacy.symbols
import phrasemachine

from oppnlp.analyze.pded.pipeline import get_unmerging_nlp
from oppnlp.analyze.priv_stmt.doc_maker import DocMaker
from oppnlp.analyze.priv_stmt.priv_stmt_extractor_utils import get_multi_tag_ranges
from oppnlp.analyze.priv_stmt.semantic_role_model import SemanticRoleModel
from oppnlp.common.nlp_utils import contain_noun_or_prp, get_lemma_pron, get_single_sent


JJ_WHITE_LIST = ['personalized'] # typically get JJ pos.


def preprocess_prp_clause(clause: str):
    clause = clause.lower()
    clause = clause.replace('to: ', 'to ')
    clause = clause.replace('for: ', 'for ')
    return clause


def get_sent_for_prp_clause(prp_clause: str):
    prefix = "we use your information "
    sentence = prefix + prp_clause

    # with nlp.disable_pipes(["ner"]):
    sent = get_single_sent(DocMaker.get_doc(sentence))
    prp_start_idx = len(prefix.split())

    return sent, prp_start_idx


def get_stem(token):
    """ Get lemma, default to a verb. """
    nlp = get_unmerging_nlp()
    lemmatizer = nlp.vocab.morphology.lemmatizer
    if isinstance(token, Token) and token.lower_ not in JJ_WHITE_LIST:
        return token.lemma_
    else:
        return lemmatizer(str(token), VERB)[0]


def is_in(idx, range_list):
    """Return true if idx in in any of the range list."""
    return any(arange[0] <= idx < arange[1] for arange in range_list)


def combine_vo_nc(vo_pair, nc_stem):
    """Combine verb-object pairs and noun chunks."""
    assert vo_pair is not None or nc_stem is not None, (
        'The input should be valid-comp: have at least 1 non-None component.')
    if len(vo_pair) == 0:
        return ['', nc_stem]
    return vo_pair


def get_cvo_pairs(vo_pairs, nc_stems):
    if len(vo_pairs) > 0:
        return vo_pairs
    else:
        return [['', nc_stem] for nc_stem in nc_stems]


prp_prefix_advs = ['solely', 'only', 'than']
simple_valid_prp_prefixes = ['in order to', 'to', 'for']
valid_prp_prefixes = [adv + ' ' + prf for adv in prp_prefix_advs for prf in simple_valid_prp_prefixes]

class PreprocPrpClause:
    """Preprocessed prp clause, also performs several extraction."""

    def __init__(self, prp_clause: str, verbose=0):
        assert isinstance(prp_clause, str), (
            f'Should be a string but got {type(prp_clause)}.')

        self._prp_clause = preprocess_prp_clause(prp_clause)
        self._sent, self._prp_clause_start_idx = get_sent_for_prp_clause(self._prp_clause)
        self._prp_span = self._sent[self._prp_clause_start_idx:]
        if verbose >= 2 and str(self._prp_span) != self._prp_clause:
            print (f'Warning: prp span and prp clause are different: {str(self._prp_span)=} {self._prp_clause=}')

    def _get_args(self, srl_verb, pred_idx):
        """Get arguments of interest."""
        arg_list = []
        arg_names = ['ARG1', 'ARG2', 'ARGM-PRP', 'ARGM-PNC', 'ARGM-MNR', 'ARGM-PRD']
        arg_ranges = []
        for arg in arg_names:
            # Sometimes there are more than 1 ranges for arg2.
            tag_arg_ranges = get_multi_tag_ranges(srl_verb.tags, arg)
            for arg_range in tag_arg_ranges:
                ## ignore cases like [understand, ''] and [the service we provide]
                if arg_range is None:
                    continue
                else:
                    arg_list.append(self._sent[arg_range[0]:arg_range[1]])
                    arg_ranges.append(arg_range)

        # Check validity.
        # Arg contains no noun.
        if not any(contain_noun_or_prp(self._sent[arg_range[0]:arg_range[1]])
                   for arg_range in arg_ranges):
            return None, []

        # sort arg list in order of the occurrence in the sentence.
        # v = provide, arg2 = you, arg1 = with products, arg2 should be before arg1.
        arg_list.sort(key=lambda arg: arg[0].i)
        arg_list_lemmas = [get_lemma_pron(t) for arg in arg_list for t in arg]
        return arg_list_lemmas, arg_ranges


    def get_vo_pairs(self, verbose=0):
        """Extract verb-object pairs from a purpose clause string.
        sent is an artificially created sentence from a purpose clause."""
        # Run SRL for each of the verb in the prp clause to collect vo pairs.
        prp_pred_idx_to_srl_verb = {}
        for t in self._prp_span:
            if t.lower_ not in JJ_WHITE_LIST and (t.pos != spacy.symbols.VERB or t.lower_ in {'including', 'following'}):
                continue

            prp_pred_idx_to_srl_verb.update(
                SemanticRoleModel(self._sent, t.i).get_pred_idx_to_srl_verb())

        # Experiment with excluding nested predicates or not.
        exclude_nested_predicates = False
        pred_idxes = list(prp_pred_idx_to_srl_verb.keys())

        vo_pairs = []

        if verbose >= 2:
            print('get_vo_pairs(), prp_pred_idx_to_srl_verb:', prp_pred_idx_to_srl_verb)
            print('get_vo_pairs(), pred indexes:', pred_idxes)

        passed_arg_ranges = []
        for pred_idx in pred_idxes:
            # This will contains many other ones.
            if str(self._sent[pred_idx]) in {'help', 'allow', 'enable', 'believe'}:
                continue

            # Assume the ARG1 is the object.
            srl_verb = prp_pred_idx_to_srl_verb[pred_idx]

            args, arg_ranges = self._get_args(srl_verb, pred_idx)

            if args is None:
                continue

            if exclude_nested_predicates:
                assert not is_in(pred_idx, arg_ranges), f'predicate should not in its args {pred_idx=} {arg_ranges=}'
                if is_in(pred_idx, passed_arg_ranges):
                    continue
                passed_arg_ranges.extend(arg_ranges)

            verb = get_stem(self._sent[pred_idx])
            new_vo_pair = (verb, args)
            if new_vo_pair not in vo_pairs:
                vo_pairs.append(new_vo_pair)

        if len(vo_pairs) == 0 and verbose >= 2:
            print(f'WARNING: Sentence does not have any verb-object pair: {self._sent}')

        if verbose >= 2:
            print('vo pairs', vo_pairs)

        return vo_pairs

    def get_nc_stems(self):
        """Get noun chunk stems."""
        phrases = phrasemachine.get_phrases(tokens=[t.text for t in self._sent], postags=[t.tag_ for t in self._sent], output='token_spans')

        # Only consider noun phrases which are within the purpose clause, excluding the prefix (for/in/in order to)
        token_spans = [token_span for token_span in phrases['token_spans'] if token_span[0] > 4]
        spans = [self._sent[token_span[0]:token_span[1]] for token_span in token_spans]
        phrase_spans = filter_spans(spans)

        return list({tuple(get_lemma_pron(t) for t in phrase_span) for phrase_span in phrase_spans})

    def get_cvo_pairs(self):
        return get_cvo_pairs(self.get_vo_pairs(), self.get_nc_stems())

    @classmethod
    def check_prp_prefix_valid(cls, prp_clause: str, prp_span: Span, verbose=0):
        assert isinstance(prp_clause, str) and isinstance(prp_span, Span)
        # Fast check by string matching. Need a space after the prefix.
        if not any(prp_clause.startswith(prefix + ' ')
                for prefix in simple_valid_prp_prefixes + valid_prp_prefixes):
            return False

        if verbose >= 2:
            print('is_prp_prefix_valid(): Pass prefix test.')

        # Assert at this point because the prp clause may be already filtered by string matching.
        if str(prp_span) != prp_clause:
            # This can be caused by parsing error such as when prp_clause has quotes.
            print(f'WARNING: prp_span should be of the prp_clause {str(prp_span)=} {prp_clause=}')

        # Slow check using Spacy.
        if prp_span[0].lower_ in prp_prefix_advs:
            start_idx = 1
        else:
            start_idx = 0

        if prp_span[start_idx].lower_ == 'for':
            return True

        assert prp_span[start_idx].lower_ in ['to', 'in'], f'Unexpected {prp_span=}'

        tok_after_to = None
        if prp_span[start_idx].lower_ == 'to':
            if len(prp_span) < start_idx + 2:
                return False
            tok_after_to = prp_span[1]
        else:
            assert prp_span[start_idx].lower_ == 'in', f'Should start with in {prp_span=}'
            if len(prp_span) < start_idx + 4 or prp_span[start_idx + 1].lower_ != 'order' or prp_span[start_idx + 2].lower_ != 'to':
                return False
            tok_after_to = prp_span[start_idx + 3]

        assert tok_after_to is not None, 'tok_after_to is None'

        # Verb base form, ignore cases like 'to authorized users'.
        return tok_after_to.tag_ == 'VB'

    def is_prp_prefix_valid(self):
        return self.check_prp_prefix_valid(self._prp_clause, self._prp_span)
