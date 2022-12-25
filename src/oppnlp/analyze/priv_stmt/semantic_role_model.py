"""Extractor for semantic roles of words."""

from collections import Counter
from pprint import pprint
from typing import NamedTuple, List
import json

from allennlp.common.util import sanitize
from allennlp.data import Instance
from spacy.tokens import Token

from oppnlp.analyze.priv_stmt.srl_utils import load_srl_predictor
from oppnlp.common.nlp_utils import contain_empty_token


class SrlVerb(NamedTuple):
    """A semantic-role-labeled verb object produced by the SRL predictor."""
    # Predicate of all arguments.
    verb: str
    # Document with description, like ['We [V: collect] information']
    description: str
    # Token tags in IOB format, like ['O', 'B-V', 'B-ARG2']
    tags: List[str]


class SrlCache:
    """Thin wrapper for caching SRL dict result from AllenNLP."""
    def __init__(self):
        self._cache = {} # Turn on when experiment datasets: dc.Cache('docmaker_cache', size_limit=int(4e9))

    @classmethod
    def _encode(self, val):
        return json.dumps(val)

    @classmethod
    def _decode(self, encoded_val):
        return json.loads(encoded_val)

    def get_srl_dict(self, word_texts: List[str], pred_idx: int):
        """Return srl dict."""
        assert 0 <= pred_idx < len(word_texts), f'pred_idx is out of range {pred_idx}'
        key = self._encode((word_texts, pred_idx))

        encoded_val = self._cache.get(key)
        if encoded_val is not None:
            return self._decode(encoded_val)

        return None

    def set_srl_dict(self, word_texts, pred_idx, srl_dict):
        key = self._encode((word_texts, pred_idx))
        val = self._encode(srl_dict)
        self._cache[key] = val


class SemanticRoleModel:
    """Abstract for a SRL-parsed span."""

    def __init__(self, tokens: List[Token], pred_idx: int):
        """pred_idx: index of the predicate.
        If pred_idx is set, SemanticRoleModel only predicts for the predicate."""
        self._tokens = tokens
        self._pred_idx = pred_idx
        self._srl_dict = None
        self._verbs: List[SrlVerb] = []
        self._srl_cache = SrlCache()

        self._parse_span()

    @property
    def _srl_predictor(self):
        return load_srl_predictor()

    def _srl_verbs(self):
        """Return the list of the predicates as SrlVerb objects."""
        return self._verbs

    def _predict(self, verbose=0):
        """Return a srl_dict. Get it from a cache or predict it."""
        words = [t.text for t in self._tokens]

        srl_dict = self._srl_cache.get_srl_dict(words, self._pred_idx)

        if srl_dict is None:
            srl_dict = self._slow_predict()
            if verbose >= 2:
                print('SRL cache missed.')
            self._srl_cache.set_srl_dict(words, self._pred_idx, srl_dict)
        elif verbose >= 2:
            print('SRL cache hit.')

        return srl_dict

    def _slow_predict(self):
        """Copy from SRL Predictor to skip the tokenization."""
        instances = self._tokens_to_instances(self._tokens)

        if not instances:
            return sanitize({"verbs": [], "words": self._tokens})

        return self._srl_predictor.predict_instances(instances)

    def _parse_span(self, verbose=0):
        """Parse spanument to extract predicates and their semantic roles."""
        assert not contain_empty_token(self._tokens), (
            f'Contain an empty token: {self._tokens}')

        self._srl_dict = self._predict()

        if verbose >= 2:
            print('SRL:', self._srl_dict)

        self._verbs = [SrlVerb(verb['verb'], verb['description'], verb['tags'])
                       for verb in self._srl_dict['verbs']]

    def get_pred_idx_to_srl_verb(self):
        """Return a mapping from the index of a predicate object to a verb obj.
        """
        pred_idx_to_srl_verb = {}
        for srl_verb in self._srl_verbs():
            pred_idx = self.get_pred_idx(srl_verb)
            if pred_idx is not None:
                pred_idx_to_srl_verb[pred_idx] = srl_verb
        return pred_idx_to_srl_verb

    def _tokens_to_instances(self, tokens):
        if self._pred_idx is None:
            return self._tokens_to_instances_all_verbs(tokens)

        verb_labels = [0] * len(tokens)
        verb_labels[self._pred_idx] = 1
        return [self._srl_predictor._dataset_reader.text_to_instance(tokens, verb_labels)]


    def _tokens_to_instances_all_verbs(self, tokens):
        # words = [token.text for token in tokens]
        instances: List[Instance] = []
        for i, word in enumerate(tokens):
            # if word.pos_ == "VERB":
            if word.tag_.startswith("VB"):
                verb_labels = [0 for _ in tokens]
                verb_labels[i] = 1
                instance = self._srl_predictor._dataset_reader.text_to_instance(tokens, verb_labels)
                instances.append(instance)
        return instances

    def get_pred_idx(self, srl_verb: SrlVerb, verbose=0):
        """Get the token offset of the verb predicate."""
        # Check and get the verb.
        # Assume the token separator is space.
        verbs = srl_verb.verb.split()
        if len(verbs) > 1:
            raise ValueError(f'Get a multi-token verb {verbs}')
        verb = verbs[0]

        if verbose >= 2:
            print(verb)

        tags = srl_verb.tags
        tag_counts = Counter(tags)
        if tag_counts['I-V'] > 0:
            pprint(tags)
            print('Warning, get a multi-token verb.')

        # Not found any verb.
        if tag_counts['B-V'] == 0:
            print('Warning, found no verb')
            return None

        if tag_counts['B-V'] > 1:
            pprint(tags)
            print('Warning, get more than 1 verb')

        # Get the index at which the token is the same with the verb value.
        found_idx = -1
        for i, tag in enumerate(tags):
            if tag == 'B-V' and str(self._tokens[i]) == verb:
                if found_idx >= 0:
                    pprint(tags)
                    raise ValueError(
                        'Found multiple positions with the verb value.')
                found_idx = i

        if found_idx == -1:
            print('WARNING: Cannot find the position with the verb value.')
            return None

        return found_idx
