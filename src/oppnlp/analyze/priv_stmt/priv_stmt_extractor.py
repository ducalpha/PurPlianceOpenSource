"""Extract PS parameters: sender, receiver, purpose and condition."""

from pathlib import Path
from typing import Dict, List
import json

from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from oppnlp.analyze.priv_stmt.doc_maker import DocMaker
from oppnlp.common.data_file_utils import get_sent_doc_iter


class PrivStmt:
    _attrs = ['sent', 'receiver', 'neg', 'action', 'action_lemma', 'sender', 'data', 'purpose', 'pred_idx', 'adv']

    def __init__(self, sent, receiver, neg, action, action_lemma, sender, data, purpose, pred_idx, adv):
        self.sent = sent
        self.receiver = receiver
        self.neg = neg
        self.action = action
        self.action_lemma = action_lemma
        self.sender = sender
        self.receiver = receiver
        self.data = data
        self.purpose = purpose
        self.pred_idx = pred_idx
        self.adv = adv

    def get(self, key):
        if key in self:
            return self[key]
        return None

    def __str__(self):
        str_dict = {k: str(v) for k, v in self._asdict().items()}
        return json.dumps(str_dict)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, item):
        return item in self._attrs

    def __eq__(self, other):
        for attr in self._attrs:
            self_attr = getattr(self, attr)
            other_attr = getattr(other, attr)
            # In case "we_implicit" and span we.
            if type(self_attr) != type(other_attr) or self_attr != other_attr:
                return False

        return True

    def __lt__(self, other):
        return str(self) < str(other)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def _asdict(self):
        return {attr: getattr(self, attr) for attr in self._attrs}


class PrivStmtExtractor:
    """Privacy statement extractor."""

    _nlp = DocMaker()

    @classmethod
    def get_priv_stmts_from_sent_doc(cls, doc: Doc, post_process=True):
        raise NotImplementedError

    @classmethod
    def get_priv_stmts_as_ents_in_doc(cls, doc, verbose=0):
        """Return a doc with PS params as entities."""
        priv_stmts = cls.get_priv_stmts_from_sent_doc(doc)

        if verbose >= 2:
            print('get_priv_stmt_as_entities(), priv_stmts', priv_stmts)

        # Set the entities of the sentence to the priv_stmt in the list.
        ps_entities = []
        for priv_stmt in priv_stmts:
            for ps_param, ps_spans in priv_stmt.items():
                for span in ps_spans:
                    ps_entities.append(
                        Span(span.doc, span.start, span.end, label=ps_param))

        # Must filter, in case of same condition for different actions.
        doc.ents = filter_spans(ps_entities)
        return doc

    @classmethod
    def get_priv_stmts_as_ents_in_doc_from_file(cls, in_file: Path, single_sent_txt):
        """Extract PS from a sentence file. Return single-sent docs with PS as
        entities."""
        return [cls.get_priv_stmts_as_ents_in_doc(sent_doc)
                for sent_doc in get_sent_doc_iter(cls._nlp, in_file, single_sent_txt)]

    @classmethod
    def extract_priv_stmts_from_file(cls, in_file: Path, single_sent_txt: bool):
        """Extract and return PS statements from the file."""
        priv_stmts: List[Dict] = []
        for sent_doc in get_sent_doc_iter(cls._nlp, in_file, single_sent_txt):
            sent_priv_stmts = cls.get_priv_stmts_from_sent_doc(sent_doc)
            priv_stmts.extend(sent_priv_stmts)

        return priv_stmts
