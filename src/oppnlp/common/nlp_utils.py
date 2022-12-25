""" Collections of utils for NLP. """

from typing import List

from colorama import Fore
from spacy.tokens import Span, Doc, Token
from spacy.language import Language


def check_single_sent(doc: Doc):
    """ Check that the doc has only 1 single sentence.
    Do not raise assertion error since nlp model can be wrong on possession marks."""
    num_sents = len(list(doc.sents))
    if num_sents != 1:
        print(Fore.RED + f'Expect doc to a single sentence but got {num_sents} sentences: {doc=}' + Fore.RESET)
        return False

    return True


def get_single_sent(doc: Doc):
    """ Check that the doc has only 1 single sentence and return it. """
    check_single_sent(doc)
    return next(doc.sents)


def parse_doc(nlp: Language, doc: Doc):
    """Parse the Doc using the pipeline."""
    for _, component in nlp.pipeline:
        doc = component(doc)

    return doc


def all_single_char_toks(span: Span):
    """Return True if the span contains single-char tokens."""
    return all(len(tok) == 1 for tok in span)


def get_ent_ranges(doc: Doc, ent_type: str):
    """Get all ranges for an entity type, i.e., pairs of start/end of tokens."""
    for ent in doc.ents:
        if ent[0].ent_type_ == ent_type:
            yield ent[0].i, ent[-1].i


def contain_empty_token(tokens: List[Token]):
    return any(str(t).strip() == '' for t in tokens)


def remove_double_spaces(text: str):
    """Return a new string with double spaces removed."""
    return ' '.join(t.strip() for t in text.split())


# all noun tags in upenn pos.
noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']

def contain_noun_or_prp(span: Span):
    """Check whether the span has any noun or pronoun."""
    return any(t.tag_ in noun_tags + ['PRP'] for t in span)


def lemmatize(tokens):
    return ' '.join(t.lemma_ for t in tokens)


def get_lemma_pron(tok):
    return tok.lemma_ if tok.lemma_ != u'-PRON-' else tok.text


def lemmatize_pron(tokens):
    """Lemmatize pronouns."""
    return ' '.join(get_lemma_pron(t) for t in tokens)
