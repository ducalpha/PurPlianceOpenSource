"""Utility functions for data files."""

from pathlib import Path
from typing import Iterable
import re

from spacy.tokens import Doc
from unidecode import unidecode

from oppnlp.analyze.priv_stmt.doc_maker import DocMaker
from oppnlp.common.nlp_utils import parse_doc, remove_double_spaces


def read_sents_file(sents_file_path):
    """Read a sents file path (1 sent per)"""
    sents = []
    with open(sents_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents


def cleanupUnicodeErrors(term):
    # https://github.com/benandow/PrivacyPolicyAnalysis/
    t = re.sub(u'\ufffc', u' ', term)
    t = re.sub(u'â€œ', u'', t)
    t = re.sub(u'â€\u009d', u'', t)
    t = re.sub(u'â\u0080\u0094', u'', t)
    t = re.sub(u'â\u0080\u009d', u'', t)
    t = re.sub(u'â\u0080\u009c', u'', t)
    t = re.sub(u'â\u0080\u0099', u'', t)
    t = re.sub(u'â€', u'', t)
    t = re.sub(u'äë', u'', t)
    t = re.sub(u'ä', u'', t)
    t = re.sub(u'\u0093', u'', t)
    t = re.sub(u'\u0092', u'', t)
    t = re.sub(u'\u0094', u'', t)
    t = re.sub(u'\u00a7', u'', t)#Section symbol
    t = re.sub(u'\u25cf', u'', t)#bullet point symbol
    t = re.sub(u'´', u'\'', t)
    t = re.sub(u'\u00ac', u'', t)
    t = re.sub(u'\u00ad', u'-', t)
    t = re.sub(u'\u2211', u'', t)
    t = re.sub(u'\ufb01', u'fi', t)
    t = re.sub(u'\uff0c', u', ', t)
    t = re.sub(u'\uf0b7', u'', t)
    t = re.sub(u'\u037e', u';', t)
    # Duc
    t = re.sub(u'\u2019', u'\'', t)
    t = re.sub(u'\u2014', u'-', t)
    return t


def transliterate_to_ascii(text):
    """Unicode to ascii."""
    return unidecode(text)


def preprocess_line(text):
    """Do some preprocessing on the line."""
    text = transliterate_to_ascii(text)
    text = cleanupUnicodeErrors(text)
    text = remove_double_spaces(text)

    return text


def get_sent_doc_iter(nlp, in_file: Path, single_sent_txt: bool, preprocess=True, verbose=0) -> Iterable[Doc]:
    """Get single-sentence Doc iterator for the in_file."""
    if in_file.suffix in ['.sent', '.sents', '.txt']:
        if in_file.suffix == '.txt' and verbose >= 2:
            print('Assume this txt file to be a sent file')
        sents_iter = read_sents_file(in_file)
    else:
        raise ValueError(f'Unsupported file format {in_file}')

    # Process sentences one by one.
    for sentence in sents_iter:
        # Manually set sentence boundaries for conll03 files.
        if in_file.suffix == '.conll03':
            line_doc = Doc(nlp.vocab, words=sentence.split())

            for i in range(len(line_doc)):
                if i == 0:
                    line_doc[i].is_sent_start = True
                else:
                    line_doc[i].is_sent_start = False

            line_doc = parse_doc(nlp, line_doc)
        else:
            if preprocess:
                sentence = preprocess_line(sentence)
                if len(sentence) == 0: continue

            line_doc = nlp(sentence)

            if single_sent_txt and len(list(line_doc.sents)) > 1 and isinstance(nlp, DocMaker):
                print("******* Cache missed. Reparse the single line as a single sentence.")
                line_doc = nlp.parse_single_sentence(sentence)

        for sent_doc in line_doc.sents:
            yield sent_doc.as_doc()
