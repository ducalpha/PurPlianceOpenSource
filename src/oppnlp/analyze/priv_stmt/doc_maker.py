"""Get Spacy doc from string."""

from spacy.tokens import Doc

from oppnlp.analyze.pded.pipeline import get_unmerging_nlp
from oppnlp.common.nlp_utils import parse_doc

class DocMaker:
    """Cache Spacy Doc."""

    _text_to_doc_data = {}
    _nlp = None

    @classmethod
    def _get_nlp(cls):
        if not cls._nlp:
            cls._nlp = get_unmerging_nlp()
        return cls._nlp

    @classmethod
    def get_doc(cls, sentence, invalidate=False, verbose=0):
        doc_data = cls._text_to_doc_data.get(sentence)
        nlp = cls._get_nlp()
        if doc_data is None:
            if verbose >= 2:
                print('DocMaker cache missed')
            doc = nlp(sentence)
            cls.set_doc(sentence, doc)
            return doc

        if verbose >= 2:
            print('DocMaker cache hit')
        return Doc(nlp.vocab).from_bytes(doc_data)


    @classmethod
    def set_doc(cls, sentence, doc):
        cls._text_to_doc_data[sentence] = doc.to_bytes(exclude=['tensor'])


    @classmethod
    def parse_single_sentence(cls, sentence):
        """Force parsing single sentence."""
        line_doc = Doc(cls._get_nlp().vocab, words=sentence.split())
        for i in range(len(line_doc)):
            if i == 0:
                line_doc[i].is_sent_start = True
            else:
                line_doc[i].is_sent_start = False
        line_doc = parse_doc(cls._get_nlp(), line_doc)
        cls.set_doc(sentence, line_doc)
        return line_doc


    def __call__(self, text):
        return self.get_doc(text)
