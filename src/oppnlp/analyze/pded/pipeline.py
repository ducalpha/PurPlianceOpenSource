""" Contains NER components for Spacy pipelines. """

from pathlib import Path
from timeit import timeit

from spacy.language import Language
from spacy.tokens import Doc
import spacy

from oppnlp.analyze.pded.entity_matcher import PronounEntityMatcher
from oppnlp.common.nlp_utils import all_single_char_toks
from oppnlp.common.gpu_utils import free_gpu


model_dir = Path(__file__).parent / 'models' / 'en_core_web_lg.high_f1_data_org.model'
assert model_dir.exists(), f'{model_dir} not exist'


def _load_nlp(model_dir_or_name):
    """Load nlp with time measurement."""
    def _get_nlp():
        """Wrapper function for timeit()."""
        nonlocal nlp
        nlp = spacy.load(model_dir_or_name)

    if free_gpu >= 0: # Note: using gpu breaks multiprocessing.
        if spacy.__version__ > '2.3.0':
            print(f'Spacy require_gpu on gpu {free_gpu}')
            spacy.require_gpu(free_gpu)
        else:
            print(f'Spacy loaded on cpu, may run slower than gpu.')

    nlp = None
    print(f'Loading model from {model_dir_or_name} ...')
    print(f'Done loading model, took:', timeit(_get_nlp, number=1), 'seconds.')
    return nlp


def remove_single_char_data_objects(doc: Doc):
    """Remove data objects which are tokens with single characters."""
    doc.ents = list(filter(lambda ent: not (ent.label_ == 'DATA'
                                       and all_single_char_toks(ent)),
                           doc.ents))
    return doc


class _PrivacyNlp:
    """Language pipeline trained for NER data types and has support for DED trees."""
    _nlp: Language = None

    @classmethod
    def get_nlp(cls):
        if cls._nlp is None:
            cls._nlp = _load_nlp(model_dir)

            if 'tagger' not in cls._nlp.pipe_names:
                tagger = cls._nlp.create_pipe("tagger")
                cls._nlp.add_pipe(tagger, first=True)

            if 'parser' not in cls._nlp.pipe_names:
                parser = cls._nlp.create_pipe("parser")
                cls._nlp.add_pipe(parser, first=True)

            cls._nlp.add_pipe(remove_single_char_data_objects)
            cls._nlp.add_pipe(PronounEntityMatcher(cls._nlp))
            cls._nlp.add_pipe(cls._nlp.create_pipe('merge_entities'))

            print(f'Done loading language model.')

        return cls._nlp


def get_nlp() -> Language:
    return _PrivacyNlp.get_nlp()


def get_unmerging_nlp(verbose=0):
    """Get a nlp pipeline with no merging for the named entities and noun chunks."""
    nlp = get_nlp()

    remove_pipes = ['merge_noun_chunks', 'merge_entities']
    for pipe in remove_pipes:
        if nlp.has_pipe(pipe):
            nlp.remove_pipe(pipe)

    if verbose >= 2:
        for pipe in nlp.pipeline:
            print(pipe)

    return nlp
