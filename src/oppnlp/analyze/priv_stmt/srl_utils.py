"""Semantic role labeler."""

from allennlp.predictors.predictor import Predictor

from oppnlp.common.gpu_utils import free_gpu


# For allennlp 1.1.0:
srl_bert_archive = "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.09.03.tar.gz"

# Caching the srl predictor.
_srl_predictor = None


def _load_srl_predictor():
    # Use the first GPU if available.
    if free_gpu >= 0:
        print('Load SRL on GPU', free_gpu)
    else:
        print('Load SRL on CPU, may run slower than on GPU.')

    return Predictor.from_path(srl_bert_archive, cuda_device=free_gpu)


def load_srl_predictor(archive_path: str=srl_bert_archive,
                       language_model_name: str = "en_core_web_lg",
                       dataset_reader_to_load: str = "validation",
                       verbose=2):
    """Load the SRL predictor. Customized from predictors.Predictor"""
    global _srl_predictor
    if _srl_predictor is None:
        if verbose >= 2:
            print('Start loading SRL predictor.')

        _srl_predictor = _load_srl_predictor()

        if verbose >= 2:
            print('SRL predictor loaded.')
    return _srl_predictor
