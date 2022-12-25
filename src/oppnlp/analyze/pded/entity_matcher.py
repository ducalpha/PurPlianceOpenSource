""" Contains entity matchers for PDED recognition pipeline. """

from spacy.pipeline import EntityRuler


class PronounEntityMatcher:
    """ Match entities which are pronouns such as 'we', 'you', 'us'. """
    name = 'pronoun_entity_matcher'

    def __init__(self, nlp):
        self._matcher = EntityRuler(nlp)
        for pronoun in ['we', 'i', 'you', 'me', 'us']:
            self._matcher.add_patterns([
                {'label': 'ORG', 'pattern': [{'LOWER': pronoun}]}
            ])

    def __call__(self, doc):
        return self._matcher(doc)
