"""Utilities for extracting and analyzing policies."""

import spacy

class DependencyGraphConstructor:
    @classmethod
    def isVerbNegated(cls, token, sentence):
        def isVerbNegatedInternal(token):
            return any(t.dep == spacy.symbols.neg for t in token.children)

        if isVerbNegatedInternal(token):
            return True

        # Check if verb is part of conjugated verb phrase, if so, check if any of those are negated
        conjugatedVerbs = cls.getConjugatedVerbs(sentence, token)
        for tok in conjugatedVerbs:
            if isVerbNegatedInternal(tok):
                return True

        # Check if verb is xcomp, if so check if prior verb is negated?
        #TODO should also do advcl
        if token.dep == spacy.symbols.xcomp or token.dep == spacy.symbols.advcl:
            return cls.isVerbNegated(token.head, sentence)
        return False

    @classmethod
    def getConjugatedVerbs(cls, sentence, targetTok = None):
        def isComma(token):
            return token.pos_ == 'PUNCT' and token.text == ','

        def isCConj(token):
            return token.pos == spacy.symbols.CCONJ and token.lemma_ in ['and', 'or', 'nor']

        def isNegation(token):
            return token.dep == spacy.symbols.neg

        def getConjugatedVerbsInternal(results, token):
            if token.pos == spacy.symbols.VERB:
                results.append(token)
            for tok in token.children:
                if tok.i < token.i:#Ensure we only look at children that appear AFTER the token in the sentence
                    continue
                if tok.dep == spacy.symbols.conj and tok.pos == spacy.symbols.VERB:
                    if not getConjugatedVerbsInternal(results, tok):
                        return False
                elif not (isComma(tok) or isCConj(tok) or isNegation(tok)):
                    return False
            return True

        def isTokenContainedIn(token, conjugatedVerbs):
            for vbuffer in conjugatedVerbs:
                if token in vbuffer:
                    return True
            return False

        conjugatedVerbs = []
        vbuffer = []
        for token in sentence:
            if token.pos == spacy.symbols.VERB:
                # Make sure we didn't already cover the verb...
                if isTokenContainedIn(token, conjugatedVerbs):
                    continue

                vbuffer = []
                getConjugatedVerbsInternal(vbuffer, token)
                if len(vbuffer) > 1:
                    conjugatedVerbs.append(vbuffer)

        if targetTok != None:
            for vbuffer in conjugatedVerbs:
                if targetTok in vbuffer:
                    return vbuffer
            return []
        return conjugatedVerbs
