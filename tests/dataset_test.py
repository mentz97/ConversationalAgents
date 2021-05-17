import pytest
import numpy as np
from simmc import GetSentences, GetAPI


def test_train_sentences():
    ids, sentences = GetSentences(train=True)

    assert len(ids) == 21196
    assert len(sentences) == 21196

    assert sentences[0] == "Is there a pattern on this one? It's hard to see in the image."
    assert ids[0] == 0


def test_train_api():
    actions, attributes = GetAPI(train=True)

    assert len(actions) == 21196
    assert len(attributes) == 21196

    assert actions[0] == 'SpecifyInfo'

    assert len(attributes[0]) == 1
    assert attributes[0][0] == 'pattern'
    assert len(attributes[1]) == 0
    assert len(attributes[19]) == 2
