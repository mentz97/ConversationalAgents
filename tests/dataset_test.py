import pytest
import numpy as np
from simmc import GetSentences, GetAPI, Mode


def test_train_sentences():
    ids, sentences = GetSentences(Mode.Train)

    assert len(ids) == 21196
    assert len(sentences) == 21196

    assert sentences[0] == "Is there a pattern on this one? It's hard to see in the image."
    assert ids[0] == 0


def test_train_api():
    actions, attributes = GetAPI(Mode.Train)

    assert len(actions) == 21196
    assert len(attributes) == 21196

    assert actions[0] == 'SpecifyInfo'

    assert len(attributes[0]) == 1
    assert attributes[0][0] == 'pattern'
    assert len(attributes[1]) == 0
    assert len(attributes[19]) == 2


def test_train_api_binary():
    actions, attributes = GetAPI(Mode.Train, return_attributes_binary=True)

    assert len(actions) == 21196
    assert len(attributes) == 21196

    assert len(attributes[0]) == 33
    assert np.count_nonzero(attributes[0]) == 1
    assert len(attributes[1]) == 33
    assert np.count_nonzero(attributes[1]) == 0
    assert len(attributes[19]) == 33
    assert np.count_nonzero(attributes[19]) == 2
