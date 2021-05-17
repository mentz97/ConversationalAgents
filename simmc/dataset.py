import json
import sklearn.preprocessing
import pkg_resources

from enum import Enum

__train_data = pkg_resources.resource_filename(
    'simmc', 'data/fashion_train_dials.json')
__train_api = pkg_resources.resource_filename(
    'simmc', 'data/fashion_train_dials_api_calls.json')
__val_data = pkg_resources.resource_filename(
    'simmc', 'data/fashion_dev_dials.json')
__val_api = pkg_resources.resource_filename(
    'simmc', 'data/fashion_dev_dials_api_calls.json')


class Mode(Enum):
    Train = 0,
    Validation = 1


def GetSentences(mode: Mode):
    if mode == Mode.Train:
        filePath = __train_data
    elif mode == Mode.Train:
        filePath = __val_data

    with open(filePath, 'r') as file:
        data = json.load(file)

        dialogues = list(map(lambda d: d['dialogue'], data['dialogue_data']))

        turn_ids = [sentence['turn_idx']
                    for dialogue in dialogues for sentence in dialogue]
        sentences = [sentence['transcript']
                     for dialogue in dialogues for sentence in dialogue]

        return turn_ids, sentences


def GetAPI(mode: Mode, return_attributes_binary: bool = False):
    if mode == Mode.Train:
        filePath = __train_api
    elif mode == Mode.Train:
        filePath = __val_api

    with open(filePath, 'r') as file:
        data = json.load(file)

        actions = [j['action'] for i in data for j in i['actions']]
        supervisions = [j['action_supervision']
                        for i in data for j in i['actions']]

        attributes = list(
            map(lambda x: x['attributes'] if x is not None else [], supervisions))

        if return_attributes_binary:
            attributes = sklearn.preprocessing.MultiLabelBinarizer().fit_transform(attributes)

        return actions, attributes
