import json
import os
import collections

from enum import Enum


__location__ = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))

__trainFile = os.path.join(__location__, 'data/fashion_train_dials.json')
__trainAPIFile = os.path.join(
    __location__, 'data/fashion_train_dials_api_calls.json')
__valFile = os.path.join(__location__, 'data/fashion_dev_dials.json')
__valAPIFile = os.path.join(
    __location__, 'data/fashion_dev_dials_api_calls.json')


def GetAPI(train: bool = False,
           return_turn_ids: bool = True,
           return_sentences: bool = True,
           return_actions: bool = True,
           return_attributes: bool = True,
           return_counter: bool = False,
           return_excluded_attributes: bool = False,
           min_attribute_occ: int = 0,
           exclude_attributes: list[str] = []):
    turn_ids = []
    sentences = []
    actions = []
    attributes_list = []
    counter = []

    obj = {}

    if all([return_turn_ids, return_sentences, return_actions, return_attributes]) is False:
        return None

    with open(__trainFile if train else __valFile, 'r') as file:
        data = json.load(file)

        dialogues = list(map(lambda d: d['dialogue'], data['dialogue_data']))

        if return_turn_ids:
            turn_ids = [sentence['turn_idx']
                        for dialogue in dialogues for sentence in dialogue]
        if return_sentences:
            sentences = [sentence['transcript']
                         for dialogue in dialogues for sentence in dialogue]

    with open(__trainAPIFile if train else __valAPIFile, 'r') as file:
        data = json.load(file)

        if return_actions:
            actions = [j['action'] for i in data for j in i['actions']]

        if return_attributes:
            supervisions = [j['action_supervision']
                            for i in data for j in i['actions']]
            attributes_list = list(
                map(lambda x: x['attributes'] if x is not None else [], supervisions))

            counter = collections.Counter(
                [y for x in list(filter(None, attributes_list)) for y in x])

            excluded_attributes = [key for key,
                                   val in counter.items() if val < min_attribute_occ]

            if return_counter:
                obj['counter'] = counter

            if return_excluded_attributes:
                obj['excluded_attributes'] = excluded_attributes

    obj['results'] = []
    for i, (turn_id, sentence, action, attributes) in enumerate(zip(turn_ids, sentences, actions, attributes_list), start=0):
        obj['results'].append({})
        if return_turn_ids:
            obj['results'][i]['turn_id'] = turn_id
        if return_sentences:
            obj['results'][i]['sentence'] = sentence
        if return_actions:
            obj['results'][i]['action'] = action
        if return_attributes:
            obj['results'][i]['attributes'] = [
                x for x in attributes if x not in excluded_attributes or x not in exclude_attributes]

    return obj
