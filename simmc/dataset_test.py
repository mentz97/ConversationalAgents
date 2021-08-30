from .dataset import GetAPI

import collections


def test_GetAPI():
    res = GetAPI(train=True, return_counter=True)

    assert all(i in res.keys() for i in ['counter', 'results'])
    assert res['counter'] == collections.Counter({'price': 2803, 'availableSizes': 2738, 'customerRating': 1040, 'brand': 891, 'info': 747, 'color': 363, 'embellishment': 291, 'pattern': 274, 'hemLength': 190, 'skirtStyle': 133, 'dressStyle': 109, 'material': 101, 'clothingStyle': 89, 'necklineStyle': 77, 'size': 68, 'jacketStyle': 56,
                                                  'sweaterStyle': 47, 'hemStyle': 41, 'sleeveStyle': 38, 'waistStyle': 31, 'sleeveLength': 15, 'clothingCategory': 7, 'skirtLength': 6, 'soldBy': 4, 'madeIn': 2, 'ageRange': 2, 'forGender': 1, 'waterResistance': 1, 'warmthRating': 1, 'sequential': 1, 'hasPart': 1, 'amountInStock': 1, 'forOccasion': 1})
    assert all(i in res['results'][0].keys() for i in ['turn_id', 'sentence',
                                                       'action', 'attributes'])
    assert res['results'][0]['turn_id'] == 0
    assert res['results'][0]['sentence'] == "Is there a pattern on this one? It's hard to see in the image."
    assert res['results'][0]['action'] == 'SpecifyInfo'
    assert res['results'][0]['attributes'] == ['pattern']


def test_GetAPI_withMinAttributeOcc(): 
    res = GetAPI(train=True, min_attribute_occ=15,
                 return_excluded_attributes=True)

    assert set([a for x in res['results'] for a in x['attributes']]) == {'info', 'necklineStyle', 'clothingStyle', 'skirtStyle', 'pattern', 'customerRating', 'sleeveLength', 'material',
                                                                         'waistStyle', 'price', 'hemStyle', 'jacketStyle', 'brand', 'size', 'color', 'embellishment', 'dressStyle', 'hemLength', 'sweaterStyle', 'availableSizes', 'sleeveStyle'}
    assert res['excluded_attributes'] == ['clothingCategory', 'madeIn', 'skirtLength', 'ageRange', 'soldBy',
                                          'forGender', 'waterResistance', 'warmthRating', 'sequential', 'hasPart', 'amountInStock', 'forOccasion']


def test_GetAPI_withMinAttributeOccAndExcludeList():
    res = GetAPI(train=True, min_attribute_occ=15, exclude_attributes=[
                 'info'], return_excluded_attributes=True)
    assert set([a for x in res['results'] for a in x['attributes']]) == {'necklineStyle', 'clothingStyle', 'skirtStyle', 'pattern', 'customerRating', 'sleeveLength', 'material',
                                                                         'waistStyle', 'price', 'hemStyle', 'jacketStyle', 'brand', 'size', 'color', 'embellishment', 'dressStyle', 'hemLength', 'sweaterStyle', 'availableSizes', 'sleeveStyle'}
    assert res['excluded_attributes'] == ['info', 'clothingCategory', 'madeIn', 'skirtLength', 'ageRange', 'soldBy',
                                          'forGender', 'waterResistance', 'warmthRating', 'sequential', 'hasPart', 'amountInStock', 'forOccasion']
