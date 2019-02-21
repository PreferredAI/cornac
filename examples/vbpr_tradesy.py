# -*- coding: utf-8 -*-

"""
Example for Visual Bayesian Personalized Ranking
Data: http://jmcauley.ucsd.edu/data/tradesy/

@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import cornac
from cornac.utils.download import cache
from cornac.data import ImageModule
from cornac.eval_methods import RatioSplit

import numpy as np
import gzip
import json
import struct

def read_image_feature(path, item_ids):
    print('Reading image feature...')
    item_feature = {}
    f = open(path, 'rb')
    count = 0
    while True:
        if count % 35807 == 0:
            print('{}%'.format(int(count / 358079 * 100)))
        count += 1
        asin = f.read(10).decode('ascii').strip()
        if asin == '':
            break
        if asin not in item_ids: # ignore unwanted items
            f.read(4096 * 4)
            continue
        feature = []
        for i in range(4096):
            feature.append(struct.unpack('f', f.read(4)))
        item_feature[asin] = np.asarray(feature).ravel()
    return item_feature

def read_user_data(path):
    triplet_data = []
    item_ids = set()
    for line in gzip.open(path, 'r').readlines():
        json_acceptable_str = line.decode('ascii').replace("'", "\"")
        json_obj = json.loads(json_acceptable_str)
        uid = json_obj['uid']
        for action in ['bought', 'want']:
            for iid in json_obj['lists'][action]:
                triplet_data.append([uid, iid, 1])
                item_ids.add(iid)
    return triplet_data, item_ids

# Download purchase user data
user_data_path = cache(url='http://jmcauley.ucsd.edu/data/tradesy/tradesy.json.gz')
triplet_data, item_ids = read_user_data(user_data_path)

# Download visual feature (BIG file)
visual_feature_path = cache(url='http://jmcauley.ucsd.edu/data/tradesy/image_features_tradesy.b')
item_feature = read_image_feature(visual_feature_path, item_ids)
item_image_module = ImageModule(id_feature=item_feature)

ratio_split = RatioSplit(data=triplet_data,
                         test_size=0.2,
                         exclude_unknowns=True,
                         verbose=True,
                         item_image=item_image_module)

vbpr = cornac.models.VBPR(k=10, d=10, n_epochs=20, batch_size=100, learning_rate=0.001,
                          lambda_t=0.1, lambda_b=0.01, lambda_e=0.0, use_gpu=True)
rec_20 = cornac.metrics.Recall(k=20)

exp = cornac.Experiment(eval_method=ratio_split,
                        models=[vbpr],
                        metrics=[rec_20])
exp.run()
