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
from collections import Counter

def read_image_feature(path, item_ids):
    print('Reading image feature...')
    item_feature = {}
    f = open(path, 'rb')
    count = 0
    while True:
        if count % 35808 == 0 or count == 358078:
            print('{}%'.format(int(count / 358078 * 100)))
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
    feedback_pairs = set()
    user_count = Counter()
    for line in gzip.open(path, 'r').readlines():
        json_acceptable_str = line.decode('ascii').replace("'", "\"")
        json_obj = json.loads(json_acceptable_str)
        uid = json_obj['uid']
        for action in ['bought', 'want']:
            for iid in json_obj['lists'][action]:
                feedback_pairs.add((uid, iid))
                user_count[uid] += 1
    # discard user with less than 5 feedback
    triplet_data = []
    item_ids = set()
    for uid, iid in feedback_pairs:
        if user_count[uid] < 5:
            continue
        triplet_data.append([uid, iid, 1])
        item_ids.add(iid)
    return triplet_data, item_ids

# Download purchase user data
user_data_path = cache(url='http://jmcauley.ucsd.edu/data/tradesy/tradesy.json.gz')
triplet_data, item_ids = read_user_data(user_data_path)

# Download visual feature (BIG file)
visual_feature_path = cache(url='http://jmcauley.ucsd.edu/data/tradesy/image_features_tradesy.b')
item_feature = read_image_feature(visual_feature_path, item_ids)
item_image_module = ImageModule(id_feature=item_feature, normalized=True)

ratio_split = RatioSplit(data=triplet_data,
                         test_size=0.01,
                         rating_threshold=0.5,
                         exclude_unknowns=True,
                         verbose=True,
                         item_image=item_image_module)

vbpr = cornac.models.VBPR(k=10, k2=20, n_epochs=50, batch_size=100, learning_rate=0.005,
                          lambda_w=1, lambda_b=0.01, lambda_e=0.0, use_gpu=True)

auc = cornac.metrics.AUC()
rec_50 = cornac.metrics.Recall(k=50)

exp = cornac.Experiment(eval_method=ratio_split,
                        models=[vbpr],
                        metrics=[auc, rec_50])
exp.run()
