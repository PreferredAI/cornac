# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Example for HypAR, using the Cellphone dataset
"""
import os
from typing import List

import cornac
from cornac.data import Reader, SentimentModality, ReviewModality
from cornac.data.text import BaseTokenizer
from cornac.eval_methods import StratifiedSplit
from cornac.metrics import NDCG, AUC, MAP, MRR, Recall, Precision


# Download datsets from: https://drive.google.com/drive/folders/1DZAUJ4aO5O4SLxCFGS3MYBQhn9DlZ7qE
# Extract the datasets to the current working directory
def dataset_converter(dataset_name):
    import pandas as pd
    print(f'--- Loading {dataset_name} ---')
    df = pd.read_csv(os.path.join('seer-ijcai2020', dataset_name, 'profile.csv'), sep=',')
    # df.drop(columns=['sentire_aspect', 'aspect_pos', 'opinion_pos', 'sentence_len', 'sentence_count'], inplace=True)

    # reviewerID,asin,overall,unixReviewTime,aspect_pos,aspect,opinion_pos,opinion,sentence,sentence_len,sentence_count,sentiment

    # Creating rating file
    ra_path = os.path.join('seer-ijcai2020', dataset_name, 'ratings.txt')
    if not os.path.isfile(ra_path):
        print('Removing duplicates')
        ratings = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']].drop_duplicates()

        print('Saving ratings')
        ratings.to_csv(ra_path, index=False, header=False, sep=',')
        del ratings
    else:
        print('Warning: ratings.txt already exists, skipping.')

    df.drop(columns=['overall', 'unixReviewTime'], inplace=True)  # space efficiency

    # Creating review file
    re_path = os.path.join('seer-ijcai2020', dataset_name, 'review.txt')
    if not os.path.isfile(re_path):
        print('Getting sentences')
        review = df.groupby(['reviewerID', 'asin'], as_index=False).agg({'sentence': '.'.join})

        print('Joining sentences')
        review['sentence'] = review['sentence'].apply(lambda x: x.replace('\t', ' '))

        print('Saving reviews')
        review.to_csv(re_path, index=False, header=False, sep='\t')
        del review
    else:
        print('Warning: review.txt already exists, skipping.')

    df.drop(columns=['sentence'], inplace=True)  # space efficiency

    # Creating sentiment file
    s_path = os.path.join('seer-ijcai2020', dataset_name, 'sentiment.txt')
    if not os.path.isfile(s_path):
        print('Joining aspects, opinions and sentiments')
        df['aspect:opinion:sentiment'] = df.apply(lambda x: f"{x['aspect']}:{x['opinion']}:{x['sentiment']}", axis=1)
        df.drop(['aspect', 'opinion', 'sentiment'], axis=1, inplace=True)

        print('Assigning aspects, opinions and sentiments to reviews')
        aos = df.groupby(['reviewerID', 'asin'], as_index=False).agg({'aspect:opinion:sentiment': ','.join})

        print('Saving sentiment')
        with open(s_path, 'w') as f:
            for _, (r, a, t) in aos.iterrows():
                f.write(f'{r},{a},{t}\n')
    else:
        print('Warning: sentiment.txt already exists, skipping.')


def load_feedback(fmt="UIR", reader: Reader = None) -> List:
    """Load the user-item ratings, scale: [1,5]

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating).
    """
    fpath = 'seer-ijcai2020/cellphone/ratings.txt'
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep=",")


def load_review(reader: Reader = None) -> List:
    """Load the user-item-review list

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, review).
    """
    fpath = 'seer-ijcai2020/cellphone/review.txt'
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt="UIReview", sep="\t")


def load_sentiment(reader: Reader = None) -> List:
    """Load the user-item-sentiments
    The dataset was constructed by the method described in the reference paper.

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, [(aspect, opinion, sentiment), (aspect, opinion, sentiment), ...]).

    References
    ----------
    https://github.com/evison/Sentires
    """
    fpath = 'seer-ijcai2020/cellphone/sentiment.txt'
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt='UITup', sep=',', tup_sep=':')

dataset_converter('cellphone')
feedback = load_feedback(fmt="UIRT", reader=Reader())
reviews = load_review()
sentiment = load_sentiment(reader=Reader())


# Instantiate an evaluation method to split data into train and test sets.
sentiment_modality = SentimentModality(data=sentiment)

review_modality = ReviewModality(
    data=reviews,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=4000,
    max_doc_freq=0.5,
)

eval_method = StratifiedSplit(
    feedback,
    group_by="user",
    chrono=True,
    sentiment=sentiment_modality,
    review_text=review_modality,
    test_size=0.2,
    val_size=0.16,
    exclude_unknowns=True,
    seed=42,
    verbose=True,
    )

# Instantiate the HypAR model, score: 0.205963068063327
hypar = cornac.models.HypAR(
    use_cuda=False,
    stemming=True,
    batch_size=256,
    num_workers=2,
    num_epochs=500,
    early_stopping=25,
    eval_interval=1,
    learning_rate=0.001,
    weight_decay=0.001,
    l2_weight=0.,
    node_dim=64,
    num_heads=3,
    fanout=-1,
    non_linear=True,
    model_selection='best',
    objective='ranking',
    review_aggregator='narre',
    predictor='dot',
    preference_module='lightgcn',
    combiner='concat',
    graph_type='aos',
    num_neg_samples=50,
    layer_dropout=.2,
    attention_dropout=.2,
    user_based=True,
    verbose=True,
    index=0,
    out_path=os.path.abspath(os.curdir),
    learn_explainability=True,
    learn_method='transr',
    learn_weight=0.5,
    embedding_type='ao_embeddings',
    debug=False
)

# Instantiate evaluation measures
metrics = [NDCG(), NDCG(20), NDCG(100), AUC(), MAP(), MRR(), Recall(10), Recall(20), Precision(10), Precision(20)]

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=eval_method, models=[hypar], metrics=metrics,
    user_based=True, verbose=True
).run()
