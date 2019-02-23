# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.eval_methods import BaseMethod
from cornac.data import reader


def test_init():
    bm = BaseMethod(None, verbose=True)

    assert not bm.exclude_unknowns
    assert 1. == bm.rating_threshold


def test_trainset_none():
    bm = BaseMethod(None, verbose=True)

    try:
        bm.evaluate(None, {}, False)
    except ValueError:
        assert True


def test_testset_none():
    from cornac.data import TrainSet

    bm = BaseMethod(None, verbose=True)
    bm.train_set = TrainSet(None, None)

    try:
        bm.evaluate(None, {}, False)
    except ValueError:
        assert True


def test_from_splits():
    data = reader.read_uir('./tests/data.txt')

    try:
        BaseMethod.from_splits(train_data=None, test_data=None)
    except ValueError:
        assert True

    try:
        BaseMethod.from_splits(train_data=data, test_data=None)
    except ValueError:
        assert True

    bm = BaseMethod.from_splits(train_data=data, test_data=data)
    assert bm.total_users == 10
    assert bm.total_items == 10


    bm = BaseMethod.from_splits(train_data=data, test_data=data,
                                val_data=data, verbose=True)
    assert bm.total_users == 10
    assert bm.total_items == 10


def test_with_modules():
    from cornac.data import TextModule, ImageModule, GraphModule

    bm = BaseMethod()

    assert bm.user_text is None
    assert bm.item_text is None
    assert bm.user_image is None
    assert bm.item_image is None
    assert bm.user_graph is None
    assert bm.item_graph is None

    bm.user_text = TextModule()
    bm.item_graph = GraphModule()
    bm._build_modules()

    try:
        bm.user_text = ImageModule()
    except ValueError:
        assert True

    try:
        bm.item_text = GraphModule()
    except ValueError:
        assert True

    try:
        bm.user_image = TextModule()
    except ValueError:
        assert True

    try:
        bm.item_image = GraphModule()
    except ValueError:
        assert True

    try:
        bm.user_graph = TextModule()
    except ValueError:
        assert True

    try:
        bm.item_graph = ImageModule()
    except ValueError:
        assert True


def test_organize_metrics():
    from cornac.metrics import MAE, AUC

    bm = BaseMethod()

    rating_metrics, ranking_metrics = bm._organize_metrics([MAE(), AUC()])
    assert 1 == len(rating_metrics) # MAE
    assert 1 == len(ranking_metrics) # AUC

    try:
        bm._organize_metrics(None)
    except ValueError:
        assert True