# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np
from cornac.utils.common import sigmoid
from cornac.utils.common import safe_indexing
from cornac.utils.common import validate_format
from cornac.utils.common import scale
from cornac.utils.common import clip
from cornac.utils.common import excepts
from cornac.utils.common import intersects


def test_sigmoid():
    assert 0 == sigmoid(-np.inf)
    assert 0.5 == sigmoid(0)
    assert 1 == sigmoid(np.inf)

    assert 0.5 > sigmoid(-0.1)
    assert 0.5 < sigmoid(0.1)


def test_scale():
    assert 1 == scale(0, 1, 5, 0, 1)
    assert 3 == scale(0.5, 1, 5, 0, 1)
    assert 5 == scale(1, 1, 5, 0, 1)

    assert all([a == b for a, b in zip(scale(np.asarray([0, 0.25, 0.5, 0.75, 1]), 1, 5),
                                       np.asarray([1, 2, 3, 4, 5]))])


def test_clip():
    assert 1 == clip(0, 1, 5)
    assert 3 == clip(3, 1, 5)
    assert 5 == clip(6, 1, 5)

    assert all([a == b for a, b in zip(clip(np.asarray([0, 3, 6]), 1, 5),
                                       np.asarray([1, 3, 5]))])


def test_intersects():
    assert 0 == len(intersects(np.asarray([1]), np.asarray(2)))
    assert 1 == len(intersects(np.asarray([2]), np.asarray(2)))

    assert all([a == b for a, b in zip(intersects(np.asarray([3, 2, 1]), np.asarray([1, 2])),
                                       np.asarray([2, 1]))])


def test_excepts():
    assert 1 == len(excepts(np.asarray([1]), np.asarray(2)))
    assert 0 == len(excepts(np.asarray([2]), np.asarray(2)))

    assert all([a == b for a, b in zip(excepts(np.asarray([3, 2, 1]), np.asarray([1, 2])),
                                       np.asarray([3]))])


def test_safe_indexing():
    assert all([a == b for a, b in zip(safe_indexing(np.asarray([3, 2, 1]), np.asarray([1, 2])),
                                       np.asarray([2, 1]))])
    assert all([a == b for a, b in zip(safe_indexing(np.asarray([3, 2, 1]), [1, 2]),
                                       np.asarray([2, 1]))])
    assert all([a == b for a, b in zip(safe_indexing([3, 2, 1], [1, 2]),
                                       [2, 1])])


def test_validate_format():
    assert 'UIR' == validate_format('UIR', ['UIR'])
    assert 'UIRT' == validate_format('UIRT', ['UIRT'])

    try:
        validate_format('iur', ['UIR'])
    except ValueError:
        assert True
