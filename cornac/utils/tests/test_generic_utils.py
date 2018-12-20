# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np


def test_which_():
    from ..generic_utils import which_

    a = np.asarray([5, 4, 1, 2, 3, 0, 0])

    assert all([a == b for a, b in zip(which_(a, '>', 1), np.asarray([0, 1, 3, 4]))])
    assert all([a == b for a, b in zip(which_(a, '<', 3), np.asarray([2, 3, 5, 6]))])
    assert all([a == b for a, b in zip(which_(a, '>=', 2), np.asarray([0, 1, 3, 4]))])
    assert all([a == b for a, b in zip(which_(a, '<=', 4), np.asarray([1, 2, 3, 4, 5, 6]))])
    assert all([a == b for a, b in zip(which_(a, '!=', 0), np.asarray([0, 1, 2, 3, 4]))])


def test_sigmoid():
    from ..generic_utils import sigmoid

    assert 0 == sigmoid(-np.inf)
    assert 0.5 == sigmoid(0)
    assert 1 == sigmoid(np.inf)

    assert 0.5 > sigmoid(-0.1)
    assert 0.5 < sigmoid(0.1)


def test_map_to():
    from ..generic_utils import map_to

    assert 1 == map_to(0, 1, 5, 0, 1)
    assert 3 == map_to(0.5, 1, 5, 0, 1)
    assert 5 == map_to(1, 1, 5, 0, 1)

    assert all([a == b for a, b in zip(map_to(np.asarray([0, 0.25, 0.5, 0.75, 1]), 1, 5),
                                       np.asarray([1, 2, 3, 4, 5]))])


def test_clipping():
    from ..generic_utils import clipping

    assert 1 == clipping(0, 1, 5)
    assert 3 == clipping(3, 1, 5)
    assert 5 == clipping(6, 1, 5)

    assert all([a == b for a, b in zip(clipping(np.asarray([0, 3, 6]), 1, 5),
                                       np.asarray([1, 3, 5]))])


def test_intersects():
    from ..generic_utils import intersects

    assert 0 == len(intersects(np.asarray([1]), np.asarray(2)))
    assert 1 == len(intersects(np.asarray([2]), np.asarray(2)))

    assert all([a == b for a, b in zip(intersects(np.asarray([3, 2, 1]), np.asarray([1, 2])),
                                       np.asarray([2, 1]))])

def test_excepts():
    from ..generic_utils import excepts

    assert 1 == len(excepts(np.asarray([1]), np.asarray(2)))
    assert 0 == len(excepts(np.asarray([2]), np.asarray(2)))

    assert all([a == b for a, b in zip(excepts(np.asarray([3, 2, 1]), np.asarray([1, 2])),
                                       np.asarray([3]))])


def test_safe_indexing():
    from ..generic_utils import safe_indexing

    assert all([a == b for a, b in zip(safe_indexing(np.asarray([3, 2, 1]), np.asarray([1, 2])),
                                       np.asarray([2, 1]))])
    assert all([a == b for a, b in zip(safe_indexing(np.asarray([3, 2, 1]), [1, 2]),
                                       np.asarray([2, 1]))])
    assert all([a == b for a, b in zip(safe_indexing([3, 2, 1], [1, 2]),
                                       [2, 1])])


def test_validate_data_format():
    from ..generic_utils import validate_data_format

    assert 'UIR' == validate_data_format('UIR')
    assert 'UIR' == validate_data_format('uir')
    assert 'UIR' == validate_data_format('uIr')

    assert 'UIRT' == validate_data_format('UIRT')
    assert 'UIRT' == validate_data_format('uirt')
    assert 'UIRT' == validate_data_format('uIrT')

    try:
        validate_data_format('iur')
    except ValueError:
        assert True