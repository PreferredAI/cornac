# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.data import reader

def test_read_uir():
    """Test read_uir function"""
    data_file = './tests/data.txt'
    triplet_data = reader.read_uir(data_file)

    assert len(triplet_data) == 10
    assert triplet_data[4][2] == 3
    assert triplet_data[6][1] == '478'
    assert triplet_data[8][0] == '543'

    try:
        reader.read_uir(data_file, 10)
    except IndexError:
        assert True