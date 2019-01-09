# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.data import Reader

def test_txt_to_triplets():
    """Test txt_to_triplets function"""

    data_file = './tests/data.txt'
    u_col = 0
    i_col = 1
    r_col = 2
    sep = '\t'

    triplet_data = Reader.read_uir_triplets(data_file, u_col, i_col, r_col, sep, skip_lines=0)

    assert len(triplet_data) == 10
    assert triplet_data[4][2] == 3
    assert triplet_data[6][1] == '478'
    assert triplet_data[8][0] == '543'

    try:
        Reader.read_uir_triplets(data_file, 10)
    except IndexError:
        assert True