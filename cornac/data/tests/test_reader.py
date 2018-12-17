# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""


from ..reader import txt_to_triplets


def test_txt_to_triplets():
    """Test txt_to_triplets function"""

    data_file = './cornac/data/tests/triplet_data.txt'
    u_col = 0
    i_col = 1
    r_col = 2
    sep = '\t'

    triplet_data = txt_to_triplets(data_file, u_col, i_col, r_col, sep, skip_lines=0)

    assert len(triplet_data) == 10
    assert triplet_data[4][2] == 3
    assert triplet_data[6][1] == '478'
    assert triplet_data[8][0] == '543'
