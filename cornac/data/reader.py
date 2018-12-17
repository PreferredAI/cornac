# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import itertools


def txt_to_triplets(path_to_data_file, u_col=0, i_col=1, r_col=2, sep='\t', skip_lines=0):
    max_index = max(u_col, i_col, r_col)
    triplet_data = []

    with open(path_to_data_file, 'r') as f:
        for line in itertools.islice(f, skip_lines, None):
            tokens = [token.strip() for token in line.split(sep)]
            if len(tokens) <= max_index:
                raise IndexError('Number of tokens ({}) < max index ({})'.format(len(tokens), max_index))
            triplet_data.append((tokens[u_col], tokens[i_col], float(tokens[r_col])))

    return triplet_data
