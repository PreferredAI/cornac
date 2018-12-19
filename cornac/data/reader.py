# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import itertools


def txt_to_uir_triplets(path_to_data_file, u_col=0, i_col=1, r_col=2, sep='\t', skip_lines=0):
    max_index = max(u_col, i_col, r_col)
    uir_triplets = []

    with open(path_to_data_file, 'r') as f:
        for line in itertools.islice(f, skip_lines, None):
            tokens = [token.strip() for token in line.split(sep)]
            if len(tokens) <= max_index:
                raise IndexError('Number of tokens ({}) < max index ({})'.format(len(tokens), max_index))
            uir_triplets.append((tokens[u_col], tokens[i_col], float(tokens[r_col])))

    return uir_triplets
