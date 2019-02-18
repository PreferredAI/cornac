# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import itertools

class Reader:

    def __init__(self):
        pass

    @staticmethod
    def read_uir_triplets(path_to_data_file, u_col=0, i_col=1, r_col=2, sep='\t', skip_lines=0):
        """Read data in the form of triplets (user, item, rating).

        Parameters
        ----------
        path_to_data_file : str
            Path to the data file.

        u_col : int
            Index of the user column (default: 0).

        i_col : int
            Index of the item column (default: 0).

        r_col : int
            Index of the rating column (default: 0).

        sep : str
            The delimiter string (default: \t).

        skip_lines : int
            Number of first lines to skip (default: 0).

        Returns
        -------
        uir_triplets : array (n_examples, 3)
            Data in the form of list of tuples of (user, item, rating).

        """

        max_index = max(u_col, i_col, r_col)
        uir_triplets = []

        with open(path_to_data_file, 'r') as f:
            for line in itertools.islice(f, skip_lines, None):
                tokens = [token.strip() for token in line.split(sep)]
                if len(tokens) <= max_index:
                    raise IndexError('Number of tokens ({}) < max index ({})'.format(len(tokens), max_index))
                uir_triplets.append((tokens[u_col], tokens[i_col], float(tokens[r_col])))

        return uir_triplets
