# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..utils.download_utils import DownloadItem
from ..utils.generic_utils import validate_data_format
from ..data import reader


class MovieLens:

    def __init__(self):
        super().__init__()


class MovieLens100K(MovieLens):

    @staticmethod
    def load_data(format='UIR', verbose=False):
        download_item = DownloadItem(url='http://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                                     relative_path='u.data', sub_dir='datasets/ml_100k')
        fpath = download_item.download_if_needed(verbose)

        format = validate_data_format(format)
        if format == 'UIR':
            return reader.txt_to_uir_triplets(fpath)


class MovieLens1M(MovieLens):

    @staticmethod
    def load_data(format='UIR', verbose=False):
        download_item = DownloadItem(url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                                     relative_path='ml-1m/ratings.dat', unzip=True, sub_dir='datasets/ml_1m')
        fpath = download_item.download_if_needed(verbose)

        format = validate_data_format(format)
        if format == 'UIR':
            return reader.txt_to_uir_triplets(fpath, sep='::')