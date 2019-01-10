# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..utils.download_utils import DownloadItem
from ..utils.generic_utils import validate_data_format
from ..data.reader import Reader


class MovieLens:

    def __init__(self):
        super().__init__()


class MovieLens100K(MovieLens):

    @staticmethod
    def load_data(format='UIR', verbose=False):
        download_item = DownloadItem(url='http://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                                     relative_path='datasets/ml_100k/u.data')
        fpath = download_item.maybe_download(verbose)

        format = validate_data_format(format)
        if format == 'UIR':
            return Reader.read_uir_triplets(fpath)


class MovieLens1M(MovieLens):

    @staticmethod
    def load_data(format='UIR', verbose=False):
        download_item = DownloadItem(url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                                     relative_path='datasets/ml_1m/ml-1m/ratings.dat', unzip=True)
        fpath = download_item.maybe_download(verbose)

        format = validate_data_format(format)
        if format == 'UIR':
            return Reader.read_uir_triplets(fpath, sep='::')