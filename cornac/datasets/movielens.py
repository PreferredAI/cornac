# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..utils.download import DownloadItem
from ..utils.common import validate_data_format
from ..data.reader import Reader


class MovieLens100K:

    @staticmethod
    def load_data(format='UIR', verbose=False):
        """Load the MovieLens 100K dataset

        Parameters
        ----------
        format: str, default: 'UIR'
            Data format to be returned.

        verbose: bool, default: False
            The verbosity flag.

        Returns
        -------
        data: array-like
            Data in the form of a list of tuples depending on the given data format.

        """
        download_item = DownloadItem(url='http://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                                     relative_path='ml-100k/u.data')
        fpath = download_item.maybe_download(verbose)
        if format == 'UIR':
            return Reader.read_uir_triplets(fpath)


class MovieLens1M:

    @staticmethod
    def load_data(format='UIR', verbose=False):
        """Load the MovieLens 1M dataset

        Parameters
        ----------
        format: str, default: 'UIR'
            Data format to be returned.

        verbose: bool, default: False
            The verbosity flag.

        Returns
        -------
        data: array-like
            Data in the form of a list of tuples depending on the given data format.

        """
        download_item = DownloadItem(url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                                     relative_path='ml-1m/ratings.dat', unzip=True)
        fpath = download_item.maybe_download(verbose)
        if format == 'UIR':
            return Reader.read_uir_triplets(fpath, sep='::')