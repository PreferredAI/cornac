# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import ast
import itertools
from collections import Counter


def ui_parser(tokens, line_idx, id_inline=False, **kwargs):
    if id_inline:
        return [(str(line_idx + 1), iid, 1.0) for iid in tokens]
    else:
        return [(tokens[0], iid, 1.0) for iid in tokens[1:]]


def uir_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], float(tokens[2]))]


def review_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], tokens[2])]


def uirt_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], float(tokens[2]), int(tokens[3]))]


def tup_parser(tokens, **kwargs):
    return [
        (
            tokens[0],
            tokens[1],
            [tuple(tup.split(kwargs.get("tup_sep"))) for tup in tokens[2:]],
        )
    ]


def ubi_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], tokens[2])]


def ubit_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], tokens[2], int(tokens[3]))]


def ubitjson_parser(tokens, **kwargs):
    return [
        (tokens[0], tokens[1], tokens[2], int(tokens[3]), ast.literal_eval(tokens[4]))
    ]


def sit_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], int(tokens[2]))]


def sitjson_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], int(tokens[2]), ast.literal_eval(tokens[3]))]


def usit_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], tokens[2], int(tokens[3]))]


def usitjson_parser(tokens, **kwargs):
    return [
        (tokens[0], tokens[1], tokens[2], int(tokens[3]), ast.literal_eval(tokens[4]))
    ]


PARSERS = {
    "UI": ui_parser,
    "UIR": uir_parser,
    "UIRT": uirt_parser,
    "UITup": tup_parser,
    "UIReview": review_parser,
    "UBI": ubi_parser,
    "UBIT": ubit_parser,
    "UBITJson": ubitjson_parser,
    "SIT": sit_parser,
    "SITJson": sitjson_parser,
    "USIT": usit_parser,
    "USITJson": usitjson_parser,
}


class Reader:
    """Reader class for reading data with different types of format.

    Parameters
    ----------
    user_set: set, default = None
        Set of users to be retained when reading data.
        If `None`, all users will be included.

    item_set: set, default = None
        Set of items to be retained when reading data.
        If `None`, all items will be included.

    min_user_freq: int, default = 1
        The minimum frequency of a user to be retained.
        If `min_user_freq = 1`, all users will be included.

    min_item_freq: int, default = 1
        The minimum frequency of an item to be retained.
        If `min_item_freq = 1`, all items will be included.

    num_top_freq_user: int, default = 0
        The number of top popular users to be retained.
        If `num_top_freq_user = 0`, all users will be included.

    num_top_freq_item: int, default = 0
        The number of top popular items to be retained.
        If `num_top_freq_item = 0`, all items will be included.

    min_basket_size: int, default = 1
        The minimum number of items of a basket to be retained.
        If `min_basket_size = 1`, all items will be included.

    max_basket_size: int, default = -1
        The maximum number of items of a basket to be retained.
        If `min_basket_size = -1`, all items will be included.

    min_basket_sequence: int, default = 1
        The minimum number of baskets of a user to be retained.
        If `min_basket_sequence = 1`, all baskets will be included.

    min_sequence_size: int, default = 1
        The minimum number of items of a sequence to be retained.
        If `min_sequence_size = 1`, all sequences will be included.

    max_sequence_size: int, default = -1
        The maximum number of items of a sequence to be retained.
        If `min_sequence_size = -1`, all sequences will be included.

    bin_threshold: float, default = None
        The rating threshold to binarize rating values (turn explicit feedback to implicit feedback).
        For example, if `bin_threshold = 3.0`, all rating values >= 3.0 will be set to 1.0,
        and the rest (< 3.0) will be discarded.

    encoding: str, default = `utf-8`
        Encoding used to decode the file.

    errors: int, default = None
        Optional string that specifies how encoding errors are to be handled.
        Pass 'strict' to raise a ValueError exception if there is an encoding error
        (None has the same effect), or pass 'ignore' to ignore errors.
    """

    def __init__(
        self,
        user_set=None,
        item_set=None,
        min_user_freq=1,
        min_item_freq=1,
        num_top_freq_user=0,
        num_top_freq_item=0,
        min_basket_size=1,
        max_basket_size=-1,
        min_basket_sequence=1,
        min_sequence_size=1,
        max_sequence_size=-1,
        bin_threshold=None,
        encoding="utf-8",
        errors=None,
    ):
        self.user_set = (
            user_set
            if (user_set is None or isinstance(user_set, set))
            else set(user_set)
        )
        self.item_set = (
            item_set
            if (item_set is None or isinstance(item_set, set))
            else set(item_set)
        )
        self.min_uf = min_user_freq
        self.min_if = min_item_freq
        self.num_top_freq_user = num_top_freq_user
        self.num_top_freq_item = num_top_freq_item
        self.min_basket_size = min_basket_size
        self.max_basket_size = max_basket_size
        self.min_basket_sequence = min_basket_sequence
        self.min_sequence_size = min_sequence_size
        self.max_sequence_size = max_sequence_size
        self.bin_threshold = bin_threshold
        self.encoding = encoding
        self.errors = errors

    def _filter(self, tuples, fmt="UIR"):
        i_pos = fmt.find("I")
        u_pos = fmt.find("U")
        r_pos = fmt.find("R")

        if self.bin_threshold is not None and r_pos >= 0:

            def binarize(t):
                t = list(t)
                t[r_pos] = 1.0
                return tuple(t)

            tuples = [binarize(t) for t in tuples if t[r_pos] >= self.bin_threshold]

        if self.num_top_freq_user > 0:
            user_freq = Counter(t[u_pos] for t in tuples)
            top_freq_users = set(
                k for (k, _) in user_freq.most_common(self.num_top_freq_user)
            )
            tuples = [t for t in tuples if t[u_pos] in top_freq_users]

        if self.num_top_freq_item > 0:
            item_freq = Counter(t[i_pos] for t in tuples)
            top_freq_items = set(
                k for (k, _) in item_freq.most_common(self.num_top_freq_item)
            )
            tuples = [t for t in tuples if t[i_pos] in top_freq_items]

        if self.user_set is not None:
            tuples = [t for t in tuples if t[u_pos] in self.user_set]

        if self.item_set is not None:
            tuples = [t for t in tuples if t[i_pos] in self.item_set]

        if self.min_uf > 1:
            user_freq = Counter(t[u_pos] for t in tuples)
            tuples = [t for t in tuples if user_freq[t[u_pos]] >= self.min_uf]

        if self.min_if > 1:
            item_freq = Counter(t[i_pos] for t in tuples)
            tuples = [t for t in tuples if item_freq[t[i_pos]] >= self.min_if]

        return tuples

    def _filter_basket(self, tuples, fmt="UBI"):
        u_pos = fmt.find("U")
        b_pos = fmt.find("B")

        if self.min_basket_size > 1:
            sizes = Counter(t[b_pos] for t in tuples)
            tuples = [t for t in tuples if sizes[t[b_pos]] >= self.min_basket_size]

        if self.max_basket_size > 1:
            sizes = Counter(t[b_pos] for t in tuples)
            tuples = [t for t in tuples if sizes[t[b_pos]] <= self.max_basket_size]

        if self.min_basket_sequence > 1:
            basket_sequence = Counter(
                u for (u, _) in set((t[u_pos], t[b_pos]) for t in tuples)
            )
            tuples = [
                t
                for t in tuples
                if basket_sequence[t[u_pos]] >= self.min_basket_sequence
            ]

        return tuples

    def _filter_sequence(self, tuples, fmt="SIT"):
        s_pos = fmt.find("S")

        if self.min_sequence_size > 1:
            sizes = Counter(t[s_pos] for t in tuples)
            tuples = [t for t in tuples if sizes[t[s_pos]] >= self.min_sequence_size]

        if self.max_sequence_size > 1:
            sizes = Counter(t[s_pos] for t in tuples)
            tuples = [t for t in tuples if sizes[t[s_pos]] <= self.max_sequence_size]

        return tuples

    def read(
        self,
        fpath,
        fmt="UIR",
        sep="\t",
        skip_lines=0,
        id_inline=False,
        parser=None,
        **kwargs
    ):
        """Read data and parse line by line based on provided `fmt` or `parser`.

        Parameters
        ----------
        fpath: str
            Path to the data file.

        fmt: str, default: 'UIR'
            Line format to be parsed ('UI', 'UIR', 'UIRT', 'UITup', 'UIReview', 'UBI', 'UBIT', or 'UBITJson')

        sep: str, default: '\t'
            The delimiter string.

        skip_lines: int, default: 0
            Number of first lines to skip

        id_inline: bool, default: False
            If `True`, user ids corresponding to the line numbers of the file,
            where all the ids in each line are item ids.

        parser: function, default: None
            Function takes a list of `str` tokenized by `sep` and
            returns a list of tuples which will be joined to the final results.
            If `None`, parser will be determined based on `fmt`.

        Returns
        -------
        tuples: list
            Data in the form of list of tuples. What inside each tuple
            depends on `parser` or `fmt`.

        """
        parser = PARSERS.get(fmt, None) if parser is None else parser
        if parser is None:
            raise ValueError(
                "Invalid line format: {}\n"
                "Supported formats: {}".format(fmt, PARSERS.keys())
            )

        with open(fpath, encoding=self.encoding, errors=self.errors) as f:
            tuples = [
                tup
                for idx, line in enumerate(itertools.islice(f, skip_lines, None))
                for tup in parser(
                    line.strip().split(sep), line_idx=idx, id_inline=id_inline, **kwargs
                )
            ]
            tuples = self._filter(tuples=tuples, fmt=fmt)
            if fmt in {"UBI", "UBIT", "UBITJson"}:
                tuples = self._filter_basket(tuples=tuples, fmt=fmt)
            elif fmt in {"SIT", "SITJson", "USIT", "USITJson"}:
                tuples = self._filter_sequence(tuples=tuples, fmt=fmt)
            return tuples


def read_text(fpath, sep=None, encoding="utf-8", errors=None):
    """Read text file and return two lists of text documents and corresponding ids.
    If `sep` is None, only return one list containing elements are lines of text
    in the original file.

    Parameters
    ----------
    fpath: str
        Path to the data file

    sep: str, default = None
        The delimiter string used to split `id` and `text`. Each line is assumed
        containing an `id` followed by corresponding `text` document.
        If `None`, each line will be a `str` in returned list.

    encoding: str, default = `utf-8`
        Encoding used to decode the file.

    errors: int, default = None
        Optional string that specifies how encoding errors are to be handled.
        Pass 'strict' to raise a ValueError exception if there is an encoding error
        (None has the same effect), or pass 'ignore' to ignore errors.

    Returns
    -------
    texts, ids (optional): list, list
        Return list of text strings with corresponding indices (if `sep` is not None).
    """
    with open(fpath, encoding=encoding, errors=errors) as f:
        if sep is None:
            return [line.strip() for line in f]
        else:
            texts, ids = [], []
            for line in f:
                tokens = line.strip().split(sep)
                ids.append(tokens[0])
                texts.append(sep.join(tokens[1:]))
            return texts, ids
