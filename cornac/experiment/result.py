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

import numpy as np

NUM_FMT = '{:.4f}'


def _table_format(data, headers=None, index=None, extra_spaces=0, h_bars=None):
    if headers is not None:
        data.insert(0, headers)
    if index is not None:
        index.insert(0, '')
        for idx, row in zip(index, data):
            row.insert(0, idx)

    column_widths = np.asarray([[len(str(v)) for v in row] for row in data]).max(axis=0)

    row_fmt = ' | '.join(['{:>%d}' % (w + extra_spaces) for w in column_widths][1:]) + '\n'
    if index is not None:
        row_fmt = '{:<%d} | ' % (column_widths[0] + extra_spaces) + row_fmt

    output = ''
    for i, row in enumerate(data):
        if h_bars is not None and i in h_bars:
            output += row_fmt.format(*['-' * (w + extra_spaces) for w in column_widths]).replace('|', '+')
        output += row_fmt.format(*row)
    return output


class Result:
    """
    Result Class for a single model
    """

    def __init__(self, model_name, metric_avg_results, metric_user_results):
        self.model_name = model_name
        self.metric_avg_results = metric_avg_results
        self.metric_user_results = metric_user_results

    def __str__(self):
        headers = list(self.metric_avg_results.keys())
        data = [[NUM_FMT.format(v) for v in self.metric_avg_results.values()]]
        return _table_format(data, headers, index=[self.model_name], h_bars=[1])


class CVResult(list):
    """
    Cross Validation Result Class for a single model
    """

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def __str__(self):
        return '[{}]\n{}'.format(self.model_name, self.table)

    def organize(self):
        headers = list(self[0].metric_avg_results.keys())
        data, index = [], []
        for f, r in enumerate(self):
            data.append([r.metric_avg_results[m] for m in headers])
            index.append('Fold %d' % f)

        data = np.asarray(data)
        mean, std = data.mean(axis=0), data.std(axis=0)
        data = np.vstack([data, mean, std])
        data = [[NUM_FMT.format(v) for v in row] for row in data]
        index.extend(['Mean', 'Std'])
        self.table = _table_format(data, headers, index, h_bars=[1, len(data) - 1])


class ExperimentResult(list):
    """
    Result Class for an Experiment
    """

    def __str__(self):
        headers = list(self[0].metric_avg_results.keys())
        data, index = [], []
        for r in self:
            data.append([NUM_FMT.format(r.metric_avg_results[m]) for m in headers])
            index.append(r.model_name)
        return _table_format(data, headers, index, h_bars=[1])


class CVExperimentResult(ExperimentResult):
    """
    Result Class for a cross-validation Experiment
    """

    def __str__(self):
        return '\n'.join([r.__str__() for r in self])
