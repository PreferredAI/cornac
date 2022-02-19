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
from collections import OrderedDict


NUM_FMT = "{:.4f}"


def _table_format(data, headers=None, index=None, extra_spaces=0, h_bars=None):
    if headers is not None:
        data.insert(0, headers)
    if index is not None:
        index.insert(0, "")
        for idx, row in zip(index, data):
            row.insert(0, idx)

    column_widths = np.asarray([[len(str(v)) for v in row] for row in data]).max(axis=0)

    row_fmt = (
        " | ".join(["{:>%d}" % (w + extra_spaces) for w in column_widths][1:]) + "\n"
    )
    if index is not None:
        row_fmt = "{:<%d} | " % (column_widths[0] + extra_spaces) + row_fmt

    output = ""
    for i, row in enumerate(data):
        if h_bars is not None and i in h_bars:
            output += row_fmt.format(
                *["-" * (w + extra_spaces) for w in column_widths]
            ).replace("|", "+")
        output += row_fmt.format(*row)
    return output


class Result:
    """
    Result Class for a single model

    Parameters
    ----------
    model_name: string, required
        The name of the recommender model.

    metric_avg_results: :obj:`OrderedDict`, required
        A dictionary containing the average result per-metric.

    metric_user_results: :obj:`OrderedDict`, required
        A dictionary containing the average result per-user across different metrics.
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
    Cross Validation Result Class for a single model. A list of :obj:`cornac.experiment.Result`.

    Parameters
    ----------
    model_name: string, required
        The name of the recommender model.
        
    Attributes
    ----------    
    metric_mean: :obj:`OrderedDict`
        A dictionary containing the mean of results across all folds per-metric.

    metric_std: :obj:`OrderedDict`
        A dictionary containing the standard deviation of results across all folds per-metric.
    """

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.metric_mean = OrderedDict()
        self.metric_std = OrderedDict()

    def __str__(self):
        return "[{}]\n{}".format(self.model_name, self.table)

    def organize(self):
        headers = list(self[0].metric_avg_results.keys())
        data, index = [], []
        for f, r in enumerate(self):
            data.append([r.metric_avg_results[m] for m in headers])
            index.append("Fold %d" % f)

        data = np.asarray(data)
        mean, std = data.mean(axis=0), data.std(axis=0)
        
        for m, mean_val, std_val in zip(headers, mean, std):
            self.metric_mean[m] = mean_val
            self.metric_std[m] = std_val
        
        data = np.vstack([data, mean, std])
        data = [[NUM_FMT.format(v) for v in row] for row in data]
        index.extend(["Mean", "Std"])
        self.table = _table_format(data, headers, index, h_bars=[1, len(data) - 1])

class PSTResult(list):
    """
    Propensity Stratified Result Class for a single model

    Parameters
    ----------
    model_name: string, required
        The name of the recommender model.
    """

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def __str__(self):
        return "[{}]\n{}".format(self.model_name, self.table)

    def organize(self):

        headers = list(self[0].metric_avg_results.keys())

        data, index, sizes = [], [], []
        for f, r in enumerate(self):
            data.append([r.metric_avg_results[m] for m in headers])
            if f == 0:
                index.append("Closed")
            elif f == 1:
                index.append("IPS")
            else:
                index.append("Q%d" % (f - 1))
            sizes.append(r.metric_avg_results["SIZE"])

        # add mean and std rows (total accumulative)
        data = np.asarray(data)
        mean, std = data.mean(axis=0), data.std(axis=0)

        # add unbiased stratified evaluation
        weights = np.asarray(sizes) / sizes[0]
        unbiased = np.average(data[2:], axis=0, weights=weights[2:]) * sum(weights[2:])

        # weighted average does not meaningful for size
        for idx, header in enumerate(headers):
            if header == "SIZE":
                unbiased[idx] = sizes[0]

        # update the table
        data = np.vstack([data, unbiased])
        data = [[NUM_FMT.format(v) for v in row] for row in data]
        index.extend(["Unbiased"])

        # add unbiased to the list
        self.append(
            Result(
                model_name=self[0].model_name,
                metric_avg_results=OrderedDict(zip(headers, unbiased)),
                metric_user_results=None,
            )
        )

        self.table = _table_format(data, headers, index, h_bars=[1, 2, 3, len(data)])


class ExperimentResult(list):
    """
    Result Class for an Experiment. A list of :obj:`cornac.experiment.Result`. 
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
    Result Class for a cross-validation Experiment.
    """

    def __str__(self):
        return "\n".join([r.__str__() for r in self])
