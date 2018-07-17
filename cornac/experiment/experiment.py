# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

import numpy as np
import pandas as pd


class Experiment:
    """ Experiment Class

    Parameters
    ----------
    eval_strategy: EvaluationStrategy object, required
        The evaluation strategy (e.g., Split).

    models: array of objects Recommender, required
        A collection of recommender models to evaluate, e.g., [C2pf, Hpf, Pmf].

    metrics: array of object metrics, required
        A collection of metrics to use to evaluate the recommender models, \
        e.g., [Ndcg, Mrr, Recall].

    res_avg: DataFrame, default: None
        The average result per model.

    res_per_user: dictionary, default: {}
        Results per user for each model.    
    """

    def __init__(self, eval_strategy, models, metrics):
        self.eval_strategy = eval_strategy
        self.models = models
        self.metrics= metrics
        
        self.res = None
        self.res_avg = None
        self.res_std = None
        self.res_per_user = {}


    #modify this function to accomodate several models
    def run_(self):
        
        model_names = []
        metric_names = []
        
        if not hasattr(self.metrics, "__len__"):
            self.metrics = np.array([self.metrics])  #test whether self.metrics is an array
        if not hasattr(self.models, "__len__"):
            self.models = np.array([self.models])    #test whether self.models is an array
        
        for mt in self.metrics:
            metric_names.append(mt.name)
        
        for model in self.models:
            print(model.name)
            model_names.append(model.name)
            res = self.eval_strategy.run_exp(model = model,metrics = self.metrics)
            self.res_per_user[model.name] = res['ResPerUser']
            if self.res_avg is None:
                self.res_avg = res['ResAvg']
            else:
                self.res_avg = np.vstack((self.res_avg,res['ResAvg']))
            
        #Formatting the results using the Pandas DataFrame
        if len(self.models) == 1:
            self.res_avg = self.res_avg.reshape(1,len(self.metrics))
        resAvg_dataFrame = pd.DataFrame(data= self.res_avg, index=model_names, columns=metric_names)
        self.res_avg = resAvg_dataFrame
        ##Metrics, take into account the metrics specified by the user
        del(resAvg_dataFrame)
 