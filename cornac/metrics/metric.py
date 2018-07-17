# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""
import numpy as np
from ..utils.util_functions import which_


#todo: take into account 'm' parameter
class Ndcg:
    """Normalized Discount Cumulative Gain.

    Parameters
    ----------
    m: int, optional, default: None
        The number of items in the top@m list, \
        if None then all items are considered to compute NDCG.

    name: string, value: 'NDCG'
        Name of the measure.

    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking".
    """

    def __init__(self, m= None):
        self.name = 'NDCG'
        self.type = 'ranking'
        self.m = m
        
    #Compute nDCG for a single user i
    def compute(self,data_test,reclist):
        #Compute Ideal DCG for user i
        irankTest_i = np.array(range(1,len(which_(data_test,'>',0))+1))
        irankTest_i = irankTest_i + 1
        irankTest_i = np.log2(irankTest_i)
        idcg_i = sum(np.divide(1,irankTest_i))
      
        #Compute DCG for user i
        rankTest_i = np.where(np.in1d(reclist,which_(data_test,'>',0)))[0]
        rankTest_i = rankTest_i +1 +1  # the second +1 because indices starst from 0 in python 
        rankTest_i = np.log2(rankTest_i)
        dcg_i      = sum(np.divide(1,rankTest_i))
    
        #Compute nDCG for user i
        ndcg_i = dcg_i/idcg_i

        return ndcg_i



#todo: take into account 'm' parameter
class Ncrr:
    """Normalized Cumulative Reciprocal Rank.

    Parameters
    ----------
    m: int, optional, default: None
        The number of items in the top@m list, \
        if None then all items are considered to compute NDCG.

    name: string, value: 'NCRR'
        Name of the measure.

    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking".
    """

    def __init__(self, m= None):
        self.name = 'NCRR'
        self.m = m
        self.type = 'ranking'

    #Compute nCRR for a single user i
    def compute(self,data_test,reclist):
        #Compute Ideal DCG for user i
        irankTest_i = np.array(range(1,len(which_(data_test,'>',0))+1))
        irankTest_i = irankTest_i
        icrr_i = sum(np.divide(1,irankTest_i))
      
        #### Compute DCG for user i
        rankTest_i = np.where(np.in1d(reclist,which_(data_test,'>',0)))[0]
        rankTest_i = rankTest_i +1  # the +1 because indices starst from 0 in python 
        crr_i      = sum(np.divide(1,rankTest_i))
    
        #Compute nDCG for user i
        ncrr_i = crr_i/icrr_i

        return ncrr_i



class Mrr:
    """Mean Reciprocal Rank.

    Parameters
    ----------
    name: string, value: 'MRR'
        Name of the measure.

    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking".
    """
    
    def __init__(self):
        self.name = 'MRR'
        self.type = 'ranking'

    #Compute MRR for a single user i
    def compute(self,data_test,reclist):

        rankTest_i = np.where(np.in1d(reclist,which_(data_test,'>',0)))[0]
        #if rankTest_i:
        mrr_i = np.divide(1,(rankTest_i[0]+1)) # +1 beacause indeces start from 0 in python
        #else:
        #    mrr_i = 0
        #    print('Error! only users with at least one heldout item should be evaluated')

        return mrr_i


class MeasureAtM:
    
    def __init__(self, m = 20, name=None):
        self.name = name
        self.m = m
        self.type = 'ranking' 
        self.tp = None
        self.tp_fn = None
        self.tp_fp = None
    
    
    #Evaluate TopMlist for a single user: Precision@M, Recall@M, F-meansure@M (F2)
    def measures_at_m(self,data_test,reclist):
  
        data_test_bin = np.full(len(data_test), 0)
        data_test_bin[which_(data_test,'>',0)] = 1
    
        pred = np.full(len(data_test), 0)
        pred[reclist[range(0,self.m)]] = 1
        
        self.tp = np.sum(pred * data_test_bin)
        self.tp_fn = np.sum(data_test_bin)
        self.tp_fp = np.sum(pred)


class Precision(MeasureAtM):
    """Precision@M.

    Parameters
    ----------
    m: int, optional, default: 20
        The number of items in the top@m list.
        
    name: string, value: 'Precision@m'
        Name of the measure.

    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking".
    """

    def __init__(self, m = 20):
        MeasureAtM.__init__(self,m = m, name="Precision@"+str(m))
    
    
    #Compute Precision@M for a single user i
    def compute(self,data_test,reclist):
        
        self.measures_at_m(data_test,reclist)
        prec = self.tp/self.tp_fp
        return prec
    
    
class Recall(MeasureAtM):
    """Recall@M.

    Parameters
    ----------
    m: int, optional, default: 20
        The number of items in the top@m list.
        
    name: string, value: 'Recall@m'
        Name of the measure.

    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking".
    """
    
    def __init__(self, m = 20):
        MeasureAtM.__init__(self,m = m, name="Recall@"+str(m))
    
    
    #Compute Precision@M for a single user i
    def compute(self,data_test,reclist):
        
        self.measures_at_m(data_test,reclist)
        rec = self.tp/self.tp_fn
        return rec
    
    
class Fmeasure(MeasureAtM):
    """F-meansure@M.

    Parameters
    ----------
    m: int, optional, default: 20
        The number of items in the top@m list.
        
    name: string, value: 'F2@m'
        Name of the measure.

    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking".
    """ 
    
    def __init__(self, m = 20):
        MeasureAtM.__init__(self,m = m, name="F2@"+str(m))
    
    
    #Compute Precision@M for a single user i
    def compute(self,data_test,reclist):
        
        self.measures_at_m(data_test,reclist)
        prec = self.tp/self.tp_fp
        rec = self.tp/self.tp_fn
        if (prec+rec):
            f2 = 2*(prec*rec)/(prec+rec)
        else:
            f2 = 0
        return f2


