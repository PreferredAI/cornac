import operator
import numpy as np


#util functions
def which_(a, op, x):
    ops = {'>' : operator.gt,
           '<' : operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '==': operator.eq,
           '!=': operator.ne}
    
    i = np.array(range(0,len(a)))

    return i[ops[op](a,x)]



#Sigmoid function                
def sigmoid(x):
  return 1. / (1. + np.exp(-x))

#Map the value of a numpy array "x" from o_min, o_max into a range[t_min,t_max] 
def map_to(x,t_min,t_max, o_min = None, o_max = None):
    if o_min is None:
        o_min = np.min(x)
    if o_max is None:
        o_max = np.max(x)
    
    return ((x-o_min)/(o_max-o_min))*(t_max-t_min) + t_min

#Perfom clipping to enforce values to lie in a specific range [min_,max_]
def clipping (x,min_,max_):
    x[which_(x,'>',max_)] = max_
    x[which_(x,'<',min_)] = min_
    return x