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