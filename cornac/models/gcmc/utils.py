import torch.nn as nn
import torch.optim as optim

def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == "leaky":
            return nn.LeakyReLU(0.1)
        elif act == "relu":
            return nn.ReLU()
        elif act == "tanh":
            return nn.Tanh()
        elif act == "sigmoid":
            return nn.Sigmoid()
        elif act == "softsign":
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act
    
def get_optimizer(opt):
    if opt == "sgd":
        return optim.SGD
    elif opt == "adam":
        return optim.Adam
    else:
        raise NotImplementedError