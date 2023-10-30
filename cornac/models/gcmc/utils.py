import torch.nn as nn
import torch.optim as optim
import numpy as np


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
        if act == "leaky_relu":
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


def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])


def torch_net_info(net, save_path=None):
    info_str = (
        "Total Param Number: {}\n".format(torch_total_param_num(net))
        + "Params:\n"
    )
    for k, v in net.named_parameters():
        info_str += "\t{}: {}, {}\n".format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(info_str)
    return info_str
