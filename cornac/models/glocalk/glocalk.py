from time import time
from scipy.sparse import csc_matrix
import numpy as np
from tqdm.auto import trange
import torch
import torch.nn as nn
import torch.nn.functional as F

    # train_set,
    # max_epoch_p, # max number of epochs for pretraining
    # max_epoch_f, # max number of epochs for finetuning
    # patience_p, # number of consecutive rounds of early stopping condition before actual stop for pretraining
    # patience_f, # and finetuning
    # tol_p, # minimum threshold for the difference between consecutive values of train rmse, used for early stopping, for pretraining
    # tol_f ,# and finetuning
    # lambda_2, # regularisation of number or parameters
    # lambda_s, # regularisation of sparsity of the final matrix
    # dot_scale, # dot product weight for global kernel

    # n_u,
    # n_hid,
    # n_dim,
    # n_layers,
    # n_m,

class GLocalK(nn.Module):
    def __init__(
    self,
    verbose,
    device,
    # Common hyperparameter settings
    n_hid , # size of hidden layers
    n_dim, # inner AE embedding size
    n_layers , # number of hidden layers
    gk_size , # width=height of kernel for convolution

    # Hyperparameters to tune for specific case
    max_epoch_p , # max number of epochs for pretraining
    max_epoch_f ,  # max number of epochs for finetuning
    patience_p , # number of consecutive rounds of early stopping condition before actual stop for pretraining
    patience_f , # and finetuning
    tol_p , # minimum threshold for the difference between consecutive values of train rmse, used for early stopping, for pretraining
    tol_f , # and finetuning
    lambda_2 , # regularisation of number or parameters
    lambda_s , # regularisation of sparsity of the final matrix
    dot_scale, # dot product weight for global kernel
    pre_min,
    pre_max,
    ):
      self.device = device
      self.n_hid = n_hid
      self.n_dim = n_dim
      self.n_layers = n_layers
      self.gk_size = gk_size
      self.max_epoch_p = max_epoch_p
      self.max_epoch_f = max_epoch_f
      self.patience_p = patience_p
      self.patience_f = patience_f
      self.tol_p = tol_p
      self.tol_f = tol_f
      self.lambda_2 = lambda_2
      self.lambda_s = lambda_s
      self.dot_scale = dot_scale
      self.verbose = verbose
      self.pre_min = pre_min
      self.pre_max = pre_max


class KernelLayer(nn.Module):
    def __init__(self, n_in, n_hid, n_dim, lambda_s, lambda_2, activation=nn.Sigmoid()):
      super().__init__()
      self.W = nn.Parameter(torch.randn(n_in, n_hid))
      self.u = nn.Parameter(torch.randn(n_in, 1, n_dim))
      self.v = nn.Parameter(torch.randn(1, n_hid, n_dim))
      self.b = nn.Parameter(torch.randn(n_hid))

      self.lambda_s = lambda_s
      self.lambda_2 = lambda_2

      nn.init.xavier_uniform_(self.W, gain=torch.nn.init.calculate_gain("relu"))
      nn.init.xavier_uniform_(self.u, gain=torch.nn.init.calculate_gain("relu"))
      nn.init.xavier_uniform_(self.v, gain=torch.nn.init.calculate_gain("relu"))
      nn.init.zeros_(self.b)
      self.activation = activation

    def forward(self, x):
      w_hat = local_kernel(self.u, self.v)
    
      sparse_reg = torch.nn.functional.mse_loss(w_hat, torch.zeros_like(w_hat))
      sparse_reg_term = self.lambda_s * sparse_reg
      
      l2_reg = torch.nn.functional.mse_loss(self.W, torch.zeros_like(self.W))
      l2_reg_term = self.lambda_2 * l2_reg

      W_eff = self.W * w_hat  # Local kernelised weight matrix
      y = torch.matmul(x, W_eff) + self.b
      y = self.activation(y)

      return y, sparse_reg_term + l2_reg_term

class KernelNet(nn.Module):
    def __init__(self, n_u, n_hid, n_dim, n_layers, lambda_s, lambda_2):
      super().__init__()
      layers = []
      for i in range(n_layers):
        if i == 0:
          layers.append(KernelLayer(n_u, n_hid, n_dim, lambda_s, lambda_2))
        else:
          layers.append(KernelLayer(n_hid, n_hid, n_dim, lambda_s, lambda_2))
      layers.append(KernelLayer(n_hid, n_u, n_dim, lambda_s, lambda_2, activation=nn.Identity()))
      self.layers = nn.ModuleList(layers)
      self.dropout = nn.Dropout(0.5)

    def forward(self, x):
      total_reg = None
      for i, layer in enumerate(self.layers):
        x, reg = layer(x)
        if i < len(self.layers)-1:
          x = self.dropout(x)
        if total_reg is None:
          total_reg = reg
        else:
          total_reg += reg
      return x, total_reg
    
class CompleteNet(nn.Module):
    def __init__(self, kernel_net, n_u, n_m, n_hid, n_dim, n_layers, lambda_s, lambda_2, gk_size, dot_scale):
      super().__init__()
      self.gk_size = gk_size
      self.dot_scale = dot_scale
      self.local_kernel_net = kernel_net
      self.global_kernel_net = KernelNet(n_u, n_hid, n_dim, n_layers, lambda_s, lambda_2)
      self.conv_kernel = torch.nn.Parameter(torch.randn(n_m, gk_size**2) * 0.1)
      nn.init.xavier_uniform_(self.conv_kernel, gain=torch.nn.init.calculate_gain("relu"))
      

    def forward(self, train_r):
      x, _ = self.local_kernel_net(train_r)
      gk = self.global_kernel(x, self.gk_size, self.dot_scale)
      x = self.global_conv(train_r, gk)
      x, global_reg_loss = self.global_kernel_net(x)
      return x, global_reg_loss

    def global_kernel(self, input, gk_size, dot_scale):
      avg_pooling = torch.mean(input, dim=1)  # Item (axis=1) based average pooling
      avg_pooling = avg_pooling.view(1, -1)

      gk = torch.matmul(avg_pooling, self.conv_kernel) * dot_scale  # Scaled dot product
      gk = gk.view(1, 1, gk_size, gk_size)

      return gk

    def global_conv(self, input, W):
      input = input.unsqueeze(0).unsqueeze(0)
      conv2d = nn.LeakyReLU()(F.conv2d(input, W, stride=1, padding=1))
      return conv2d.squeeze(0).squeeze(0)

class Loss(nn.Module):
    def forward(self, pred_p, reg_loss, train_m, train_r):
      # L2 loss
      diff = train_m * (train_r - pred_p)
      sqE = torch.nn.functional.mse_loss(diff, torch.zeros_like(diff))
      loss_p = sqE + reg_loss
      return loss_p
    


######################################################
### PRE TRAINING
######################################################

def learn(self,train_set,val_set):

  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.autograd import Variable
  from torch.utils.data import Dataset, DataLoader
  from torch.nn.parameter import Parameter

  #model :Pretraining KernetNet
  glocalk = self.glk


  pre_min = glocalk.pre_min
  pre_max = glocalk.pre_max
  n_u = train_set.num_users
  n_m = train_set.num_items

  train_r = train_set.csc_matrix.toarray().T
  train_m = np.greater(train_r, 1e-12).astype('float32')


  if glocalk.verbose:
    print(f"trainset shape: {train_r.shape}")
    print(f"pre_min {pre_min} pre_max: {pre_max}")

  # test_r = val_set.csc_matrix.toarray().T

  model = KernelNet(n_u, glocalk.n_hid, glocalk.n_dim, glocalk.n_layers, glocalk.lambda_s, glocalk.lambda_2).double().to(glocalk.device)
  complete_model = CompleteNet(model, n_u, n_m, glocalk.n_hid, glocalk.n_dim, glocalk.n_layers, glocalk.lambda_s, glocalk.lambda_2, glocalk.gk_size, glocalk.dot_scale).double().to(glocalk.device)


  # Pre-Training
  optimizer = torch.optim.AdamW(complete_model.local_kernel_net.parameters(), lr=0.001)

  def closure():
    optimizer.zero_grad()
    x = torch.Tensor(train_r).double().to(glocalk.device)
    m = torch.Tensor(train_m).double().to(glocalk.device)
    complete_model.local_kernel_net.train()
    pred, reg = complete_model.local_kernel_net(x)
    loss = Loss().to(glocalk.device)(pred, reg, m, x)
    loss.backward()
    return loss

  last_rmse = np.inf


  def monitor_value():
    return -last_rmse

  self.monitor_value = monitor_value

  progress_bar = trange(1, glocalk.max_epoch_p + 1, disable=not glocalk.verbose)

  counter = 0

  for epoch in progress_bar:
    optimizer.step(closure)
    complete_model.local_kernel_net.eval()


    pre, _ = model(torch.Tensor(train_r).double().to(glocalk.device))
    
    pre = pre.float().cpu().detach().numpy()
    
    # error = (test_m * (np.clip(pre, 1., 5.) - test_r) ** 2).sum() / test_m.sum()  # test error
    # test_rmse = np.sqrt(error)

    error_train = (train_m * (np.clip(pre, pre_min, pre_max) - train_r) ** 2).sum() / train_m.sum()  # train error
    train_rmse = np.sqrt(error_train)
    # if glocalk.verbose:
    #   print(f"Pre-training epoch:{epoch-1} rmse: {train_rmse:.4f} count : {counter}")

    if (glocalk.patience_p is not None) and ((last_rmse-train_rmse) < glocalk.tol_p):
      counter += 1
    else:
      counter = 0
    last_rmse = train_rmse
    progress_bar.set_postfix(rmse=last_rmse)
    if counter >= glocalk.patience_p:
      print("Pre-training Early Stopping")
      break

  print(f"Pre-training Finished! rmse: {last_rmse:.4f}")
  #################################################################
  # Finetuning
  #################################################################
  counter = 0
  self.reset_info()
  # Fine-Tuning
  optimizer_2 = torch.optim.AdamW(complete_model.parameters(), lr=0.001)

  def closure_2():
    optimizer_2.zero_grad()
    x = torch.Tensor(train_r).double().to(glocalk.device)
    m = torch.Tensor(train_m).double().to(glocalk.device)
    complete_model.train()
    pred, reg = complete_model(x)
    loss = Loss().to(glocalk.device)(pred, reg, m, x)
    loss.backward()
    return loss

  last_rmse = np.inf

  progress_bar = trange(1, glocalk.max_epoch_f + 1, disable=not glocalk.verbose)

  for epoch in progress_bar:
    optimizer_2.step(closure_2)
    complete_model.eval()

    pre, _ = complete_model(torch.Tensor(train_r).double().to(glocalk.device))
    
    pre = pre.float().cpu().detach().numpy()


    error_train = (train_m * (np.clip(pre, pre_min, pre_max) - train_r) ** 2).sum() / train_m.sum()  # train error
   
   
   
    train_rmse = np.sqrt(error_train)
    if (glocalk.verbose) and ((epoch-1)%50 ==0):
      print(f"Fine-training epoch:{epoch-1} rmse: {train_rmse:.4f} count : {counter}")

    if (glocalk.patience_f is not None) and ((last_rmse-train_rmse) < glocalk.tol_f):
      counter += 1
    else:
      counter = 0
    last_rmse = train_rmse
    progress_bar.set_postfix(rmse=last_rmse)
    if counter >= glocalk.patience_f:
      print("Fine-training Early Stopping")
      break

  print(f"Fine-training Finished! rmse: {last_rmse:.4f}")
  return pre


###########################################################
# Network Functions
###########################################################

def local_kernel(u, v):
    dist = torch.norm(u - v, p=2, dim=2)
    hat = torch.clamp(1. - dist**2, min=0.)
    return hat
