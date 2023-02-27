from typing import Callable, List, Optional, Union
import torch
import torch.nn as nn
from torch.nn import Parameter, Sequential, LeakyReLU, ReLU
from torch.nn.functional import softmax, dropout
import ptens 
from ptens import ptensors0, ptensors1, ptensors2, graph
from ptens.functions import linear, linmaps0, outer, unite1, gather, relu, cat
class Linear(torch.nn.Module):
  def __init__(self,in_channels: int, out_channels: int, bias: bool = True) -> None:
    super().__init__()
    #This follows Glorot initialization for weights.
    self.w = torch.nn.parameter.Parameter(torch.empty(in_channels,out_channels))
    self.b = torch.nn.parameter.Parameter(torch.empty(out_channels)) if bias else None
    self.reset_parameters()
  def reset_parameters(self):
    self.w = torch.nn.init.xavier_uniform_(self.w)
    if not self.b is None:
      self.b = torch.nn.init.zeros_(self.b)
  def forward(self,x: ptensors1) -> ptensors1:
    assert x.get_nc() == self.w.size(0)
    return x * self.w if self.b is None else linear(x,self.w,self.b) 

class ConvolutionalLayer_0P(torch.nn.Module):
  r"""
  An implementation of GCNConv in ptens.
  """
  def __init__(self, channels_in: int, channels_out: int, bias : bool = True) -> None:
    super().__init__()
    self.lin = Linear(channels_in,channels_out,bias)
  def reset_parameters(self):
    self.lin.reset_parameters()
  def forward(self, features: ptensors1, graph: graph, symm_norm: Optional[ptensors0] = None) -> ptensors1:
    r"""
    Give symm_norm if you want symmetric normalization.
    """
    if not symm_norm is None:
      features = outer(features,symm_norm)
    # VxVxk -> VxVxk
    F = gather(features,graph)
    # [VxVxk] [VxVxk*2] -> [VxVxk*3]
    # [VxV*k*2] -> [VxV*k']
    F = self.lin(F)
    if not symm_norm is None:
      features = outer(F,symm_norm)
    return F

# TODO: find a better place for the below function
def generate_generic_shape(name: str, size: int) -> graph:
  shape_types = ['star','cycle','path']
  assert name in shape_types
  assert size >= 0
  def generate_star():
    nonlocal size
    edge_index = torch.tensor([[0,i,i,0] for i in range(1,size + 1)],dtype=torch.float32)
    edge_index = edge_index.view(-1,2).transpose(0,1)
    return graph.from_edge_index(edge_index)
  def generate_cycle():
    nonlocal size
    edge_index = torch.tensor([[i,(i + 1) % size,(i + 1) % size,i] for i in range(size)],dtype=torch.float32)
    edge_index = edge_index.view(-1,2).transpose(0,1)
    return graph.from_edge_index(edge_index)
  def generate_path():
    nonlocal size
    edge_index = torch.tensor([[i,i + 1,i + 1,i] for i in range(size - 1)],dtype=torch.float32)
    edge_index = edge_index.view(-1,2).transpose(0,1)
    return graph.from_edge_index(edge_index)
  return [generate_star,generate_cycle,generate_path][shape_types[size]]()
def create_edge_ptensors1(edge_index: torch.Tensor, edge_attributes: torch.Tensor) -> ptensors1:
  atoms = edge_index.transpose(0,1).tolist()
  return ptensors1.from_matrix(edge_attributes,atoms)

class ConvolutionalLayer_1P(torch.nn.Module):
  def __init__(self, channels_in: int, channels_out: int, bias : bool = True, reduction_type : str = "sum") -> None:
    r"""
    reduction_types: "sum" and "mean"
    """
    super().__init__()
    assert reduction_type == "sum" or reduction_type == "mean"
    self.lin = Linear(2*channels_in,channels_out,bias)
    self.use_mean = reduction_type == "mean"
  def reset_parameters(self):
    self.lin.reset_parameters()
  def forward(self, features: ptensors1, graph: graph, symm_norm: Optional[ptensors0] = None) -> ptensors1:
    r"""
    Give symm_norm if you want symmetric normalization.
    """
    if not symm_norm is None:
      features = outer(features,symm_norm)
    # VxVxk -> VxVxk*2
    F = unite1(features,graph,self.use_mean)
    # [VxVxk] [VxVxk*2] -> [VxVxk*3]
    # [VxV*k*2] -> [VxV*k']
    F = self.lin(F)
    if not symm_norm is None:
      features = outer(F,symm_norm)
    return F
class ConvolutionalLayer_1P_V2(torch.nn.Module):
  def __init__(self, channels_in: int, channels_out: int, target_domains: Callable[[graph],List[List]], incoming_order : int = 1, bias : bool = True, reduction_type : str = "sum") -> None:
    r"""
    reduction_types: "sum" and "mean"
    """
    super().__init__()
    assert reduction_type == "sum" or reduction_type == "mean"
    self.lin = Linear([1,2,5][incoming_order]*channels_in,channels_out,bias)
    self.use_mean = reduction_type == "mean"
    self.target_domains = target_domains
  def reset_parameters(self):
    self.lin.reset_parameters()
  def forward(self, features: Union[ptensors0,ptensors1], graph: graph, symm_norm: Optional[ptensors0] = None) -> ptensors1:
    r"""
    Give symm_norm if you want symmetric normalization.
    """
    if symm_norm is not None:
      features = outer(features,symm_norm)
    if isinstance(features,ptensors0):
      F = ptensors0.transfer1(features,self.target_domains(graph))
    elif isinstance(features,ptensors1):
      F = ptensors1.transfer1(features,self.target_domains(graph))
    else:
      raise Exception("Unhandled 'features' type: " + str(type(features)))
    F = self.lin(F)
    if symm_norm is not None:
      features = outer(F,symm_norm)
    return F
class Reduce_1P_0P(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
    super().__init__()
    self.lin = Linear(in_channels,out_channels,bias)
  def forward(self, features: ptensors1) -> ptensors0:
    features = linmaps0(features)
    a = self.lin(features)
    return a
class Dropout(torch.nn.Module):
  def __init__(self, prob: torch.float = 0.5, device : str = 'cuda') -> None:
    super().__init__()
    self.p = prob
    self.device = device
  def cuda(self, device = None):
    self.device = device
    return super().cuda(device)
  def forward(self, x):
    dropout = (torch.rand(x.get_nc(),device=self.device) > self.p).float()
    if isinstance(x,ptensors0):
      return ptensors0.mult_channels(x,dropout)
    elif isinstance(x,ptensors1):
      return ptensors1.mult_channels(x,dropout)
    elif isinstance(x,ptensors2):
      return ptensors2.mult_channels(x,dropout)
    else:
      raise NotImplementedError('Dropout not implemented for type \"' + str(type(x)) + "\"")
class BatchNorm(torch.nn.Module):
  def __init__(self, num_features: int, eps: torch.float = 1E-5, momentum: torch.float = 0.1) -> None:
    super().__init__()
    # TODO: consider using 'UnitializedParameter' here
    self.running_mean = None
    self.running_var = None
    self.eps = eps
    self.momentum = momentum
  def forward(self, x):
    r"""
    x can be any type of ptensors
    """
    x_vals : torch.Tensor = x.torch()
    if self.running_mean is None:
      self.has_had_first_batch = True
      self.running_mean = x_vals.mean(0)
      self.running_var = x_vals.var(0)
    else:
      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_vals.mean()
      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_vals.var(unbiased=False)
    y = (self.running_var + self.eps)**-0.5
    b = -self.running_mean * y
    test = torch.einsum('ij,jk->ij',x_vals,torch.diag(y)) + b
    print(test.mean(),test.var())
    # TODO: make this better
    return linear(x,torch.diag(y),b)
class PNormalize(torch.nn.Module):
  def __init__(self, p: int = 2, eps: torch.float = 1E-5) -> None:
    super().__init__()
    self.eps = eps
    self.p = p
  def forward(self, x):
    r"""
    x can be any type of ptensors
    """
    x_vals : torch.Tensor = x.torch()
    pnorm = torch.linalg.norm(x_vals,self.p,0)
    pnorm = (self.eps + pnorm)**-1
    if isinstance(x,ptensors0):
      return ptensors0.mult_channels(x,pnorm)
    elif isinstance(x,ptensors1):
      return ptensors1.mult_channels(x,pnorm)
    elif isinstance(x,ptensors2):
      return ptensors2.mult_channels(x,pnorm)
    else:
      raise NotImplementedError('PNormalize not implemented for type \"' + str(type(x)) + "\"")

class GraphAttentionLayer_P0(nn.Module):
    """
    An implementation of GATConv layer in ptens. 
    """
    def __init__(self, in_channels: int, out_channels: int, d_prob = 0.5, leakyrelu_alpha = 0.5, relu_alpha = 0.5, concat=True):
        super(GraphAttentionLayer_P0, self).__init__()
        self.d_prob = d_prob
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.leakyrelu_alpha = leakyrelu_alpha
        self.relu_alpha = relu_alpha
        self.concat = concat

        self.W = Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.W.data)
        self.a = Parameter(torch.empty(out_channels, 1))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = LeakyReLU(self.leakyrelu_alpha)
        self.relu = relu(self.relu_alpha)

    # ptensors0 -> tensor -> do -> ptensors0
    def forward(self, h: ptensors0, adj: ptensors0):
        h_torch = h.torch()
        adj_torch = adj.torch()
        Wh = torch.mm(h_torch, self.W) 
        e = self._prepare_attentional_mechanism_input(Wh) 

        zero_vec = -9e15*torch.ones_like(e) 
        attention = torch.where(adj_torch > 0, e, zero_vec) 
        attention = softmax(attention, dim=1)
        attention = dropout(attention, self.d_prob, training = self.training)
        h_prime = torch.matmul(attention, Wh)
        
        h_prime_p0 = ptens.ptensors0.from_matrix(h_prime)
        if self.concat:
            return relu(h_prime_p0)
        else:
            return h_prime_p0
      
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_channels, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_channels:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'
        
class GraphAttentionLayer_P1(nn.Module):
    """
    An implementation of GATConv layer in ptens. 
    """
#   linmaps0->1: copy/ptensor by len(domain)
#   linmaps1->0: sum/ptensor
    def __init__(self, in_channels: int, out_channels: int, d_prob: torch.float = 0.5, alpha = 0.5, concat=True):
        super(GraphAttentionLayer_P1, self).__init__()
        self.d_prob = d_prob
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.concat = concat

        self.W = Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = Parameter(torch.empty(2*out_channels, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.relu = relu(self.alpha)

    def forward(self, h: ptensors1, adj: ptensors1):
        h = ptens.linmaps0(h).torch() # ptensors1 -> ptensors0 -> torch
        Wh = torch.mm(h, self.W) 
        e_p1 = self._prepare_attentional_mechanism_input(Wh)
        # e_p1 -> e_p0 -> size of e_p0 -> size of e_p1                      
        e_p0 = ptens.linmaps0(e_p1) 
        e_p0_r, e_p0_c = e_p0.torch().size()
        e_p1_r = e_p0_r + e_p1.get_nc()                                   
        e_p1_c = e_p0_c
        zero_vec = -9e15*torch.ones_like(e_p0_r, e_p0_c)
        # ptensors1 -> ptensors0 -> torch -> do -> ptensors0 -> ptensors1 
        adj_torch = ptens.linmaps0(adj).torch()
        e_torch = e_p0.torch()
        attention = torch.where(adj_torch > 0, e_torch, zero_vec)
        attention = softmax(attention, dim=1)
        attention_p1 = ptens.linmaps1(ptens.ptensors0.from_matrix(attention))
        attention_p1 = Dropout(attention_p1, self.d_prob)  
        h_prime = attention*Wh
        if self.concat:
            return relu(h_prime)
        else:
            return h_prime 

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_channels, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_channels:, :])
        e = Wh1 + Wh2.T
        e_p1 = ptens.linmaps1(ptens.ptensors0.from_matrix(e))
        return self.relu(e_p1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'
