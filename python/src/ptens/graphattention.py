import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter, Sequential, LeakyReLU, ReLU
from torch.nn.functional import softmax, dropout
import ptens 
from ptens.functions import linear, relu, cat, elu
from ptens.modules import Dropout
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
        Wh = torch.mm(h_torch, self.W) # h: tensor (N, in_channels); Wh: tensor (N, out_channels)
        e = self._prepare_attentional_mechanism_input(Wh) # tensor

        zero_vec = -9e15*torch.ones_like(e) 
        attention = torch.where(adj_torch > 0, e, zero_vec) 
        attention = softmax(attention, dim=1)
        attention = dropout(attention, self.d_prob, training = self.training)
        h_prime = torch.matmul(attention, Wh) # tensor
        
        h_prime_p0 = ptens.ptensors0.from_matrix(h_prime) # ptensors0
        if self.concat:
            return relu(h_prime_p0)
        else:
            return h_prime_p0
        

    def _prepare_attentional_mechanism_input(self, Wh): # Wh: tensor
        Wh1 = torch.matmul(Wh, self.a[:self.out_channels, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_channels:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e) # tensor

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
        Wh = torch.mm(h, self.W) # h.shape: (N, in_channels), Wh.shape: (N, out_channels)
        e_p1 = self._prepare_attentional_mechanism_input(Wh)
        # e_p1 -> e_p0 -> size of e_p0 -> size of e_p1                      ?? 
        e_p0 = ptens.linmaps0(e_p1) 
        e_p0_r, e_p0_c = e_p0.torch().size()
        e_p1_r = e_p0_r + e_p1.get_nc()                                   # rm
        e_p1_c = e_p0_c
        zero_vec = -9e15*torch.ones_like(e_p0_r, e_p0_c)
        # ptensors1 -> ptensors0 -> torch -> do -> ptensors0 -> ptensors1   ?? 
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
            return h_prime # ptensors1

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_channels, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_channels:, :])
        e = Wh1 + Wh2.T
        e_p1 = ptens.linmaps1(ptens.ptensors0.from_matrix(e))
        return self.relu(e_p1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'

