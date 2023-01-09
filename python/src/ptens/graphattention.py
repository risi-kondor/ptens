import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter, Sequential
from torch.nn.functional import softmax
import ptens 
from ptens.functions import linear, relu, cat
from ptens.modules import Dropout
class GraphAttentionLayer_P1(nn.Module):
    """
    An implementation of GATConv layer in ptens. 
    """
#   linmaps0->1: copy/ptensor by len(domain)
#   linmaps1->0: sum/ptensor
    def __init__(self, in_channels: int, out_channels: int, d_prob: torch.float = 0.5, alpha = 0.5, concat=True):
        super(GraphAttentionLayer, self).__init__()
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
        h = ptens.linmaps0(h).torch()
        Wh = torch.mm(h, self.W) # h.shape: (N, in_channels), Wh.shape: (N, out_channels)
        e_p1 = self._prepare_attentional_mechanism_input(Wh)
        # e_p1 -> e_p0 -> size of e_p0 -> size of e_p1                      ?? linmaps1->0; linmaps0->1
        e_p0 = ptens.linmaps0(e_p1) 
        e_p0_r, e_p0_c = e_p0.torch().size()
        e_p1_r = e_p0_r + e_p1.get_nc()
        e_p1_c = e_p0_c
        #zero_vec = -9e15*torch.ones_like(e_p1_r, e_p1_c)
        zero_vec = -9e15*torch.ones_like(e_p0_r, e_p0_c)
        # ptensors1 -> ptensors0 -> torch -> do -> ptensors0 -> ptensors1   ?? linmaps1->0; linmaps0->1
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

