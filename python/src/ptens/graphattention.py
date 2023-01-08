import torch
import torch.nn as nn
from torch.nn import Parameter, Sequential
import ptens
from ptens import ptensors0, ptensors1, ptensors2, graph
from ptens.functions import linear, relu, linmaps0, outer, unite1, gather
from ptens.modules import Linear, Dropout
import enum

class LayerType(enum.Enum):
    IMP1 = 0,
    IMP2 = 1,
    IMP3 = 2
    
class GraphAttention(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 num_heads_per_layer,
                 num_features_per_layer,
                 add_skip_connection=True,
                 bias = True,
                 dropout = 0.6,
                 layer_type = LayerType.IMP3,
                 log_attention_weights = False):
        
        super().__init__()
        assert num_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'
        GraphAttentionLayer = get_layer_type(layer_type)
        num_heads_per_layer= [1] + num_leads_per_layer

        # collect GAT layers
        for i in range(num_layers):
            layer = GraphAttentionLayer(num_in_features = num_features_per_layer[i]*num_heads_per_layer[i],
                                        num_out_features = num_features_per_layer[i+1],
                                        num_heads = num_heads_per_layer[i+1],
                                        concat = True if i < num_layers - 1 else False, # last GAT layer does mean avg, the others do concat
                                        activation = relu if i < num_layers - 1 else None, 
                                        dropout_prob = dropout,
                                        add_skip_connection = add_skip_connection,
                                        bias = bias,
                                        log_attention_weights = log_attention_weights)
            gat_layers.append(layer)
        self.gat_net = Sequential(*gat_layers,)

        def forward(self, data):
            return self.gat_net(data)

class GraphAttentionLayer(torch.nn.Module):
    head_dim = 1
    def __init__(self,
                 num_in_features,
                 num_out_features,
                 num_heads,
                 layer_type,
                 concat = True,
                 activation = relu, 
                 dropout_prob = 0.6,
                 add_skip_connection = True,
                 bias = True,
                 log_attention_weights = False):
        super().__init__()
        # saving this in forward prop in children layers (im1/2/3)
        self.num_heads = num_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        if layer_type == LayerType.IMP1:
            self.proj_param = Parameter(torch.Tensor(num_heads,
                                                     num_in_features,
                                                     num_out_features))
        else:
            self.linear_proj = Linear(num_in_features,
                                      num_heads*num_out_features,
                                      bias = False)
        self.scoring_fn_target = Parameter(torch.Tensor(1,
                                                        num_heads,
                                                        num_out_features))
        self.scoring_fn_source = Parameter(torch.Tensor(1,
                                                        num_heads,
                                                        num_out_features))
        if layer_type = LayerType.IMP1: # simple reshape in the case of implementation(IMP)1
            self.scoring_fn_target = Parameter(self.scoring_fn_target.reshape(num_heads,
                                                                              num_out_features,
                                                                              1))
            self.scoring_fn_source = Parameter(self.scoring_fn_source.reshape(num_heads,
                                                                              num_out_features,
                                                                              1))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(num_heads*num_features))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = Linear(num_in_features, num_heads*num_out_features, bias = False)
        else:
            self.register_parameter('skip_proj', None)

        self.activation = activation
        self.dropout = Dropout(p = dropout_prob)     
        self.attention_weights = None
        self.init_params(layer_type)

    def init_params(self, layer_type):
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:
            self.attention_weights = attention_coefficients
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1,self.num_heads, self.num_out_features)

        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_heads*self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias
        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)
