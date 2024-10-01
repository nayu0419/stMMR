import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class SelfAttention(nn.Module):
    """
    attention_1
    """
    def __init__(self, dropout):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):

        queries = q
        keys = k
        values = v
        n, d = queries.shape
        scores = torch.mm(queries, keys.t()) / math.sqrt(d)
        att_weights = F.softmax(scores, dim=1)
        att_emb = torch.mm(self.dropout(att_weights), values)
        return att_weights, att_emb

class MLP(nn.Module):

    def __init__(self, z_emb, dropout_rate):
        super(MLP, self).__init__()
        self.mlpx = nn.Sequential(
            nn.Linear(z_emb, z_emb),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.mlpi = nn.Sequential(
            nn.Linear(z_emb, z_emb),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
    def forward(self, z_x, z_y):
        q_x = self.mlpi(z_x)
        q_y = self.mlpi(z_y)
        return q_x, q_y


class MGCN(nn.Module):
    def __init__(self, nfeatX, nfeatI, hidden_dims):
        super(MGCN, self).__init__()
        self.GCNA1_1 = GraphConvolution(nfeatX, hidden_dims[0])
        # self.GCNA1_2 = GraphConvolution(hidden_dims[0], hidden_dims[0])
        self.GCNA1_3 = GraphConvolution(hidden_dims[0], hidden_dims[1])
        # self.GCNA1_3 = GraphConvolution(hidden_dims, hidden_dims)
        self.GCNA2_1 = GraphConvolution(nfeatI, hidden_dims[0])
        # self.GCNA2_2 = GraphConvolution(hidden_dims[0], hidden_dims[0])
        self.GCNA2_3 = GraphConvolution(hidden_dims[0], hidden_dims[1])
        # self.GCNA2_3 = GraphConvolution(hidden_dims, hidden_dims)

    # def forward(self, x, i, a):
    #     emb1 = F.relu(self.GCNA1_1(x, a))
    #     emb1 = F.dropout(emb1, 0)
    #     emb1 = self.GCNA1_3(emb1, a)
    #     emb2 = F.relu(self.GCNA2_1(i, a))
    #     emb2 = F.dropout(emb2, 0)
    #     emb2 = self.GCNA2_3(emb2, a)
    #     return emb1, emb2
    def forward(self, x, i, a):
        emb1 = self.GCNA1_1(x, a)
        # emb1 = self.GCNA1_2(emb1, a)
        emb1 = self.GCNA1_3(emb1, a)
        emb2 = self.GCNA2_1(i, a)
        # emb2 = self.GCNA2_2(emb2, a)
        emb2 = self.GCNA2_3(emb2, a)
        return emb1, emb2

# class MGCN(nn.Module):
#     def __init__(self, nfeatX, nfeatI, hidden_dims):
#         super(MGCN, self).__init__()
#         self.GCNA1_1 = nn.Linear(nfeatX, hidden_dims[0])
#         # self.GCNA1_2 = GraphConvolution(hidden_dims[0], hidden_dims[0])
#         self.GCNA1_3 = nn.Linear(hidden_dims[0], hidden_dims[1])
#         # self.GCNA1_3 = GraphConvolution(hidden_dims, hidden_dims)
#         self.GCNA2_1 = nn.Linear(nfeatI, hidden_dims[0])
#         # self.GCNA2_2 = GraphConvolution(hidden_dims[0], hidden_dims[0])
#         self.GCNA2_3 = nn.Linear(hidden_dims[0], hidden_dims[1])
#         # self.GCNA2_3 = GraphConvolution(hidden_dims, hidden_dims)
#
#     def forward(self, x, i, a):
#         emb1 = self.GCNA1_1(x)
#         # emb1 = self.GCNA1_2(emb1, a)
#         emb1 = self.GCNA1_3(emb1)
#         emb2 = self.GCNA2_1(i)
#         # emb2 = self.GCNA2_2(emb2, a)
#         emb2 = self.GCNA2_3(emb2)
#         return emb1, emb2

class decoder(torch.nn.Module):
    def __init__(self, nhid1, nfeat):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid1, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1,  nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)


    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]
