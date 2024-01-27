import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.cluster import KMeans
import torch.optim as optim
from random import shuffle
import pandas as pd
import numpy as np
import scanpy as sc
from stMMR.layers import GraphConvolution,SelfAttention,MLP,MGCN,decoder
import sys

class stMMR(nn.Module):
    def __init__(self,nfeatX,nfeatI,hidden_dims):
        super(stMMR, self).__init__()
        self.mgcn = MGCN(nfeatX,nfeatI,hidden_dims)
        self.attlayer1 = SelfAttention(dropout=0.1)
        self.attlayer2 = SelfAttention(dropout=0.1)
        self.fc = nn.Linear(hidden_dims[1]*2, hidden_dims[1])
        self.mlp = MLP(hidden_dims[1], dropout_rate=0.1)
        self.ZINB = decoder(hidden_dims[1],nfeatX)

    def forward(self,x,i,a):
        emb_x,emb_i = self.mgcn(x,i,a)
        ## attention for omics specific information of scRNA-seq
        att_weights_x, att_emb_x = self.attlayer1(emb_x, emb_x, emb_x)

        ## attention for omics specific information of scATAC
        att_weights_i, att_emb_i = self.attlayer2(emb_i, emb_i, emb_i)

        q_x, q_i = self.mlp(emb_x, emb_i)

        # cl_loss = crossview_contrastive_Loss(q_x, q_i)

        # capture the consistency information
        emb_con = torch.cat([q_x, q_i], dim=1)
        z_xi = self.fc(emb_con)


        z_I = 20*att_emb_x + 1*att_emb_i + 10*z_xi

        [pi, disp, mean]  = self.ZINB(z_I)

        return z_I, q_x, q_i, pi, disp, mean


