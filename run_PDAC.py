#-*- coding : utf-8 -*-
import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from stMMR.utils import *
from stMMR.process import *
from stMMR import train_model
from datetime import datetime

section_id = "PDAC"
k=4

im_re = pd.read_csv("Data/PDAC/image_representation/ViT_pca_representation.csv", header=0, index_col=0, sep=',')
print(section_id, k)
counts_file = os.path.join('Data/PDAC/GSM3036911_PDAC-A-ST1-filtered.txt')
coor_file = os.path.join('Data/PDAC/spatial_location.csv')
manual_file = os.path.join('Data/PDAC/layer_manual_PDAC.csv')
counts = pd.read_csv(counts_file,header=0, index_col=0, sep='	')
coor_df = pd.read_csv(coor_file,header=0, index_col=0, sep=',')
label_df = pd.read_csv(manual_file,header=0, index_col=0, sep=',')
print(section_id,counts.shape, coor_df.shape)
counts=counts.T
counts.index  = coor_df.index
adata = sc.AnnData(counts)
adata.var_names_make_unique()
prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
# prefilter_specialgenes(adata)
sc.pp.normalize_total(adata, target_sum=1e4)
adata.obs["array_row"] = coor_df["y"]*-1
adata.obs["array_col"] = coor_df["x"]
adata.obsm["spatial"] = coor_df.loc[adata.obs_names, ["x", "y"]].to_numpy()
adata.obsm["im_re"] = im_re


adata.obs['Ground Truth'] = label_df["Region"]
adata.obs['ground_truth'] = adata.obs['Ground Truth']

adata.obsm["adj"] = calculate_adj_matrix(adata)

adata= train_model.train(adata,k,n_epochs=50,h=[3000,3000],radius=50,l=0.8,embed=False)

# ax = sc.pl.scatter(adata, alpha=1, x="array_row", y="array_col", color="stMMR", legend_fontsize=18, show=False,
#                    size=100000 / adata.shape[0])
obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['stMMR'], obs_df['Ground Truth'])
print(ARI)
