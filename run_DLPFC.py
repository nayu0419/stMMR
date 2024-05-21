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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--section_id', default=151673, type=int)
parser.add_argument('--k', default=7, type=int)
args = parser.parse_args()

section_id = str(args.section_id)
k = args.k
im_re = pd.read_csv(os.path.join('Data/DLPFC/',section_id, "image_representation/ViT_pca_representation.csv"),
                    header=0, index_col=0,sep=',')
print(section_id, k)
input_dir = os.path.join('Data/DLPFC/', section_id)
adata = sc.read_visium(path=input_dir, count_file=section_id+'_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()
prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
# prefilter_specialgenes(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
adata = adata[:, adata.var['highly_variable']]
adata.obsm["im_re"] = im_re

Ann_df = pd.read_csv(os.path.join('Data/DLPFC/',
                                  section_id, "cluster_labels_" + section_id + '.csv'), sep=',', header=0, index_col=0)
adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']

Ann_df = Ann_df.replace(1, "Layer 1")
Ann_df = Ann_df.replace(2, "Layer 2")
Ann_df = Ann_df.replace(3, "Layer 3")
Ann_df = Ann_df.replace(4, "Layer 4")
Ann_df = Ann_df.replace(5, "Layer 5")
Ann_df = Ann_df.replace(6, "Layer 6")
Ann_df = Ann_df.replace(7, "WM")
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']

adata.obsm["adj"] = calculate_adj_matrix(adata)
adata= train_model.train(adata,k,n_epochs=200,h=[3000,3000],enhancement=False,radius=0)

obs_df = adata.obs.dropna()
obs_df.to_csv("result/{}_type_stMMR.csv".format(section_id))
ARI = adjusted_rand_score(obs_df['stMMR'], obs_df['Ground Truth'])
print('Adjusted rand index = %.2f' % ARI)

sc.pl.spatial(adata, color=["stMMR", "Ground Truth"], title=['stMMR (ARI=%.2f)' % ARI, "Ground Truth"],
           save=section_id)


