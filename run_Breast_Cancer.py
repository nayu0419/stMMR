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

section_id = "V1_Breast_Cancer_Block_A_Section_1"
k=20

im_re = pd.read_csv(os.path.join('Data',section_id,
        "image_representation/VIT_pca_representation.csv"), header=0, index_col=0, sep=',')
print(section_id, k)

adata = sc.read_visium("Data/V1_Breast_Cancer_Block_A_Section_1",
                count_file="V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
# prefilter_specialgenes(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
adata.obsm["im_re"] = im_re

Ann_df = pd.read_csv("Data/V1_Breast_Cancer_Block_A_Section_1/metadata.tsv", sep="	", header=0, na_filter=False,
                     index_col=0)
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'fine_annot_type']
adata.obs['ground_truth'] = adata.obs['Ground Truth']
adata =  adata[:, adata.var['highly_variable']]

adata.obsm["adj"] = calculate_adj_matrix(adata)

adata= train_model.train(adata,k,n_epochs=50,h=[3000,3000],radius=50,l=0.63,lr=0.0000005)
# 13 epoch=14
#
obs_df = adata.obs.dropna()
# obs_df.to_csv("result/{}_type_stMMR.csv".format(section_id))
ARI = adjusted_rand_score(obs_df['stMMR'], obs_df['Ground Truth'])
print('Adjusted rand index = %.5f' % ARI)



plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, color=["stMMR", "Ground Truth"], title=['stMMR (ARI=%.2f)' % ARI, "Ground Truth"],
           save=section_id)