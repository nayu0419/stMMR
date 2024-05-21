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
plt.rc('font',family='Times New Roman')
section_id = "V1_Mouse_Brain_Sagittal_Anterior_Section_1"
k=52

im_re = pd.read_csv(os.path.join('Data',
              section_id, "image_representation/ViT_pca_representation.csv"), header=0, index_col=0, sep=',')
print(section_id, k)

adata = sc.read_visium("Data/{}".format(section_id),
                       count_file="{}_filtered_feature_bc_matrix.h5".format(section_id))
adata.var_names_make_unique()
prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
# prefilter_specialgenes(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
adata.obsm["im_re"] = im_re

Ann_df = pd.read_csv("Data/{}/metadata.tsv".format(section_id), sep="	", header=0, na_filter=False,
                     index_col=0)
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']
adata.obs['ground_truth'] = adata.obs['Ground Truth']
adata =  adata[:, adata.var['highly_variable']]

adata.obsm["adj"] = calculate_adj_matrix(adata)

adata= train_model.train(adata,k,n_epochs=50,h=[3000,3000],radius=0,l=1,embed=False)
obs_df = adata.obs.dropna()

ARI = adjusted_rand_score(obs_df['stMMR'], obs_df['Ground Truth'])
print('Adjusted rand index = %.5f' % ARI)

text_kwargs = {'fontfamily': 'Times New Roman'}
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, color=["stMMR", "Ground Truth"], title=['stMMR (ARI=%.2f)' % ARI, "Ground Truth"],
           save=section_id)
