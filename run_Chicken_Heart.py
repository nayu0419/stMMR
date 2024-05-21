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
parser.add_argument('--section_id', default=14, type=int)
parser.add_argument('--k', default=6, type=int)
args = parser.parse_args()

section_id = args.section_id
k = args.k

# section_id = 14
# k=6
# section_id = 4
# k=5
#
# section_id = 10
# k=7
# #
# section_id = 7
# k=7

im_re = pd.read_csv("Data/chicken_heart/D{}/ViT_pca_representation.csv".format(section_id),
                    header=0, index_col=0, sep=',')
print(section_id, k)
adata = sc.read_10x_h5("Data/chicken_heart/D{}"
    "/chicken_heart_spatial_RNAseq_D{}_filtered_feature_bc_matrix.h5".format(section_id,section_id))
adata.var_names_make_unique()
prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
# prefilter_specialgenes(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
adata.obsm["im_re"] = im_re
coor_df = pd.read_csv(os.path.join('Data/chicken_heart/D{}'
    '/chicken_heart_spatial_RNAseq_D{}_tissue_positions_list.csv'.format(section_id,section_id),
         ), sep=",",header=None,na_filter=False, index_col=0)
# adata.obs["y_pixel"] = coor_df[4]
# adata.obs["x_pixel"] = coor_df[5]
adata.obs["array_row"] = coor_df[4]*-1
adata.obs["array_col"] = coor_df[5]
adata.obsm["spatial"] = coor_df.loc[adata.obs_names, [4,5]].to_numpy()

Ann_df = pd.read_csv("Data/chicken_heart/D{}/D{}.csv".format(section_id,section_id), sep=",", header=0,
                      na_filter=False, index_col=0)
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, "region"]
adata.obs['ground_truth'] = adata.obs['Ground Truth']
adata =  adata[:, adata.var['highly_variable']]
# sc.pp.neighbors(adata, use_rep='X')
# sc.tl.umap(adata)
#
# sc.pl.umap(adata, color='Ground Truth')

adata.obsm["adj"] = calculate_adj_matrix(adata)
adata= train_model.train(adata,k,n_epochs=200,h=[3000,3000],lr=0.00005)

obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['stMMR'], obs_df['Ground Truth'])
print('Adjusted rand index = %.5f' % ARI)

ax = sc.pl.scatter(adata, x="array_col", y="array_row", color='Ground Truth', legend_fontsize=18, show=False,
                   size=100000 / adata.shape[0])
title = "Ground Truth"
ax.set_title(title, fontsize=23)
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
dpi = 600
plt.savefig("figures/scatter_Truth_{}.pdf".format(section_id))
plt.close()

ax = sc.pl.scatter(adata, x="array_col", y="array_row", color='stMMR', legend_fontsize=18, show=False,
                   size=100000 / adata.shape[0])
title = "{}: ARI={:.2}".format("stMMR", ARI)
ax.set_title(title, fontsize=23)
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
dpi = 600
plt.savefig("figures/scatter_stMMR_{}.pdf".format(section_id))
plt.close()

sc.pp.neighbors(adata, use_rep='emb_pca')
adata.uns['iroot'] = np.flatnonzero(adata.obs['stMMR'] =="3")[0]  #D14

sc.tl.diffmap(adata)
sc.tl.dpt(adata, n_branchings=1)
adata.obs['dpt_pseudotime'] =1-adata.obs['dpt_pseudotime']
ax = sc.pl.scatter(adata, x="array_col", y="array_row", color='dpt_pseudotime', legend_fontsize=18, show=False,
                   size=50000 / adata.shape[0])
title = "{}: ARI={:.2}".format("stMMR", ARI)
ax.set_title(title, fontsize=23)
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
dpi = 600
plt.savefig("figures/dpt_{}.pdf".format(section_id), dpi=dpi)
plt.close()
obs_df = adata.obs.dropna()
obs_df.to_csv("result/{}_type_stMMR.csv".format(section_id))

sc.pp.neighbors(adata, use_rep='emb_pca')
sc.tl.umap(adata)
#
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata, color=["stMMR",'Ground Truth',"dpt_pseudotime"], title=['stMMR (ARI=%.2f)' % ARI , "Ground Truth",'pSM'],
           save="umap{}".format(section_id))
