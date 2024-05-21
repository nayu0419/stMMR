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


ARIlist=[]

fov=1
for fov in range(1,21):
    # fov = 10
    df= pd.read_csv("Data/Lung9_Rep1-Flat_files_and_images/Lung9_Rep1_exprMat_file.csv", sep=",", header=0, na_filter=False,
                        index_col=None)
    df=df[df.fov==fov]
    df.index = df["cell_ID"]
    df=df.drop(0,axis=0)
    df =df.add_suffix("_{}".format(fov),axis=0)
    df=df.drop(['fov','cell_ID'], axis=1)
    adata = ad.AnnData(df)
    adata.var_names_make_unique()
    prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.log1p(adata)
    Ann_df = pd.read_csv("Data/Lung9_Rep1-Flat_files_and_images/meta.csv", sep=",", header=0, na_filter=False,
                         index_col=0)
    mut2les=  {     'B-cell':"Lymphocytes", 'NK':"Lymphocytes",'T CD4 memory': 'Lymphocytes',
                    'T CD4 naive': 'Lymphocytes','T CD8 memory': 'Lymphocytes',
                    'T CD8 naive':'Lymphocytes','Treg':'Lymphocytes', 'endothelial': 'Endothelial',
                    'epithelial': 'Epithelial', 'fibroblast': 'Fibroblast',
                    'mDC': 'Myeloid', 'macrophage': 'Myeloid', 'mast': 'Mast',
                    'monocyte': 'Myeloid', 'neutrophil': 'Neutrophil',
                    'pDC':'Myeloid','plasmablast':'Myeloid','tumor 9':'Tumor','tumor 12':'Tumor'
                    ,'tumor 13':'Tumor','tumor 6':'Tumor','tumor 5':'Tumor'}
    l2n=  {     "Lymphocytes":0, 'Neutrophil':1,'Mast': 2,'Endothelial':3,
                    'Fibroblast':4,'Epithelial': 5,'Myeloid':6,'Tumor':7}
    n2l=  {     0:"Lymphocytes", 1:'Neutrophil',2:'Mast',3:'Endothelial',
                    4:'Fibroblast',5:'Epithelial',6:'Myeloid',7:'Tumor'}
    adata.obs['Ground Truth'] = Ann_df['cell_type'].map(mut2les)
    adata.obs['ground_truth'] = adata.obs['Ground Truth'].map(l2n)
    print(adata.obs['Ground Truth'].value_counts())

    posdf= pd.read_csv("Data\Lung9_Rep1-Flat_files_and_images\Lung9_Rep1_metadata_file.csv", sep=",", header=0, na_filter=False,
                        index_col=None)
    posdf=posdf[posdf.fov==fov]
    posdf.index = posdf["cell_ID"]
    posdf =posdf.add_suffix("_{}".format(fov),axis=0)
    adata.obs["array_row"]=posdf["CenterX_local_px"]
    adata.obs["array_col"]=posdf["CenterY_local_px"]

    im_re = pd.read_csv("Data/Lung9_Rep1-Flat_files_and_images/image_representation/fov{}.csv".format(fov),
                        header=0, index_col=0, sep=',')
    im_re = im_re.sort_index(axis=0)
    im_re.index = range(1, len(im_re) + 1)
    im_re = im_re.add_suffix("_{}".format(fov), axis=0)
    adata.obsm["im_re"] = im_re

    adata.obsm["adj"] = calculate_adj_matrix(adata)
    adata = train_model.train(adata, knn=8,radius=0,n_epochs=200,h=[3000,3000])

    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['stMMR'], obs_df['Ground Truth'])
    print(ARI)
    sc.pl.scatter(adata, alpha=1, x="array_row", y="array_col", color='stMMR',  legend_fontsize=18, show=False,
    title='%s (ARI=%.2f)' % ('stMMR',ARI),size=100000 / adata.shape[0],save="{}stMMR".format(fov))
    ARIlist.append(ARI)
    obs_df = adata.obs.dropna()
    obs_df['stMMR'] = munkres_newlabel(obs_df['ground_truth'], obs_df['stMMR'].astype('float32'))
    adata.obs['stMMR'] = obs_df['stMMR']
    adata.obs['stMMR'] = adata.obs['stMMR'].map(n2l).astype('category')
    sc.pl.scatter(adata, alpha=1, x="array_row", y="array_col", color='stMMR',  legend_fontsize=18, show=False,
    title='%s (ARI=%.2f)' % ('SpaMMR',ARI),size=100000 / adata.shape[0],save="{}stMMR".format(fov))
    obs_df.to_csv("result/nano_{}_type_stMMR.csv".format(fov))

print("ari mean", np.mean(ARIlist))
print("ari median", np.median(ARIlist))
print(ARIlist)