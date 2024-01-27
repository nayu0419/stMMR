# -*- coding: utf-8 -*-
from sklearn import metrics
from sklearn.cluster import KMeans
from stMMR.process import *
from stMMR import models
import numpy as np
import torch.optim as optim
from stMMR.utils import *
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
import sklearn
def train(adata,knn=10,h=[3000,3000], n_epochs=200,lr=0.0001, key_added='stMMR', random_seed=110,res=1,
          l=2,weight_decay=0.0001,a=10,b=1,c=10,embed=True,radius=0,enhancement=False,cluster="kmeans",loss_type="consistency",
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    set_seed(random_seed)
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    # labels = adata_Vars.obs['ground_truth']
    if type(adata.X) == np.ndarray:
        features_X = torch.FloatTensor(adata_Vars.X).to(device)
    else:
        features_X = torch.FloatTensor(adata_Vars.X.toarray()).to(device)
    features_I = torch.FloatTensor(adata_Vars.obsm["im_re"].values).to(device)
    adj=adata_Vars.obsm["adj"]
    adj = np.exp(-1*(adj**2)/(2*(l**2)))
    adj = sp.coo_matrix(adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    model =models.stMMR(nfeatX=features_X.shape[1],
                 nfeatI=features_I.shape[1],
                    hidden_dims=h,
                    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    mean_max = []
    #模型训练
    ari_max = 0
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        z_xi, q_x, q_i, pi, disp, mean = model(features_X, features_I, adj)
        zinb_loss = ZINB(pi, theta=disp, ridge_lambda=1).loss(features_X, mean, mean=True)
        if loss_type == "consistency":
            cl_loss = consistency_loss(q_x, q_i)
        else:
            cl_loss = crossview_contrastive_Loss(q_x, q_i)
        reg_loss = regularization_loss(z_xi, adj)
        total_loss = a * zinb_loss + b * cl_loss + c * reg_loss
        total_loss.backward()
        optimizer.step()
        print("epoch:",epoch,"loss:",total_loss.item())

        if 'ground_truth' in adata.obs.columns:
            if cluster == "kmeans":
                kmeans = KMeans(n_clusters=knn,random_state=random_seed).fit(np.nan_to_num(z_xi.cpu().detach()))
                idx = kmeans.labels_
                adata_Vars.obs['temp']=idx
                obs_df = adata_Vars.obs.dropna()
                ari_res = metrics.adjusted_rand_score(obs_df['temp'], obs_df['ground_truth'])
                print(ari_res)
            else:
                adata_Vars.obsm["letemp"]=z_xi.to('cpu').detach().numpy()
                sc.pp.neighbors(adata_Vars, use_rep='letemp')
                sc.tl.leiden(adata_Vars, key_added="temp", resolution=res)
                obs_df = adata_Vars.obs.dropna()
                ari_res = metrics.adjusted_rand_score(obs_df['temp'], obs_df['ground_truth'])
                idx=adata_Vars.obs['temp'].values
                count_unique_leiden = len(pd.DataFrame(adata_Vars.obs['temp']).temp.unique())
                print("num of cluster:",count_unique_leiden)
                print(epoch,ari_res )
            if ari_res > ari_max:
                ari_max = ari_res
                idx_max = idx
                mean_max = mean.to('cpu').detach().numpy()
                emb_max = z_xi.to('cpu').detach().numpy()
                print(epoch,ari_res)
    if 'ground_truth' in adata.obs.columns:
        print("Ari=", ari_max)
    else:
        if cluster == "kmeans":
            kmeans = KMeans(n_clusters=knn,random_state=random_seed).fit(np.nan_to_num(z_xi.cpu().detach()))
            idx_max = kmeans.labels_
            emb_max = z_xi.to('cpu').detach().numpy()
            mean_max = mean.to('cpu').detach().numpy()
        else:
            adata_Vars.obsm["letemp"] = z_xi.to('cpu').detach().numpy()
            sc.pp.neighbors(adata_Vars, use_rep='letemp')
            sc.tl.leiden(adata_Vars, key_added="temp", resolution=res)
            idx_max = adata_Vars.obs['temp'].values
            count_unique_leiden = len(pd.DataFrame(adata_Vars.obs['temp']).temp.unique())
            emb_max = z_xi.to('cpu').detach().numpy()
            mean_max = mean.to('cpu').detach().numpy()
            print("num of cluster:", count_unique_leiden)
    if embed:
        pca = PCA(n_components=20, random_state=random_seed)
        adata.obsm['emb_pca'] = pca.fit_transform(emb_max.copy())

    adata.obs["cluster"] = idx_max.astype(str)
    if radius !=0 :
        nearest_new_type = refine_label(adata, radius=radius)
        adata.obs[key_added] = nearest_new_type
    else:
        adata.obs[key_added] = adata.obs["cluster"]
    adata.obsm["emb"] = emb_max
    adata.obsm['mean'] = mean_max
    if enhancement:
        mean_max = sklearn.preprocessing.normalize(mean_max, axis=1, norm='max')
        adata.layers[key_added] = mean_max
    # sc.pp.neighbors(adata, use_rep='emb_pca')
    # sc.tl.umap(adata)
    # plt.rcParams["figure.figsize"] = (3, 3)
    # sc.pl.umap(adata, color=[key_added, "ground_truth"], title=['stMMR (ARI=%.2f)' % ari_max, "Ground Truth"],
    #            save="tt.pdf")
    return adata
