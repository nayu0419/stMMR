import os
import torch
import random
import numpy as np
import scanpy as sc
from torch.backends import cudnn
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.spatial import distance_matrix
def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-",Gene3Pattern="mt-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp3 = np.asarray([not str(name).startswith(Gene3Pattern) for name in adata.var_names], dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2,id_tmp3)
    adata._inplace_subset_var(id_tmp)

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)


def refine_nearest_labels(adata, radius=50, key='label'):
    new_type = []
    df = adata.obsm['spatial']
    old_type = adata.obs[key].values
    df = pd.DataFrame(df,index=old_type)
    distances = distance_matrix(df, df)
    distances_df = pd.DataFrame(distances, index=old_type, columns=old_type)

    for index, row in distances_df.iterrows():
        # row[index] = np.inf
        nearest_indices = row.nsmallest(radius).index.tolist()
        # for i in range(1):
        #     nearest_indices.append(index)
        max_type = max(nearest_indices, key=nearest_indices.count)
        new_type.append(max_type)
        # most_common_element, most_common_count = find_most_common_elements(nearest_indices)
        # nearest_labels.append(df.loc[nearest_indices, 'label'].values)

    return [str(i) for i in list(new_type)]

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  #其思想是 按照(row_index, column_index, value)的方式存储每一个非0元素，所以存储的数据结构就应该是一个以三元组为元素的列表List[Tuple[int, int, int]]
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)) #from_numpy()用来将数组array转换为张量Tensor vstack（）：按行在下边拼接
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
