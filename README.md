# stMMR: Multi-Modal Feature Representation in Spatial Transcriptomics with Similarity Contrastive Learning


**Understanding the spatial distribution and function of cells from spatial transcriptomics (ST) is of great value for deciphering the development and differentiation of tissues. However, the inherent significant heterogeneity and varying spatial resolutions of spatial transcriptomics multimodal data present challenges in the joint analysis of these modalities. In this study, we propose a novel method, named stMMR, to achieve multi-modal feature representation based on similarity contrastive learning. stMMR integrates histological imaging information with gene expression data through adjacency relationships. It uses self-attention module for deep embedding of features within a modality and incorporates similarity contrastive learning for integrating features across modalities. stMMR demonstrates superior performances in multiple analyses using datasets generated by different platforms, including domain identification, developmental trajectory inference as well as enhancement of gene expression. Using stMMR, we systematically analyzed the developmental process of the chicken heart and conducted an in-depth examination of key genes in breast cancer and lung cancer. In conclusion, stMMR is capable of effectively integrating the multimodal information and exhibits superior adaptability as well as stability in handling different types of ST data. **


![](https://github.com/nayu0419/ETHLR/blob/main/ETHLR_Pipeline.png)

## requirements
- MATLAB R2020a

## run

```
ETHLR_main.m
```

## File:

- funs: subfunction used in the ETHLR process
- Datasets: the pan-cancer omics data (for example, PAAD_CHOL_ESCA)
- The original dataset can be downloaded in [TCGA] (https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga)
- Put the other datasets in the folder "Datasets"

where *dataset* contains three categories of information, namely gene expression (GE), copy number variation (CNV) and methylation (ME). We represent a pan-cancer omics dataset as a third-order tensor.
