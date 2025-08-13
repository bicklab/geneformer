# !pwd

## install metacells
# !pip install metacells
# !pip install igraph
# !pip install psutil

## import libraries for visualization
import metacells as mc
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import seaborn as sb
import scanpy

from math import hypot
from matplotlib.collections import LineCollection

def run_metacells(filename, output, target_metacell_size=50000):
    
    
    group_data = scanpy.read_10x_mtx(filename)
    excluded_gene_names = ['IGHMBP2', 'IGLL1', 'IGLL5', 'IGLON5', 'NEAT1', 'TMSB10', 'TMSB4X']
    excluded_gene_patterns = ['MT-.*']
    
    # exclude low quality genes
    mc.pl.analyze_clean_genes(group_data,
                          excluded_gene_names=excluded_gene_names,
                          excluded_gene_patterns=excluded_gene_patterns,
                          random_seed=123456)
    
    mc.pl.pick_clean_genes(group_data)
    
    properly_sampled_min_cell_total = 800
    properly_sampled_max_cell_total = 8000
    properly_sampled_max_excluded_genes_fraction = 0.1
    
    mc.pl.analyze_clean_cells(
        group_data,
        properly_sampled_min_cell_total=properly_sampled_min_cell_total,
        properly_sampled_max_cell_total=properly_sampled_max_cell_total,
        properly_sampled_max_excluded_genes_fraction=properly_sampled_max_excluded_genes_fraction)
    
    mc.pl.pick_clean_cells(group_data)
    clean = mc.pl.extract_clean_data(group_data)
    
    suspect_gene_names = ['PCNA', 'MKI67', 'TOP2A', 'HIST1H1D', 'FOS', 'JUN', 'HSP90AB1', 'HSPA1A', 'ISG15', 'WARS' ]
    suspect_gene_patterns = [ 'MCM[0-9]', 'SMC[0-9]', 'IFI.*' ]
    suspect_genes_mask = mc.tl.find_named_genes(clean, names=suspect_gene_names,
                                            patterns=suspect_gene_patterns)
    suspect_gene_names = sorted(clean.var_names[suspect_genes_mask])
    
    mc.pl.relate_genes(clean, random_seed=123456)
    
    module_of_genes = clean.var['related_genes_module']
    suspect_gene_modules = np.unique(module_of_genes[suspect_genes_mask])
    suspect_gene_modules = suspect_gene_modules[suspect_gene_modules >= 0]
    
    similarity_of_genes = mc.ut.get_vv_frame(clean, 'related_genes_similarity')
    suspect_modules = []
    
    # (eg. if median similarity score for matrix is greater than 0.75, exclude module)
    for gene_module in suspect_gene_modules:
        module_genes_mask = module_of_genes == gene_module
        similarity_of_module = similarity_of_genes.loc[module_genes_mask, module_genes_mask]
        similarity_of_module.index = \
        similarity_of_module.columns = [
            '(*) ' + name if name in suspect_gene_names else name
            for name in similarity_of_module.index
        ]
        module_mean_value = similarity_of_module.mean().mean()
        if module_mean_value > 0.75:
            suspect_modules.append(gene_module)
            
    forbidden_genes_mask = suspect_genes_mask
    for gene_module in suspect_modules:
        module_genes_mask = module_of_genes == gene_module
        forbidden_genes_mask |= module_genes_mask
    forbidden_gene_names = sorted(clean.var_names[forbidden_genes_mask])
    
    max_parallel_piles = mc.pl.guess_max_parallel_piles(clean)
    mc.pl.set_max_parallel_piles(max_parallel_piles)
    mc.pl.set_max_parallel_piles(1)
    
    
    mc.pl.divide_and_conquer_pipeline(clean,
                                      forbidden_gene_names=forbidden_gene_names,
                                      random_seed=123456,
                                      target_metacell_size=target_metacell_size)

    clean.obs["metacell"].to_csv(output)

    run_metacells("/grand/GeomicVar/pershy1/data", "/grand/GeomicVar/pershy1/output/de/", 10000)





