# Closing the loop: Teaching single-cell foundation models to learn from perturbations

## Abstract 
The application of transfer learning models to large scale single-cell datasets has enabled the development of single-cell foundation models (scFMs) that can predict cellular responses to perturbations in silico. Although these predictions can be experimentally tested, current scFMs are unable to “close the loop” and learn from these experiments to create better predictions. Here, we introduce a “closed-loop” framework that extends the scFM by incorporating perturbation data during model fine-tuning. Our closed-loop model improves prediction accuracy, increasing positive predictive value in the setting of T-cell activation three-fold. We applied this model to RUNX1-familial platelet disorder, a rare pediatric blood disorder and identified two therapeutic targets (mTOR and CD74-MIF signaling axis) and two novel pathways (protein kinase C and phosphoinositide 3-kinase). This work establishes that iterative incorporation of experimental data to foundation models enhances biological predictions, representing a crucial step toward realizing the promise of "virtual cell" models for biomedical discovery.

## T-cell activation experiments
We used publicly available single-cell RNA sequencing data from 4 studies where T cells were either unstimulated or stimulated via CD3-CD28 beads or phorbol myristate acetate/ionomycin (PMA/ionomycin): 
1.	[Kartha et al](https://www.sciencedirect.com/science/article/pii/S2666979X22001082) performed scRNAseq on resting and stimulated primary human peripheral blood mononuclear cells (PBMCs) from 4 donors. We included the scRNAseq data from 3,708 T cells which were unstimulated and 3,797 T cells which were stimulated with PMA/ionomycin along with Brefeldin A, a protein secretion inhibitor which allows us to isolate the primary effect of T cell activation. They were stimulated with PMA and Ionomycin calcium salt with or without Brefeldin A.
2.	[Cano-Gomez et al](https://www.nature.com/articles/s41467-020-15543-y) performed scRNAseq on primary T cells with and without anti-CD3/anti-CD28 beads in the absence of cytokines (Th0). CD4+ T cells were isolated from the PBMCs of 6 donors. T cells were then stimulated with anti-CD3/anti-CD28 human T-activator dynabeads. They profiled gene expression (RNA-seq) at 16 hours. A total of 5,269 resting T cells were profiled and 7,309 stimulated T cells were profiled.
3.	[Lawlor et al](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2021.636720/full) performed scRNAseq on PBMCs from 10 donors. PBMCs were cultured in control media (resting) versus plate-bound anti-CD3 + anti-CD28 (stimulated). 4,589 T cells were stimulated and 3,694 T cells were resting.
4.	[Szabo et al](https://www.nature.com/articles/s41467-019-12464-3) performed scRNAseq T cells, after CD3+ selection, from the blood of two deceased adult organ donors and two healthy adult volunteers with and without T-cell receptor stimulation using anti-CD3/CD28 T-activator for 16 hours. There were 9,147 resting T cells and 8,478 activated T cells.

For analysis with Geneformer, the single-cell RNAseq data was combined into a Seurat object and converted to a Loom object for tokenization. For DE analysis, Harmony was used to integrate the data from the four studies. Cells were grouped based on similarity of cell states using the Python package Metacell-2 with a targeted metacell size of 160,000 transcripts. Differential expression testing was performed using the Wald test for negative binomial regression through DESeq2 for genes that had at least 10 transcripts in at least 85% of metacells.

## T-cell in silico perturbation


We used the CRISPRa and CRISPRi results from {Schmidt, Science, 2022}(https://www.science.org/doi/10.1126/science.abj4008) to evaluate our predictions. 

## 
```
cd /Users/pershy1/geneformer
git add .
git commit -m "Updating geneformer scripts" (replace with more descriptive if desired)
git push origin main
```

Voila! Now, polaris should match local which should match GitHub.
