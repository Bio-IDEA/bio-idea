feature: Features are extracted from the genomic bins containing genes, including multi-omics signals such as RNA-seq, CTCF, DNase, H3K27ac, H3K4me3, and TCI.

label: Indicates whether the current genomic bin contains a tumor risk gene.

1.HGB_Model.py: This script implements our Histogram Gradient Boosting Tree model, which integrates multi-omics features and TCI to build a genome-wide predictor for tumor risk genes.

2.feature_group.py: This script is used to evaluate the combination of TCI with various omics signals, aiming to investigate the contribution of TCI to model prediction performance.
