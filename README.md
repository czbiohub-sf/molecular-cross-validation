# Molecular Cross Validation

Single-cell RNA sequencing enables researchers to study the gene expression of individual cells. In high-throughput methods, however, the portrait of each individual cell is noisy, representing only a few thousand of the hundreds of thousands of mRNA molecules originally present.

 **Molecular cross-validation** is a statistically principled and data-driven approach for estimating the accuracy of any single-cell denoising method without the need for ground-truth. To perform molecular cross-validation, one splits the molecules captured from each cell into two groups, which is equivalent to taking two independent draws from that cell's original mRNA content. The first group is used for model fitting and the second for validation.
 
In this repository, we show how to correctly calibrate three denoising methods---principal component analysis, network diffusion, and a deep autoencoder---and how to select the best one for a given dataset.

For details about the method and performance on real and simulated data, see the preprint _Molecular Cross-Validation for Single-Cell RNA-seq_, available [here](https://www.biorxiv.org/content/10.1101/786269v1).
