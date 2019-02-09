# noise2self-single-cell


## Datasets

### Synthetic

1. Simscity: latent space to counts, poisson sampling
2. Hierarchical: add structure to noise (in manner of correlated bursting, either
  in gene space, or in a second latent space, larger than the bottleneck
  which would be used.)

### Real

1. Developmental (Paul et al or TM Marrow)
2. Hepatocytes (TM)
3. Deep sequenced (subset of 10X 1 million PBMCs)

## Models

1. PCA (on sqrt normalized data)
2. MAGIC (on sqrt normalized, predicting sqrt normalized (MSE) or counts (NB))
3. Autoencoder (DCA style, predicting counts (NB/Poisson))
4. Resnet (predicting counts)

## Metrics

Evaluation metrics we can compute on data with or without ground-truth labels.

### Gene-Gene correlations

Correlation. (Requires ground truth or prior bio knowledge.)

### Cell-Cell similarity

Silhouette. (Requires clustering or labels.)

### Self-supervised loss

MSE or Poisson or NB.
