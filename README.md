# Histopathological feature analyses

Analyses used to investigate associations between the DoMore-v1-CE-CRC marker and
histopathological features for the study *"Integration of AI-Based Pathological Risk
Assessment and ctDNA Detection for Enhanced Risk Stratification in Resected Colon
Cancer".* See this manuscript, and especially the *Methods* section for
additional details.

## Create image tile features

Create features for image tiles with the UNI foundation model
([https://huggingface.co/MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)) with

```
tile-features_uni/src/main.py
```

Note that this requires access to the UNI foundation model.

## Tile feature clustering

First, collect tile features into a single `.csv` file with size *(m + 1) x (n + 1)*,
where *m* is the number of included tile features and *n* is the size of each feature
vector (1024 from UNI). Expected header `path, <feature-1>, ..., <feature-1024>`.
Provided data in the expected format, this can be computed with
```
utils/collect_tile_features.py
```

Input this `.csv` file to 
```
clustering/src/main.py](clustering/src/main.py
```
to perform the clustering.

Compute cluster centres from clustered features with
```
utils/compute_cluster_centres.py
```
and classify tile
features according to the above cluster centres with
```
utils/classify_tile_features.py
```

## Clustering result analyses

Collect DoMore-v1-CE-CRC tile scores and merge them with corresponding tile feature
clusters with
```
clustering/analyses/associate_clusters_with_histotyping.py
```
and use the resulting file (where required) for the various analyses scripts in
```
clustering/analyses/
```

## Shapley analysis

For each whole slide image, we define a feature as the percentage of tiles for each
cluster and associate this with the DoMore-v1-CE-CRC score for the respective
whole slide image. To perform Shapley analysis, use scripts in
```
shap/
```

## Agglomerative clustering

To perform agglomerative clustering, use
```
utils/compute_and_visualize_superclusters.py
```
