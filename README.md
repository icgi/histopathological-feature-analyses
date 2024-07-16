# Histopathological feature analyses

Analyses used to investigate associations between the DoMore-v1-CE-CRC marker and
histopathological features for the study *"Novel clinical decision support system
optimizing adjuvant chemotherapy for colon cancer by integrating deep learning-based
pathology marker and circulating tumor DNA molecular residual disease: CIRCULATE-Japan
GALAXY substudy".*

## Create image tile features

Create features for image tiles with the UNI foundation model
(`https://huggingface.co/MahmoodLab/UNI`) with 

```
tile-features_uni/src/main.py
```

Note that this requires access to the UNI foundation model.

## Tile feature clustering

First, collect tile features into a single `.csv` file with size *(m + 1) x (n + 1)*,
where *m* is the number of included tile features and *n* is the size of each feature
vector (1024 from UNI). Expected header `path, <feature-1>, ..., <feature-1024>`.

```
utils/collect_tile_features.py
```

Input this `.csv` file to

```
clustering/src/main.py
```

Compute cluster centres with

```
utils/compute_cluster_centres.py
```

and classify tile features according to the above cluster centres with

```
utils/classify_tile_features.py
```

## Clustering result analyses

Collect DoMore-v1-CE-CRC tile scores and merge them with corresponding tile feature
clusters with

```
analyses/associate_clusters_with_histotyping.py
```

and use the resulting file (where required) for the various analyses scripts in
`analyses`.
