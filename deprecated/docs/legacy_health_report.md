# Health report for mantis_unified_atlas_multimodal_v3.csv

* Rows: 5610  
* Columns: 48  
* Vector column: vector

## Embedding integrity

* Unique vector dimensions: [512]

* Inconsistent dims: False

* Sample NaNs: 0, Infs: 0


## Naming integrity

* Unique titles: 48

* Duplicate titles: 5562


## Dataset mix

* Dataset counts: {'CRC_VAL_HE_7K': 1600, 'MEDMNIST_PATHMNIST': 1600, 'HF_LC25000': 800, 'HF_BACH': 800, 'HF_BREAKHIS_RCL_7500': 800, 'ANCHOR': 10}

* Suffixes appearing in >1 dataset_key: 1600


## Hierarchy columns

* Found: ['cluster_L1', 'cluster_L2', 'cluster_L3', 'cluster_L4', 'cluster_L5', 'cluster_L6', 'cluster_L7']


### Unknown/Unspecified counts

* cluster_L2: {'Unknown': 1890}

* cluster_L4: {'Unknown': 1889}

* cluster_L6: {'Unspecified': 1889}

* cluster_L7: {'Unspecified': 4111}


## kNN connectivity (from nn columns)

* k=12 edges=46626

* connected components=21

* top component sizes=[3377, 273, 217, 177, 164, 102, 101, 101, 90, 89]

* cross-dataset edges ratio=0.0442
