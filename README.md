# Human Cornea snRNAseq

This repository contains a preprocessing script designed to convert **Human Cornea single-nucleus RNA sequencing (snRNAseq)** data into `.parquet` files suitable for machine learning and downstream analyses.

Dataset Reference: 
[longevity-db/human-cornea-snRNAseq on Hugging Face](https://huggingface.co/datasets/longevity-db/human-cornea-snRNAseq)

---

## About the Dataset

The **Human Cornea Aging Atlas** provides single-nucleus transcriptomic profiles of human corneal tissues. This dataset includes samples across a range of donor ages, making it a valuable resource for understanding age-related changes in the cornea at cellular resolution.

The dataset contains:
- Raw 10x Genomics count matrices
- Per-cell metadata (age, sex, sample, barcode, etc.)
- Cell types and cluster assignments

---

## Script Workflow

The `processing.py` script performs the following steps:

1. **Loads and concatenates** raw 10x files from multiple samples.
2. **Merges external metadata** (e.g., GSE186433 metadata CSV).
3. **Applies Scanpy preprocessing**:
   - Filtering cells/genes
   - Normalization and log transformation
4. **Performs dimensionality reduction**:
   - PCA (Principal Component Analysis)
   - UMAP (Uniform Manifold Approximation and Projection)
5. **Identifies and saves**:
   - Highly variable genes (HVGs)
   - Gene-level summary statistics
6. **Exports all data** into structured `.parquet` files for further analysis.

All outputs are saved under the `macaque_retina_aging_10x_processed/` directory (rename as needed).

---

## Output Files

| File Name                                 | Description |
|------------------------------------------|-------------|
| `expression.parquet`                     | Log-normalized expression matrix |
| `cell_metadata.parquet`                  | Metadata per cell (barcode, age, donor, etc.) |
| `gene_metadata.parquet`                  | Gene-level annotations |
| `pca_embeddings.parquet`                 | PCA-transformed coordinates |
| `pca_explained_variance.parquet`         | PCA variance explained per component |
| `umap_embeddings.parquet`                | 2D UMAP coordinates for each cell |
| `highly_variable_gene_metadata.parquet`  | Subset of top variable genes |
| `gene_statistics.parquet`                | Mean expression, detection frequency |

---

## Requirements

Install required Python packages:

```bash
pip install scanpy pandas numpy scikit-learn umap-learn pyarrow
```

---

## Usage

1. Download the [dataset from Hugging Face](https://huggingface.co/datasets/longevity-db/human-cornea-snRNAseq).
2. Place all `.tsv.gz`, `.mtx.gz`, and metadata `.csv` in a local folder.
3. Update paths in `processing.py`:

```python
RAW_DATA_PARENT_DIR = "/path/to/raw/files"
EXTERNAL_METADATA_FILE_PATH = "/path/to/GSE186433_metadata_percell.csv"
```

4. Run the script:

```bash
python processing.py
```

---

## Citation
- Please ensure you cite the original source of the Human Cornea Atlas data from GSE186433.
- Original Publication: Català, P., Groen, N., Dehnen, J. A., Soares, E., et al. (2021). 
"Single cell transcriptomics reveals the heterogeneity of the human cornea to identify novel markers of the limbus and stroma." Scientific Reports, 11(1), 21727. PMID: 34741068 DOI: 10.1038/s41598-021-01188-7

## Final Output
- https://huggingface.co/datasets/longevity-db/human-cornea-snRNAseq


Contributed by VPA Team
Venkatachalam, Pooja, Albert
