import pandas as pd
import anndata as ad
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import numpy as np
import scanpy as sc
import re # Needed for regular expressions to parse filenames

# --- Configuration ---
# IMPORTANT: This should be the path to the directory containing ALL the .tsv.gz and .mtx.gz files directly.
RAW_DATA_PARENT_DIR = "/path/to/your/file.h5ad" # <<<--- CONFIRM THIS PATH!

# IMPORTANT: Path to the external metadata file (e.g., GSE186433_metadata_percell.csv)
EXTERNAL_METADATA_FILE_PATH = "/Users/venkatachalamsubramanianperiyasubbu/Downloads/GSE186433_metadata_percell.csv" # <<<--- CONFIRM THIS PATH!

# Name of the directory to store the output Parquet files
OUTPUT_DIR_NAME = "macaque_retina_aging_10x_processed"

# PCA Configuration
N_PCA_COMPONENTS = 50
APPLY_SCALING_BEFORE_PCA = True

# UMAP Configuration
N_UMAP_COMPONENTS = 2
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# HVG Configuration
N_TOP_HVGS = 4000

# --- 1. Create Output Directory ---
os.makedirs(OUTPUT_DIR_NAME, exist_ok=True)
print(f"Created output directory: {OUTPUT_DIR_NAME}")

# --- 2. Load and Concatenate Individual 10x Genomics Samples ---
print("Loading individual 10x Genomics samples...")
adatas = []

# Find unique sample prefixes in the RAW_DATA_PARENT_DIR based on filenames
# Expected filename format: GSMXXXXXXX_gYYY_barcodes.tsv.gz
# We want to extract "GSMXXXXXXX_gYYY" as the sample prefix
all_files_in_dir = os.listdir(RAW_DATA_PARENT_DIR)
sample_prefixes = set()
for filename in all_files_in_dir:
    # Use regex to capture the part before '_barcodes.tsv.gz', '_features.tsv.gz', '_matrix.mtx.gz'
    match = re.match(r"^(GSM\d+_g\d+)_", filename)
    if match:
        sample_prefixes.add(match.group(1)) # Add the full sample prefix (e.g., 'GSM5651509_g001')

if not sample_prefixes:
    print(f"Error: No 10x Genomics files found with expected prefix (e.g., 'GSM..._g...') in '{RAW_DATA_PARENT_DIR}'.")
    print("Please ensure your files are directly in this directory and follow the 'GSMXXXXXXX_gYYY_' naming convention.")
    exit()

for sample_prefix in sorted(list(sample_prefixes)):
    try:
        # sc.read_10x_mtx can take a directory and a 'prefix' argument.
        # It will then look for:
        # {RAW_DATA_PARENT_DIR}/{sample_prefix}_barcodes.tsv.gz
        # {RAW_DATA_PARENT_DIR}/{sample_prefix}_features.tsv.gz
        # {RAW_DATA_PARENT_DIR}/{sample_prefix}_matrix.mtx.gz
        adata_sample = sc.read_10x_mtx(RAW_DATA_PARENT_DIR, prefix=f"{sample_prefix}_", var_names='gene_symbols', make_unique=True)
        adata_sample.obs['original_sample_prefix'] = sample_prefix # Store the original sample prefix

        # Make cell barcodes unique across samples by prepending the sample_prefix
        # This will result in unique cell_ids like "GSM5651509_g001_AAACCTGAGCGGATTG-1"
        adata_sample.obs_names = [f"{sample_prefix}_{bc}" for bc in adata_sample.obs_names]
        print(f"  Loaded {sample_prefix}: {adata_sample.shape[0]} cells, {adata_sample.shape[1]} genes.")
        adatas.append(adata_sample)
    except Exception as e:
        print(f"  Warning: Could not load data for {sample_prefix}. Skipping. Error: {e}")

if not adatas:
    print("Error: No AnnData objects were successfully loaded. Exiting.")
    exit()

# Concatenate all AnnData objects
print(f"\nConcatenating {len(adatas)} samples...")
adata = ad.concat(adatas, axis=0, join='outer', merge='same')
adata.var_names_make_unique() # Ensure genes are unique after concatenation
print(f"Combined AnnData object shape: {adata.shape} (Cells x Genes)")

# --- 3. Merge External Metadata ---
print("\nMerging external metadata...")
try:
    # Use compression='infer' just in case it's a .csv.gz even if not explicitly named
    external_meta_df = pd.read_csv(EXTERNAL_METADATA_FILE_PATH, compression='infer')

    # The external metadata file (GSE186433_metadata_percell.csv) typically has:
    # - A column like 'orig.ident' (e.g., 'g001')
    # - A column like 'barcode' (the raw 10x cell barcode, e.g., 'AAACCTGAGCGGATTG-1')
    # We need to create a combined ID in this DataFrame to match adata.obs_names.

    # Check for expected columns
    if 'orig.ident' not in external_meta_df.columns:
        raise KeyError("'orig.ident' column not found in external metadata. Please check its name.")
    if 'barcode' not in external_meta_df.columns:
        raise KeyError("'barcode' column not found in external metadata. Please check its name.")

    # Create a map from short sample ID ('g001') to full sample prefix ('GSM5651509_g001')
    # This map is built from the sample prefixes successfully loaded from your 10x files.
    short_to_full_id_map = {prefix.split('_')[1]: prefix for prefix in adata.obs['original_sample_prefix'].unique()}
    external_meta_df['full_sample_prefix'] = external_meta_df['orig.ident'].map(short_to_full_id_map)

    # Create the combined cell ID in external_meta_df to match adata.obs_names
    external_meta_df['combined_cell_id'] = external_meta_df['full_sample_prefix'] + '_' + external_meta_df['barcode']

    # Set this new combined ID as the index for merging
    external_meta_df.set_index('combined_cell_id', inplace=True)

    # Filter external_meta_df to only rows that exist in our loaded adata (important for memory and accuracy)
    external_meta_df = external_meta_df.loc[external_meta_df.index.intersection(adata.obs_names)]

    # Merge with adata.obs
    original_obs_columns = adata.obs.columns.tolist()
    adata.obs = adata.obs.merge(external_meta_df, left_index=True, right_index=True, how='left', suffixes=('', '_ext'))

    # Clean up duplicated columns (e.g., 'sample_id_ext') if original columns were loaded from 10x meta
    for col in original_obs_columns:
        if f"{col}_ext" in adata.obs.columns and col != 'original_sample_prefix': # Don't overwrite essential generated info
            adata.obs[col] = adata.obs[col].fillna(adata.obs[f"{col}_ext"])
            adata.obs.drop(columns=[f"{col}_ext"], inplace=True)

    new_obs_columns = [col for col in adata.obs.columns if col not in original_obs_columns and not col.endswith('_ext')]
    if new_obs_columns:
        print(f"  Successfully merged external metadata. Added columns: {new_obs_columns}")
    else:
        print("  No new columns were added from external metadata. Check index alignment and column names.")

except FileNotFoundError:
    print(f"  Warning: External metadata file not found at '{EXTERNAL_METADATA_FILE_PATH}'. Skipping merge.")
except KeyError as e:
    print(f"  Warning: Missing expected column for metadata merge: {e}. Skipping merge.")
    print("  Please check if 'orig.ident' and 'barcode' columns exist in your metadata CSV and their capitalization.")
except Exception as e:
    print(f"  Warning: An error occurred during metadata merge. Skipping merge. Error: {e}")
    print("  Please inspect your external metadata CSV and ensure its structure matches expected format for merging.")


# --- 4. Simple Scanpy Preprocessing ---
print("\nStarting simple Scanpy preprocessing (QC, Normalization, Log-transformation)...")

# Basic filtering
print(f"  Initial cells: {adata.n_obs}, genes: {adata.n_vars}")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
print(f"  After basic filtering: cells: {adata.n_obs}, genes: {adata.n_vars}")

# Normalize total counts per cell
sc.pp.normalize_total(adata, target_sum=1e4)
# Log-transform the data
sc.pp.log1p(adata)
print("  Normalization and log-transformation complete.")


# --- 5. Save Core AnnData Components to Parquet ---
print("\nSaving core AnnData components to Parquet...")

# Save adata.X (Expression Matrix - now normalized and log-transformed)
expression_parquet_path = os.path.join(OUTPUT_DIR_NAME, "expression.parquet")
df_expression = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
df_expression.index.name = "cell_id"
df_expression.to_parquet(expression_parquet_path, index=True)
print(f"Saved expression data to: {expression_parquet_path}")

# Save adata.var (Gene Metadata)
gene_metadata_parquet_path = os.path.join(OUTPUT_DIR_NAME, "gene_metadata.parquet")
adata.var.index.name = "gene_id"
adata.var.to_parquet(gene_metadata_parquet_path, index=True)
print(f"Saved gene metadata to: {gene_metadata_parquet_path}")

# Save adata.obs (Cell Metadata)
cell_metadata_parquet_path = os.path.join(OUTPUT_DIR_NAME, "cell_metadata.parquet")
for col in adata.obs.select_dtypes(include=['category', 'object']).columns:
    adata.obs[col] = adata.obs[col].astype(str)
adata.obs.index.name = "cell_id"
adata.obs.to_parquet(cell_metadata_parquet_path, index=True)
print(f"Saved cell metadata to: {cell_metadata_parquet_path}")


# --- 6. Perform PCA and save results ---
print(f"\nStarting PCA with {N_PCA_COMPONENTS} components...")

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=N_TOP_HVGS)
adata_for_dr = adata[:, adata.var['highly_variable']].copy()
print(f"  Identified {adata_for_dr.shape[1]} highly variable genes for PCA.")

X_for_pca = adata_for_dr.X.toarray()

if APPLY_SCALING_BEFORE_PCA:
    print("  Scaling HVG data before PCA...")
    scaler = StandardScaler()
    X_for_pca = scaler.fit_transform(X_for_pca)
else:
    print("  Skipping scaling before PCA.")

pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
pca_transformed_data = pca.fit_transform(X_for_pca)

pca_columns = [f"PC{i+1}" for i in range(N_PCA_COMPONENTS)]
df_pca = pd.DataFrame(pca_transformed_data, index=adata.obs_names, columns=pca_columns)
df_pca.index.name = "cell_id"

pca_parquet_path = os.path.join(OUTPUT_DIR_NAME, "pca_embeddings.parquet")
df_pca.to_parquet(pca_parquet_path, index=True)
print(f"Saved PCA embeddings to: {pca_parquet_path}")

df_explained_variance = pd.DataFrame({
    'PrincipalComponent': [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    'ExplainedVarianceRatio': pca.explained_variance_ratio_,
    'CumulativeExplainedVarianceRatio': np.cumsum(pca.explained_variance_ratio_)
})
explained_variance_parquet_path = os.path.join(OUTPUT_DIR_NAME, "pca_explained_variance.parquet")
df_explained_variance.to_parquet(explained_variance_parquet_path, index=False)
print(f"Saved PCA explained variance ratio to: {explained_variance_parquet_path}")


# --- 7. Perform UMAP and save results ---
print(f"\nStarting UMAP with {N_UMAP_COMPONENTS} components...")
X_for_umap = pca_transformed_data

reducer = umap.UMAP(n_components=N_UMAP_COMPONENTS,
                    n_neighbors=UMAP_N_NEIGHBORS,
                    min_dist=UMAP_MIN_DIST,
                    random_state=42,
                    transform_seed=42)

umap_embeddings = reducer.fit_transform(X_for_umap)

umap_columns = [f"UMAP{i+1}" for i in range(N_UMAP_COMPONENTS)]
df_umap = pd.DataFrame(umap_embeddings, index=adata.obs_names, columns=umap_columns)
df_umap.index.name = "cell_id"

umap_parquet_path = os.path.join(OUTPUT_DIR_NAME, "umap_embeddings.parquet")
df_umap.to_parquet(umap_parquet_path, index=True)
print(f"Saved UMAP embeddings to: {umap_parquet_path}")


# --- 8. Save Highly Variable Genes Metadata ---
if 'highly_variable' in adata.var.columns and adata.var['highly_variable'].any():
    df_hvg = adata.var[adata.var['highly_variable']].copy()
    hvg_metadata_parquet_path = os.path.join(OUTPUT_DIR_NAME, "highly_variable_gene_metadata.parquet")
    df_hvg.index.name = "gene_id"
    df_hvg.to_parquet(hvg_metadata_parquet_path, index=True)
    print(f"Saved highly variable gene metadata to: {hvg_metadata_parquet_path}")
else:
    print("No highly variable genes marked. Skipping saving HVG metadata.")


# --- 9. Save Basic Gene Statistics ---
print("\nCalculating basic gene statistics...")
df_gene_stats = pd.DataFrame(index=adata.var_names)
df_gene_stats.index.name = "gene_id"

df_gene_stats['mean_expression'] = np.asarray(adata.X.toarray()).mean(axis=0)
df_gene_stats['n_cells_expressed'] = np.asarray(adata.X.toarray() > 0).sum(axis=0)

gene_stats_parquet_path = os.path.join(OUTPUT_DIR_NAME, "gene_statistics.parquet")
df_gene_stats.to_parquet(gene_stats_parquet_path, index=True)
print(f"Saved gene statistics to: {gene_stats_parquet_path}")


print(f"\nAll essential Parquet files for Macaque Retina Aging Atlas have been created in the '{OUTPUT_DIR_NAME}' directory.")
print("You can now use these files for your submission to the hackathon!")