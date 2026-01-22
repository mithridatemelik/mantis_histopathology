#!/usr/bin/env python
"""Regenerate workshop exports from checkpoints.

This script is meant to be run after merging the PR, to refresh:
  - exports/clustering/
  - exports/kg/
  - exports/kg_qa/
  - exports/semantic/

It reuses cached embeddings in checkpoints/ to avoid re-downloading images.

Requires:
  - pyarrow (parquet IO)
  - (optional) umap-learn + hdbscan for the sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from histo_cartography.embeddings import align_embeddings, fuse_embeddings_concat_pca, embed_text_tfidf_svd
from histo_cartography.clustering import (
    hdbscan_parameter_sweep,
    kmeans_baseline_exports,
    name_clusters_by_label_composition,
)
from histo_cartography.kg import build_kg_tables
from histo_cartography.kg_qa import write_kg_qa_exports
from histo_cartography.exports import save_parquet, export_rdf_turtle, export_neo4j_csv
from histo_cartography.semantic import write_minimal_ontology_turtle, write_mapping_spec_md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--force", action="store_true", help="Overwrite existing exports")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    cfg = yaml.safe_load((root / "pipeline_config.yaml").read_text())

    ckpt = root / cfg["paths"]["checkpoints_dir"]
    exp = root / cfg["paths"]["exports_dir"]

    # --- Load checkpoints
    items_df = pd.read_parquet(ckpt / "B" / "items.parquet")
    image_emb_df = pd.read_parquet(ckpt / "C" / "image_embeddings.parquet")
    morph_emb_df = pd.read_parquet(ckpt / "C" / "morph_embeddings.parquet")

    # --- Text embeddings (label-safe)
    text_cfg = (cfg.get("embeddings") or {}).get("text") or {}
    use_text = bool(text_cfg.get("use_text_modality", False))
    if "text" not in items_df.columns:
        items_df["text"] = ""
    if str(text_cfg.get("text_template_version", "v2_no_label")) == "v2_no_label":
        # defensive leakage scrub
        items_df.loc[items_df["text"].astype(str).str.contains("Label=", regex=False), "text"] = ""

    if use_text:
        text_emb_df = embed_text_tfidf_svd(
            items_df,
            n_components=int(text_cfg.get("svd_dim", 128)),
            max_features=int(text_cfg.get("max_features", 8192)),
            allow_label_fallback=False,
        )
    else:
        text_emb_df = pd.DataFrame(
            {
                "item_id": items_df["item_id"].tolist(),
                "modality": "text",
                "model_id": "text/disabled",
                "dim": 1,
                "vector": [[0.0] for _ in range(len(items_df))],
            }
        )

    # --- Fuse versions
    target_dim = int(cfg["embeddings"]["fusion"]["target_dim"])
    n_unique_texts = int(items_df["text"].fillna("").astype(str).nunique())
    n_labels = int(items_df["label"].nunique()) if "label" in items_df.columns else 0
    include_text_in_fusion = bool(use_text and (n_unique_texts > max(9, n_labels)))

    def _fuse(dfs):
        aligned = align_embeddings(dfs)
        return fuse_embeddings_concat_pca(aligned, target_dim=target_dim)

    fused_image = _fuse([image_emb_df])
    fused_image_morph = _fuse([image_emb_df, morph_emb_df])
    fused_full = _fuse([image_emb_df, morph_emb_df, text_emb_df]) if include_text_in_fusion else fused_image_morph

    # cache
    save_parquet(fused_image, ckpt / "C" / "fused_embeddings__image.parquet")
    save_parquet(fused_image_morph, ckpt / "C" / "fused_embeddings__image_morph.parquet")
    save_parquet(fused_full, ckpt / "C" / "fused_embeddings__full.parquet")
    # default
    save_parquet(fused_full, ckpt / "C" / "fused_embeddings.parquet")

    # --- Clustering exports
    clust_out_dir = exp / "clustering"
    clust_out_dir.mkdir(parents=True, exist_ok=True)
    gt_labels = items_df["label"].fillna("").astype(str)

    kmeans_k = int(((cfg.get("cartography") or {}).get("kmeans") or {}).get("k", 9))
    kmeans_assign_df, _ = kmeans_baseline_exports(fused=fused_full, labels=gt_labels, out_dir=clust_out_dir, k=kmeans_k)

    sweep_cfg = (cfg.get("cartography") or {}).get("clustering_sweep") or {}
    _, best_run, best_clusters = hdbscan_parameter_sweep(
        fused=fused_full,
        labels=gt_labels,
        out_dir=clust_out_dir,
        umap_n_neighbors=sweep_cfg.get("umap_n_neighbors", [10, 20, 50]),
        umap_min_dist=sweep_cfg.get("umap_min_dist", [0.0, 0.1, 0.3]),
        hdbscan_min_cluster_size=sweep_cfg.get("hdbscan_min_cluster_size", [5, 8, 10, 15]),
        hdbscan_min_samples=sweep_cfg.get("hdbscan_min_samples", [None, 5, 10]),
        umap_n_components_cluster=int(((cfg.get("cartography") or {}).get("clustering") or {}).get("umap_n_components_for_clustering", 10)),
        max_noise_ratio=float(((cfg.get("cartography") or {}).get("clustering") or {}).get("best_run", {}).get("max_noise_ratio", 0.40)),
        make_plots=True,
    )

    clusters_df = best_clusters if best_clusters is not None else kmeans_assign_df
    clusters_df, cluster_summary_df, comp_df = name_clusters_by_label_composition(
        clusters=clusters_df,
        items=items_df,
        label_col="label",
        label_name_map=None,
        min_label_fraction_for_name=float(((cfg.get("cartography") or {}).get("clustering") or {}).get("cluster_name", {}).get("min_label_fraction_for_name", 0.10)),
    )

    save_parquet(clusters_df, ckpt / "C" / "clusters.parquet")
    save_parquet(cluster_summary_df, ckpt / "C" / "cluster_summary.parquet")
    comp_df.to_csv(clust_out_dir / "cluster_label_composition.csv", index=False)

    # --- KG build + exports
    k_sim = int(cfg["kg"]["similarity_edges"]["k"])
    min_sim = float(cfg["kg"]["similarity_edges"]["min_sim"])

    embedding_version = str(fused_full["model_id"].iloc[0]) if ("model_id" in fused_full.columns and len(fused_full)) else "fused/unknown"
    pipeline_version = f"schema={cfg['project']['schema_version']};clustering={best_run or 'fallback'}"
    entities_df, edges_df, prov_df = build_kg_tables(
        items=items_df,
        clusters=clusters_df,
        fused=fused_full,
        k_sim=k_sim,
        min_sim=min_sim,
        embedding_version=embedding_version,
        pipeline_version=pipeline_version,
    )

    save_parquet(entities_df, ckpt / "D" / "entities.parquet")
    save_parquet(edges_df, ckpt / "D" / "edges.parquet")
    save_parquet(prov_df, ckpt / "D" / "provenance.parquet")

    kg_out_dir = exp / "kg"
    kg_out_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(entities_df, kg_out_dir / "entities.parquet")
    save_parquet(edges_df, kg_out_dir / "edges.parquet")
    save_parquet(prov_df, kg_out_dir / "provenance.parquet")

    base_iri = str(cfg.get("kg", {}).get("rdf", {}).get("base_iri", "http://example.org/histo/"))
    export_rdf_turtle(entities_df, edges_df, kg_out_dir / "consolidated_graph.ttl", base_iri=base_iri)
    export_neo4j_csv(entities_df, edges_df, kg_out_dir / "neo4j")

    sem_out_dir = exp / "semantic"
    sem_out_dir.mkdir(parents=True, exist_ok=True)
    write_minimal_ontology_turtle(sem_out_dir / "ontology.ttl", base_iri=base_iri)
    write_mapping_spec_md(sem_out_dir / "mapping_spec.md")

    # --- KG QA
    qa_out_dir = exp / "kg_qa"
    qa_out_dir.mkdir(parents=True, exist_ok=True)
    write_kg_qa_exports(entities_df, edges_df, items_df, qa_out_dir)

    print("✅ Regeneration complete")
    print("Clustering exports:", clust_out_dir)
    print("KG exports:", kg_out_dir)
    print("KG QA exports:", qa_out_dir)
    print("Semantic exports:", sem_out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
