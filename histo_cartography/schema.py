from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


# Canonical parquet-first schemas (explicit, typed).
ITEMS_SCHEMA_V1: Dict[str, str] = {
    "item_id": "string",
    "source": "string",
    "split": "string",
    "label": "string",
    "text": "string",
    "image_path": "string",
    "width": "int64",
    "height": "int64",
    "mpp": "float64",
}

EMBEDDINGS_SCHEMA_V1: Dict[str, str] = {
    "item_id": "string",
    "modality": "string",
    "model_id": "string",
    "dim": "int64",
    # vector stored as list[float] in parquet; in pandas we keep as object of list/np.ndarray
    "vector": "object",
}

FUSED_SCHEMA_V1: Dict[str, str] = {
    "item_id": "string",
    "model_id": "string",
    "dim": "int64",
    "vector": "object",
}

ENTITIES_SCHEMA_V1: Dict[str, str] = {
    "entity_id": "string",
    "entity_type": "string",
    "name": "string",
    "description": "string",
}

EDGES_SCHEMA_V1: Dict[str, str] = {
    "edge_id": "string",
    "src": "string",
    "dst": "string",
    "rel": "string",
    "weight": "float64",
    "provenance_id": "string",
}

PROVENANCE_SCHEMA_V1: Dict[str, str] = {
    "provenance_id": "string",
    "source_item_id": "string",
    "evidence_type": "string",
    "evidence": "string",
    "confidence": "float64",
}


def validate_df_schema(df: pd.DataFrame, schema: Dict[str, str], *, strict: bool = True) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    # Required columns
    for col in schema.keys():
        if col not in df.columns:
            errors.append(f"missing column: {col}")

    if strict:
        extra_cols = [c for c in df.columns if c not in schema]
        if extra_cols:
            errors.append(f"unexpected columns: {extra_cols[:20]}")

    # Dtype checks (best-effort; pandas dtypes can be tricky)
    for col, dtype in schema.items():
        if col not in df.columns:
            continue
        if dtype == "string":
            # allow object or string
            if str(df[col].dtype) not in ("object", "string"):
                errors.append(f"dtype mismatch for {col}: expected string-like, got {df[col].dtype}")
        elif dtype.startswith("int"):
            if not str(df[col].dtype).startswith("int") and str(df[col].dtype) != "Int64":
                errors.append(f"dtype mismatch for {col}: expected int, got {df[col].dtype}")
        elif dtype.startswith("float"):
            if not str(df[col].dtype).startswith("float"):
                errors.append(f"dtype mismatch for {col}: expected float, got {df[col].dtype}")
        elif dtype == "object":
            # accept anything
            pass

    ok = len(errors) == 0
    return ok, errors
