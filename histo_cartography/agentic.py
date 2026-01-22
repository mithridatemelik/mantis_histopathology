from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


RELATIONSHIP = Literal["same_as", "subtype_of", "overlaps_with", "related_to", "unrelated"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_text(s: str) -> str:
    return sha256(s.encode("utf-8")).hexdigest()


def try_load_colab_secret(secret_name: str) -> Optional[str]:
    """Best-effort read from google.colab.userdata without hard dependency."""
    try:
        from google.colab import userdata  # type: ignore

        v = userdata.get(secret_name)
        if v:
            return str(v)
    except Exception:
        return None
    return None


def get_env_or_colab_secret(secret_name: str, *, also_try_alt_names: Sequence[str] = ()) -> Optional[str]:
    """Get secret from env first, else Colab secrets."""
    v = os.environ.get(secret_name)
    if v:
        return v
    # Try colab userdata
    v = try_load_colab_secret(secret_name)
    if v:
        return v
    for alt in also_try_alt_names:
        v = os.environ.get(alt)
        if v:
            return v
        v = try_load_colab_secret(alt)
        if v:
            return v
    return None


def ensure_openai_api_key() -> str:
    key = get_env_or_colab_secret("OPENAI_API_KEY", also_try_alt_names=("openai_api_key", "histopathology"))
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not found.\n\n"
            "In Colab, set it via:\n"
            "  1) 🔑 Runtime → Secrets → Add new secret (key: OPENAI_API_KEY)\n"
            "  2) Or: %env OPENAI_API_KEY=... (not recommended for shared notebooks)\n"
        )
    # Never print keys.
    os.environ["OPENAI_API_KEY"] = key
    return key


def get_openai_client():
    """Return an OpenAI client (openai>=1.x)."""
    key = ensure_openai_api_key()
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "openai python package is not installed.\n"
            "Install in Colab: !pip -q install openai\n"
        ) from e
    return OpenAI(api_key=key)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON from an LLM response, with a fallback that extracts the first {...} block."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    raise ValueError("Failed to parse JSON from model response.")


def chat_json(
    *,
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_retries: int = 3,
    sleep_s: float = 1.0,
) -> Dict[str, Any]:
    """Call OpenAI Chat Completions and return parsed JSON."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=float(temperature),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = resp.choices[0].message.content or ""
            return _extract_json(text)
        except Exception as e:
            last_err = e
            time.sleep(float(sleep_s) * (attempt + 1))
    raise RuntimeError(f"OpenAI call failed after {max_retries} retries: {last_err}")


# ------------------------
# Agent 1: cluster cleanup
# ------------------------

AGENT1_SYSTEM = """You are Agent 1: a careful histopathology semantic cartographer.
You receive a summary of one embedding cluster and you must propose:
- a short, unique cluster_name (3–6 words, lowercase, no punctuation except hyphens)
- a cluster_description (1–3 sentences)
- cluster_keywords (3–8 lowercase keywords)

Rules:
- Do not mention file paths or IDs.
- Do not include secrets.
- Prefer medically meaningful terms when possible, but stay within the evidence given.
- Output strictly valid JSON with keys: cluster_name, cluster_description, cluster_keywords.
"""

def build_agent1_user_prompt(cluster_summary: Dict[str, Any]) -> str:
    # Keep the prompt compact but evidence-rich.
    return (
        "Summarize this cluster with a name/description/keywords.\n\n"
        f"cluster_id: {cluster_summary.get('cluster_id')}\n"
        f"n_items: {cluster_summary.get('n_items')}\n"
        f"dominant_labels: {cluster_summary.get('dominant_labels')}\n"
        f"sample_texts: {cluster_summary.get('sample_texts')}\n"
        "\nReturn JSON only."
    )


def cluster_signature(cluster_summary: Dict[str, Any]) -> str:
    # Stable-ish signature based on evidence (not on LLM output).
    payload = {
        "cluster_id": int(cluster_summary.get("cluster_id", -999)),
        "n_items": int(cluster_summary.get("n_items", 0)),
        "dominant_labels": cluster_summary.get("dominant_labels", []),
        "sample_texts": cluster_summary.get("sample_texts", [])[:5],
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def ensure_unique_names(clusters_semantic: pd.DataFrame) -> pd.DataFrame:
    """Deterministically ensure cluster_name uniqueness.

    If duplicates exist, append '-c{cluster_id}' to the later duplicates.
    """
    df = clusters_semantic.copy()
    if "cluster_name" not in df.columns:
        return df
    seen: Dict[str, int] = {}
    new_names: List[str] = []
    for _, r in df.iterrows():
        name = str(r.get("cluster_name", "")).strip().lower()
        cid = int(r.get("cluster_id", -1))
        if not name:
            name = f"cluster-c{cid}"
        if name in seen:
            name2 = f"{name}-c{cid}"
            new_names.append(name2)
        else:
            seen[name] = cid
            new_names.append(name)
    df["cluster_name"] = new_names
    return df


def run_agent1_cluster_cleanup(
    *,
    clusters_summary: pd.DataFrame,
    memory_path: str,
    model: str,
    temperature: float = 0.2,
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clusters_semantic_df, memory_df).

    - clusters_summary: one row per cluster with evidence columns
    - memory_path: parquet path for persisted memory
    """
    client = get_openai_client()
    mem_path = Path(memory_path)
    mem_path.parent.mkdir(parents=True, exist_ok=True)

    if mem_path.exists():
        memory_df = pd.read_parquet(mem_path)
    else:
        memory_df = pd.DataFrame(
            columns=[
                "cluster_id",
                "signature",
                "model",
                "temperature",
                "prompt",
                "response_json",
                "cluster_name",
                "cluster_description",
                "cluster_keywords",
                "created_at_utc",
            ]
        )

    out_rows: List[Dict[str, Any]] = []
    for _, row in clusters_summary.iterrows():
        cid = int(row["cluster_id"])
        if cid == -1:
            continue
        summary = row.to_dict()
        sig = cluster_signature(summary)

        existing = memory_df[(memory_df["cluster_id"] == cid) & (memory_df["signature"] == sig)]
        if (not force_rebuild) and (len(existing) > 0):
            rec = existing.iloc[-1].to_dict()
            out_rows.append(
                {
                    "cluster_id": cid,
                    "cluster_name": rec.get("cluster_name"),
                    "cluster_description": rec.get("cluster_description"),
                    "cluster_keywords": (json.loads(rec.get("cluster_keywords")) if isinstance(rec.get("cluster_keywords"), str) and rec.get("cluster_keywords").strip().startswith("[") else rec.get("cluster_keywords")),
                    "n_items": int(summary.get("n_items", 0)),
                    "dominant_labels": summary.get("dominant_labels"),
                }
            )
            continue

        prompt = build_agent1_user_prompt(summary)
        resp = chat_json(
            client=client,
            model=model,
            system_prompt=AGENT1_SYSTEM,
            user_prompt=prompt,
            temperature=temperature,
        )

        # Normalize fields
        name = str(resp.get("cluster_name", "")).strip().lower()
        desc = str(resp.get("cluster_description", "")).strip()
        kws = resp.get("cluster_keywords", [])
        if isinstance(kws, str):
            # allow comma separated
            kws = [k.strip().lower() for k in kws.split(",") if k.strip()]
        if not isinstance(kws, list):
            kws = []
        kws = [str(k).strip().lower() for k in kws if str(k).strip()]

        out_rows.append(
            {
                "cluster_id": cid,
                "cluster_name": name,
                "cluster_description": desc,
                "cluster_keywords": kws,
                "n_items": int(summary.get("n_items", 0)),
                "dominant_labels": summary.get("dominant_labels"),
            }
        )

        memory_df = pd.concat(
            [
                memory_df,
                pd.DataFrame(
                    [
                        {
                            "cluster_id": cid,
                            "signature": sig,
                            "model": model,
                            "temperature": float(temperature),
                            "prompt": prompt,
                            "response_json": json.dumps(resp, ensure_ascii=False),
                            "cluster_name": name,
                            "cluster_description": desc,
                            "cluster_keywords": json.dumps(kws, ensure_ascii=False),
                            "created_at_utc": _utc_now_iso(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    clusters_semantic = pd.DataFrame(out_rows)
    clusters_semantic = ensure_unique_names(clusters_semantic)

    # Persist memory
    memory_df.to_parquet(mem_path, index=False)
    return clusters_semantic, memory_df


# ------------------------
# Agent 2: cluster linking
# ------------------------

AGENT2_SYSTEM = """You are Agent 2: a careful ontology linker for histopathology cluster concepts.

Given two semantic clusters, classify their relationship as exactly one of:
- same_as
- subtype_of
- overlaps_with
- related_to
- unrelated

Return strictly valid JSON with keys:
- relationship (one of the above)
- confidence (0.0 to 1.0)
- rationale (1-3 sentences, grounded in the provided evidence)

Rules:
- Do not invent medical claims beyond the evidence.
- Do not mention file paths or IDs.
- Do not include secrets.
"""

def candidate_pairs_from_centroids(
    centroids: pd.DataFrame,
    *,
    k: int = 5,
    min_sim: float = 0.3,
) -> pd.DataFrame:
    """Compute a candidate shortlist of cluster pairs based on cosine similarity of centroid vectors."""
    df = centroids.copy()
    if df.empty:
        return pd.DataFrame(columns=["src_cluster_id", "dst_cluster_id", "similarity"])
    # Expect columns: cluster_id, vector
    X = np.array(df["vector"].tolist(), dtype=np.float32)
    # normalize
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sims = X @ X.T
    cluster_ids = df["cluster_id"].astype(int).tolist()

    rows: List[Dict[str, Any]] = []
    for i, cid in enumerate(cluster_ids):
        # Exclude self
        sim_row = sims[i].copy()
        sim_row[i] = -1.0
        # top-k indices
        idx = np.argsort(sim_row)[::-1][: int(k)]
        for j in idx:
            s = float(sim_row[j])
            if s < float(min_sim):
                continue
            cid2 = int(cluster_ids[int(j)])
            # Store directed pair (cid -> cid2) for tool shortlist
            rows.append({"src_cluster_id": int(cid), "dst_cluster_id": int(cid2), "similarity": s})

    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand
    # Deduplicate pairs by keeping max similarity for same directed pair
    cand = cand.sort_values("similarity", ascending=False).drop_duplicates(["src_cluster_id", "dst_cluster_id"])
    return cand.reset_index(drop=True)


def link_signature(src: Dict[str, Any], dst: Dict[str, Any], similarity: float) -> str:
    payload = {
        "src_cluster_id": int(src.get("cluster_id", -999)),
        "dst_cluster_id": int(dst.get("cluster_id", -999)),
        "src_name": str(src.get("cluster_name", "")).strip().lower(),
        "dst_name": str(dst.get("cluster_name", "")).strip().lower(),
        "src_keywords": src.get("cluster_keywords", []),
        "dst_keywords": dst.get("cluster_keywords", []),
        "similarity": round(float(similarity), 4),
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def build_agent2_user_prompt(src: Dict[str, Any], dst: Dict[str, Any], similarity: float) -> str:
    """Build the Agent 2 prompt used to classify a relationship between two clusters.

    Keep this prompt:
      - evidence-first (no hallucinated ontology)
      - compact (cheap in tokens)
      - JSON-only output request for robust parsing
    """
    return f"""Classify the relationship between these two clusters.

cosine_similarity_centroid: {float(similarity):.4f}

Use only the evidence below. If evidence is weak, prefer 'related_to' or 'unrelated'.

SRC cluster_id: {src.get('cluster_id')}
SRC n_items: {src.get('n_items')}
SRC dominant_labels: {src.get('dominant_labels')}
SRC name: {src.get('cluster_name')}
SRC description: {src.get('cluster_description')}
SRC keywords: {src.get('cluster_keywords')}

DST cluster_id: {dst.get('cluster_id')}
DST n_items: {dst.get('n_items')}
DST dominant_labels: {dst.get('dominant_labels')}
DST name: {dst.get('cluster_name')}
DST description: {dst.get('cluster_description')}
DST keywords: {dst.get('cluster_keywords')}

Return JSON only."""




def normalize_relationship(x: Any) -> RELATIONSHIP:
    s = str(x or "").strip().lower()
    allowed = {"same_as", "subtype_of", "overlaps_with", "related_to", "unrelated"}
    return s if s in allowed else "related_to"  # type: ignore[return-value]


def run_agent2_cluster_linking(
    *,
    clusters_semantic: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    memory_path: str,
    model: str,
    temperature: float = 0.2,
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cluster_links_df, memory_df)."""
    client = get_openai_client()
    mem_path = Path(memory_path)
    mem_path.parent.mkdir(parents=True, exist_ok=True)

    if mem_path.exists():
        memory_df = pd.read_parquet(mem_path)
    else:
        memory_df = pd.DataFrame(
            columns=[
                "src_cluster_id",
                "dst_cluster_id",
                "signature",
                "model",
                "temperature",
                "prompt",
                "response_json",
                "relationship",
                "confidence",
                "rationale",
                "created_at_utc",
            ]
        )

    clus_map = {int(r["cluster_id"]): r for r in clusters_semantic.to_dict(orient="records")}

    out_rows: List[Dict[str, Any]] = []
    for _, pair in candidate_pairs.iterrows():
        src_id = int(pair["src_cluster_id"])
        dst_id = int(pair["dst_cluster_id"])
        sim = float(pair.get("similarity", 0.0))

        src = clus_map.get(src_id)
        dst = clus_map.get(dst_id)
        if not src or not dst:
            continue

        sig = link_signature(src, dst, sim)
        existing = memory_df[(memory_df["src_cluster_id"] == src_id) & (memory_df["dst_cluster_id"] == dst_id) & (memory_df["signature"] == sig)]
        if (not force_rebuild) and (len(existing) > 0):
            rec = existing.iloc[-1].to_dict()
            out_rows.append(
                {
                    "src_cluster_id": src_id,
                    "dst_cluster_id": dst_id,
                    "similarity": sim,
                    "relationship": rec.get("relationship"),
                    "confidence": float(rec.get("confidence") or 0.0),
                    "rationale": rec.get("rationale"),
                }
            )
            continue

        prompt = build_agent2_user_prompt(src, dst, sim)
        resp = chat_json(
            client=client,
            model=model,
            system_prompt=AGENT2_SYSTEM,
            user_prompt=prompt,
            temperature=temperature,
        )

        rel = normalize_relationship(resp.get("relationship"))
        conf = float(resp.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        rationale = str(resp.get("rationale", "")).strip()

        out_rows.append(
            {
                "src_cluster_id": src_id,
                "dst_cluster_id": dst_id,
                "similarity": sim,
                "relationship": rel,
                "confidence": conf,
                "rationale": rationale,
            }
        )

        memory_df = pd.concat(
            [
                memory_df,
                pd.DataFrame(
                    [
                        {
                            "src_cluster_id": src_id,
                            "dst_cluster_id": dst_id,
                            "signature": sig,
                            "model": model,
                            "temperature": float(temperature),
                            "prompt": prompt,
                            "response_json": json.dumps(resp, ensure_ascii=False),
                            "relationship": rel,
                            "confidence": conf,
                            "rationale": rationale,
                            "created_at_utc": _utc_now_iso(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    links_df = pd.DataFrame(out_rows)
    memory_df.to_parquet(mem_path, index=False)
    return links_df, memory_df
