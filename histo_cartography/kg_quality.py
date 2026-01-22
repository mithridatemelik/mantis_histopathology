from __future__ import annotations

"""Knowledge graph quality checks + diagnostic plots.

This module supports stage 06 "KG health gates" and visual diagnostics.

We intentionally keep it:
- dependency-light (networkx + matplotlib)
- evidence-aware (warn if edges lack evidence)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame):
    import networkx as nx  # type: ignore

    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(str(r["node_id"]), node_type=str(r.get("node_type", "")), name=str(r.get("name", "")))
    for _, r in edges.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]), rel=str(r.get("rel", "")), weight=float(r.get("weight", 0.0)))
    return G


def kg_health_summary(nodes: pd.DataFrame, edges: pd.DataFrame) -> Dict[str, Any]:
    import networkx as nx  # type: ignore

    G = build_graph(nodes, edges)

    node_type_counts = nodes["node_type"].value_counts(dropna=False).to_dict() if "node_type" in nodes.columns else {}
    rel_counts = edges["rel"].value_counts(dropna=False).to_dict() if "rel" in edges.columns else {}

    # Orphans
    deg = dict(G.degree())
    orphan_nodes = [n for n, d in deg.items() if d == 0]

    # Components (weakly)
    comps = list(nx.weakly_connected_components(G))
    comp_sizes = sorted([len(c) for c in comps], reverse=True)

    # Degree stats
    deg_vals = np.asarray(list(deg.values()), dtype=float) if deg else np.asarray([], dtype=float)

    out: Dict[str, Any] = {
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "node_type_counts": {str(k): int(v) for k, v in node_type_counts.items()},
        "rel_counts": {str(k): int(v) for k, v in rel_counts.items()},
        "n_orphan_nodes": int(len(orphan_nodes)),
        "orphan_nodes_sample": orphan_nodes[:25],
        "n_components_weak": int(len(comp_sizes)),
        "component_sizes_top": comp_sizes[:25],
        "degree_min": float(deg_vals.min()) if deg_vals.size else None,
        "degree_max": float(deg_vals.max()) if deg_vals.size else None,
        "degree_mean": float(deg_vals.mean()) if deg_vals.size else None,
        "degree_p99": float(np.quantile(deg_vals, 0.99)) if deg_vals.size else None,
    }

    # Hierarchy cycle check for subtype_of
    if "rel" in edges.columns:
        sub = edges[edges["rel"].astype(str).str.lower() == "subtype_of"]
        if len(sub):
            H = nx.DiGraph()
            H.add_edges_from([(str(r["src"]), str(r["dst"])) for _, r in sub.iterrows()])
            try:
                cyc = list(nx.simple_cycles(H))
            except Exception:
                cyc = []
            out["n_subtype_edges"] = int(len(sub))
            out["n_subtype_cycles"] = int(len(cyc))
            out["subtype_cycles_sample"] = [c[:10] for c in cyc[:5]]
        else:
            out["n_subtype_edges"] = 0
            out["n_subtype_cycles"] = 0

    return out


def plot_bar_counts(series: pd.Series, *, title: str, xlabel: str = "", ylabel: str = "count"):
    import matplotlib.pyplot as plt  # type: ignore

    s = series.copy()
    fig = plt.figure(figsize=(8, 4))
    plt.bar(s.index.astype(str), s.values)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    return fig


def plot_degree_distribution(G, *, title: str = "Degree distribution (in+out)"):
    import matplotlib.pyplot as plt  # type: ignore

    deg = [d for _, d in G.degree()]
    fig = plt.figure(figsize=(7, 4))
    plt.hist(deg, bins=30, edgecolor="black")
    plt.title(title)
    plt.xlabel("degree")
    plt.ylabel("count of nodes")
    plt.tight_layout()
    return fig


def plot_component_sizes(G, *, title: str = "Connected components (weak) sizes"):
    import matplotlib.pyplot as plt  # type: ignore
    import networkx as nx  # type: ignore

    sizes = [len(c) for c in nx.weakly_connected_components(G)]
    sizes = sorted(sizes, reverse=True)
    fig = plt.figure(figsize=(7, 4))
    plt.bar(range(min(30, len(sizes))), sizes[:30])
    plt.title(title)
    plt.xlabel("component rank")
    plt.ylabel("size")
    plt.tight_layout()
    return fig
