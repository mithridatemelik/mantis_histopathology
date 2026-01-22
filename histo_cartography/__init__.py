"""Histopathology multimodal cartography + knowledge graph (Colab-ready).

This package is intentionally self-contained and conservative about dependencies
to avoid Colab binary-compatibility crashes.

Project focus:
- real histopathology images (CRC-VAL-HE-7K / NCT-CRC-HE-100K via Zenodo)
- modality-specific embeddings (image/text/morphology)
- fusion into a configurable super-vector
- cartography (2D map) + clustering + labeling
- knowledge graph construction + exports (Parquet/RDF/Neo4j CSV)
- crash-safe checkpointing + manifests + resume mode

"""

__all__ = [
    "paths",
    "runtime",
    "checkpoint",
    "datasets",
    "schema",
    "embeddings",
    "eda",
    "eda_reports",
    "viz",
    "critic",
    "image_viz",
    "cartography",
    "clustering",
    "relationship_verification",
    "kg",
    "kg_quality",
    "kg_qa",
    "exports",
    "semantic",
    "stats_tests",
    "debug_tools",
    "artifact_registry",
    "agentic",
]

__version__ = "0.1.0"
