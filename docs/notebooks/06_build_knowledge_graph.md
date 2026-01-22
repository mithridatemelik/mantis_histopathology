# 06_build_knowledge_graph.ipynb

## purpose
Build a knowledge graph from:
- agent 1 semantic clusters
- agent 2 cluster links
- dataset + label membership derived from `items_after_agent2.parquet`

By default this stage **includes dataset nodes** so multi-dataset runs are represented in the KG.

## inputs (reads)
- `exports/stage_04_agent1_cleanup/clusters_semantic.parquet`
- `exports/stage_05_agent2_linking/cluster_links.parquet`
- `exports/stage_05_agent2_linking/items_after_agent2.parquet`

## outputs (writes)
Directory: `exports/stage_06_knowledge_graph/`

Core parquet:
- `kg_nodes.parquet`
- `kg_edges.parquet`
- `kg_provenance.parquet`
- `kg_summary.parquet`

Visualization:
- `kg_visualization.html`

Optional diagnostics:
- `eda/` (KG stats, degree plot, parquet audit)

Optional exports:
- `neo4j/nodes.csv`, `neo4j/rels.csv`, `neo4j/load.cypher`
- `kg.ttl` (RDF Turtle; requires `rdflib`)
