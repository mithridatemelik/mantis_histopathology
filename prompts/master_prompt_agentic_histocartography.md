# master prompt: agentic histocartography colab workflow (dag + parquet-first + knowledge graph)

This file preserves the design brief used to build this repository’s DAG notebooks.

It is intentionally **token-free** and safe to share.

---

## primary goals

Deliver a clean, **Google Colab–ready**, **modular notebook DAG** that produces **Mantis-ready exports** from histopathology patch datasets and adds a **real, live (non-mocked) agentic pipeline** for:

1) **cluster cleanup / semantic cluster building** (agent 1)  
2) **post-cleanup cluster linking** (agent 2)  
3) **final knowledge graph construction** from the agentic outputs

Architecture reference:

**user → prompt → agent 1 (memory + reasoning + chat) → agent 2 (tools + database + human-in-the-loop) → output**

---

## non-negotiable constraints (summary)

- secrets must never be hardcoded; use Colab Secrets / env vars
- notebooks must run end-to-end in Colab
- parquet-first data contracts; every stage writes parquet, downstream reads parquet
- explicit DAG with stage folders under `exports/`
- agent 1 and agent 2 must call a real LLM endpoint (OpenAI)
- all filenames lowercase and meaningful; deprecate legacy artifacts
