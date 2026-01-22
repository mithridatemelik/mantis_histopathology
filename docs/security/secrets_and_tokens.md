# secrets and tokens

## non-negotiable rules

- **never hardcode** tokens in notebooks, python modules, yaml, or markdown.
- **never print** tokens to stdout.
- prefer **Colab Secrets** or environment variables.

This repository intentionally contains **no embedded secrets**.

---

## required

### `OPENAI_API_KEY`
Used by the **agentic stages**:

- stage 04: agent 1 cluster cleanup
- stage 05: agent 2 cluster linking
- stage 06: knowledge graph construction (optional LLM enrichment)

Set via **Colab Secrets**:

1. Open **Runtime → Secrets**
2. Add:
   - key: `OPENAI_API_KEY`
   - value: your OpenAI API key

The notebooks read it with:

- `os.environ["OPENAI_API_KEY"]`
- or `google.colab.userdata.get("OPENAI_API_KEY")`

✅ The key is never printed.

---

## optional

### `HF_TOKEN`
Only needed if you:
- download private Hugging Face repos / datasets, or
- access gated artifacts.

### `MANTIS_TOKEN`
Only needed if you enable optional upload-to-Mantis API steps.

---

## local runs (non-colab)

You can set env vars in a terminal:

```bash
export OPENAI_API_KEY="..."
export HF_TOKEN="..."
export MANTIS_TOKEN="..."
```

---

## what to do if a key was leaked

If you ever accidentally committed a token:

1. **revoke it immediately** (OpenAI / Hugging Face / Mantis dashboard)
2. remove it from the repo history (git filter-repo / BFG)
3. rotate secrets and re-run notebooks

---

## automated scans

Before sharing, scan the repo for patterns like:

- `sk-...` (OpenAI)
- `hf_...` (Hugging Face)
- `Bearer ...`

This repo's refactor process includes such scans and removes any matches.
