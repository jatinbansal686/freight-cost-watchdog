# Token / cost log -- one full cold run

Model: `openai/gpt-oss-20b` via NVIDIA NIM (`https://integrate.api.nvidia.com/v1`), `temperature=0`.

- LLM calls: **40** (10 note_enrichment, 30 explanation)
- Total input (prompt) tokens: **11681**
- Total output (completion) tokens: **12875**
- Actual cost on the NVIDIA NIM free tier: **$0.00**
- Rough equivalent at a published `openai/gpt-oss-20b` rate ($0.03/1M input, $0.13/1M output tokens): **$0.0020** ($0.0004 input + $0.0017 output)
