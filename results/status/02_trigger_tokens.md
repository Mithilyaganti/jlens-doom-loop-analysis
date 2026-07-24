# Trigger Token Extraction — Qwen3.5-4B

- Prompts attempted: 28
- Loops detected: 0 (0.0%)
- Unique trigger token IDs: 0
- Max new tokens: 2048, temperature: 0.01
- Quantization: bitsandbytes NF4, backend: HF transformers generate
- Table: results/trigger_tokens_qwen3.5-4b.csv (empirical only; empty if zero loops)
- Generation log: results/trigger_gen_log.jsonl

## Honest finding
Under this setup, **zero** generations met Liquid find_inner_repetition criteria
(min_repeats=4, min_total_repeated=60 chars). The empirical trigger CSV is therefore empty.
We do **not** invent frequencies. Exp1 uses Liquid operational _RESTART_WORDS seed as labeled
non-empirical seed where needed.

Liquid blog reported 22.9% loop rate for Qwen3.5-4B under their Antidoom vLLM pipeline
(max_new_tokens up to 4000). Possible drivers of the gap: generation backend, max length,
prompt sampling, chat-template details, model revision. Our measurement is the rate under
the conditions we actually ran.

Top-10 triggers:
`
(empty — no loops observed)
`
