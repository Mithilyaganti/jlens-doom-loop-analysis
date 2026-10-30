# Corpus notices

Selected prompts are redistributed under the upstream terms recorded in
`loop_lingua/licenses.py`. Checked 2026-10-30.

- **MMLU-ProX-Lite** (`li-lab/MMLU-ProX-Lite`): dataset card states MIT.
  Citation: Xuan et al., 2025, arXiv:2503.10497.
  The matched 300 are from the **test** split only (question ids 70–657).
  Validation (ids 0–69) is not included.
- **MMLU-Pro** (upstream of those items): Apache-2.0.
  Citation: Wang et al., 2024, arXiv:2406.01574.
- **MGSM** (`google-research/url-nlp` `mgsm/LICENSE`): CC-BY-4.0.
  Citation: Shi et al., 2022, arXiv:2210.03057.
  English items originate in GSM8K (Cobbe et al., 2021, arXiv:2110.14168, MIT).
  Hindi is not part of MGSM. The 250 English–Telugu pairs are a supplement,
  not part of the matched 300.
- Romanized Telugu is an algorithmic ISO 15919 transliteration of the Telugu
  prompt, produced here. It is adapted material; the Telugu source terms still apply.
- Tenglish slots are empty placeholders, not dataset text.

Gold letters are in `gold.jsonl` for a future scoring script. The sampler
does not read that file. Chain-of-thought fields from MMLU-ProX-Lite are not stored.
