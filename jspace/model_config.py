"""Active experiment model configuration.

Default Phase target: LiquidAI/LFM2-2.6B (public post-trained LFM2, NOT Antidoom-trained).

Notes
-----
- Blog's "early LFM2.5-2.6B" with 10.2% loops is NOT public.
- Public `LiquidAI/LFM2-2.6B` is the closest released 2.6B Liquid model.
- No separate Base/Instruct for this size: one post-trained checkpoint with
  *dynamic hybrid reasoning* (<think>...</think> on complex prompts).
  Do NOT force-disable thinking — leave the model's default behavior.
- Antidoom-fixed weights are NOT published on HF for this model; downloading
  LiquidAI/LFM2-2.6B is safe (pre-Antidoom public release).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    slug: str  # filesystem-safe short name for results/
    display_name: str
    # Precision preference for local HF path (vLLM may use fp8 separately)
    hf_quantize: bool
    # Pre-fitted J-lens (None → must fit or run generation-only)
    lens_path: Path | None
    lens_hf_repo: str | None
    lens_hf_file: str | None
    # Hybrid Liquid architecture needs special J-lens handling
    hybrid_architecture: bool
    # Leave thinking/reasoning at model defaults
    force_disable_thinking: bool = False


LFM2_26B = ModelConfig(
    model_id="LiquidAI/LFM2-2.6B",
    slug="lfm2-2.6b",
    display_name="LFM2-2.6B",
    hf_quantize=True,  # NF4 for 8GB laptop; Colab may use bf16/fp8 via vLLM
    lens_path=ROOT / "lenses" / "lfm2-2.6b.pt",
    lens_hf_repo=None,  # no neuronpedia pre-fit
    lens_hf_file=None,
    hybrid_architecture=True,
    force_disable_thinking=False,
)

QWEN35_4B = ModelConfig(
    model_id="Qwen/Qwen3.5-4B",
    slug="qwen3.5-4b",
    display_name="Qwen3.5-4B",
    hf_quantize=True,
    lens_path=ROOT / "lenses" / "qwen3.5-4b.pt",
    lens_hf_repo="neuronpedia/jacobian-lens",
    lens_hf_file="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt",
    hybrid_architecture=False,
    force_disable_thinking=False,
)

_REGISTRY = {
    "lfm2-2.6b": LFM2_26B,
    "LiquidAI/LFM2-2.6B": LFM2_26B,
    "qwen3.5-4b": QWEN35_4B,
    "Qwen/Qwen3.5-4B": QWEN35_4B,
}


def get_active_model() -> ModelConfig:
    """Resolve model from JLENS_MODEL env (default: LFM2-2.6B)."""
    key = os.environ.get("JLENS_MODEL", "LiquidAI/LFM2-2.6B").strip()
    hf_q = os.environ.get("JLENS_HF_QUANTIZE")
    if key in _REGISTRY:
        cfg = _REGISTRY[key]
        if hf_q is not None:
            return ModelConfig(
                model_id=cfg.model_id,
                slug=cfg.slug,
                display_name=cfg.display_name,
                hf_quantize=hf_q == "1",
                lens_path=cfg.lens_path,
                lens_hf_repo=cfg.lens_hf_repo,
                lens_hf_file=cfg.lens_hf_file,
                hybrid_architecture=cfg.hybrid_architecture,
                force_disable_thinking=cfg.force_disable_thinking,
            )
        return cfg
    # Allow raw HF ids not in registry
    slug = key.split("/")[-1].lower().replace("_", "-")
    return ModelConfig(
        model_id=key,
        slug=slug,
        display_name=slug,
        hf_quantize=os.environ.get("JLENS_HF_QUANTIZE", "1") == "1",
        lens_path=ROOT / "lenses" / f"{slug}.pt",
        lens_hf_repo=None,
        lens_hf_file=None,
        hybrid_architecture="lfm" in slug.lower() or "liquid" in key.lower(),
        force_disable_thinking=False,
    )


def artifact_paths(cfg: ModelConfig | None = None) -> dict[str, Path]:
    """Canonical results paths for the active model."""
    cfg = cfg or get_active_model()
    results = ROOT / "results"
    return {
        "results": results,
        "workspace_band": results / f"workspace_band_{cfg.slug}.json",
        "workspace_band_metrics": results / f"workspace_band_metrics_{cfg.slug}.json",
        "trigger_csv": results / f"trigger_tokens_{cfg.slug}.csv",
        "trigger_log": results / "trigger_gen_log.jsonl",
        "baseline_summary": results / "baseline_pass_summary.json",
        "baseline_checkpoint": results / "checkpoints" / f"baseline_pass_{cfg.slug}.json",
        "prompt_sample": results / "prompt_sample_ids.json",
        "jlens_fit_status": results / f"jlens_fit_status_{cfg.slug}.json",
        "run_report": results / f"RUN_REPORT_{cfg.slug}_antidoom_mix_200.md",
        "exp2": results / "exp2",
        "exp3": results / "exp3",
        "lens": cfg.lens_path,
    }
