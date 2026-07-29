#!/usr/bin/env python
"""Re-cache Exp2 J-lens readouts with trigger-centered windows (fix truncation bug)."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("exp2_recache")


def main() -> int:
    from jspace.loading import load_stack, clear_cuda
    from jspace.model_config import get_active_model, artifact_paths
    from jspace.analysis import load_json
    from jspace.generation import cache_jlens_readouts_for_sequence, apply_chat_template

    cfg = get_active_model()
    paths = artifact_paths(cfg)
    band = load_json(paths["workspace_band"])
    ws_layers = list(band["key_workspace_layers"])
    stack = load_stack()
    if stack.lens is None:
        raise RuntimeError("lens required for recache")

    only_loops = os.environ.get("JLENS_EXP2_RECACHE_LOOPS_ONLY", "1") == "1"
    bases = [paths["exp2"] / "looping"]
    if not only_loops:
        bases.append(paths["exp2"] / "nonlooping")

    dirs: list[Path] = []
    for b in bases:
        if b.is_dir():
            dirs.extend(sorted(b.glob("p*")))

    logger.info("Recaching %d traces; ws_layers=%s", len(dirs), ws_layers)
    n_ok = 0
    n_fail = 0
    for i, d in enumerate(dirs):
        meta_path = d / "meta.json"
        tok_path = d / "tokens.pt"
        if not meta_path.is_file() or not tok_path.is_file():
            logger.warning("skip incomplete %s", d.name)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tok = torch.load(tok_path, map_location="cpu", weights_only=True)
        gen_ids = list(tok.get("generated_ids") or [])
        prompt_len = tok.get("prompt_len")
        prompt_text = meta.get("prompt_text") or ""

        # Reconstruct full_ids: prefer stored full_ids; else re-tokenize prompt_text.
        if "full_ids" in tok and tok["full_ids"] is not None:
            full_ids = torch.tensor([tok["full_ids"]], dtype=torch.long)
            if prompt_len is None:
                prompt_len = len(tok["full_ids"]) - len(gen_ids)
        else:
            if not prompt_text:
                logger.error("%s missing prompt_text; cannot reconstruct", d.name)
                n_fail += 1
                continue
            # meta prompt_text may already be chat-formatted (from generation).
            ids = stack.tokenizer(
                prompt_text, return_tensors="pt", truncation=False, add_special_tokens=False
            ).input_ids
            if prompt_len is None:
                prompt_len = int(ids.shape[1])
            # Match the original generation's prompt_len so trigger indices stay valid.
            cur = int(ids.shape[1])
            if cur > int(prompt_len):
                ids = ids[:, : int(prompt_len)]
                logger.warning("%s truncated re-encoded prompt %d -> %d", d.name, cur, prompt_len)
            elif cur < int(prompt_len):
                pad_id = stack.tokenizer.pad_token_id or stack.tokenizer.eos_token_id or 0
                pad = torch.full((1, int(prompt_len) - cur), int(pad_id), dtype=torch.long)
                ids = torch.cat([ids, pad], dim=1)
                logger.warning("%s padded re-encoded prompt %d -> %d", d.name, cur, prompt_len)
            full_ids = torch.cat(
                [ids, torch.tensor([gen_ids], dtype=torch.long)], dim=1
            )

        trig = meta.get("trigger_token_index") if meta.get("is_loop") else None
        try:
            cached = cache_jlens_readouts_for_sequence(
                stack,
                full_ids=full_ids,
                prompt_len=int(prompt_len),
                trigger_token_index=trig,
                workspace_layers=ws_layers,
                top_k=10,
                cache_tier3=False,
                max_ctx=1536,
            )
        except Exception as e:
            logger.exception("recache failed %s: %s", d.name, e)
            n_fail += 1
            clear_cuda()
            continue

        if cached.get("readouts") is None:
            logger.error("no readouts for %s", d.name)
            n_fail += 1
            clear_cuda()
            continue

        local = cached.get("trigger_local_in_readouts")
        if meta.get("is_loop") and local is None:
            logger.error(
                "%s still missing trigger_local_in_readouts after fix (trig=%s n_gp=%s)",
                d.name,
                trig,
                cached.get("n_gen_positions_cached"),
            )
            n_fail += 1
            clear_cuda()
            continue

        torch.save(cached["readouts"], d / "readouts.pt")
        if cached.get("residuals_ws"):
            torch.save(cached["residuals_ws"], d / "residuals_workspace_band.pt")
        # Persist reconstruction aids + mapping.
        torch.save(
            {
                "generated_ids": gen_ids,
                "prompt_len": int(prompt_len),
                "full_ids": full_ids[0].tolist(),
            },
            tok_path,
        )
        meta["trigger_local_in_readouts"] = local
        meta["n_gen_positions_cached"] = cached.get("n_gen_positions_cached")
        meta["trunc"] = cached.get("trunc")
        meta["recached_trigger_window"] = True
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        n_ok += 1
        logger.info(
            "recached %s (%d/%d) local_trig=%s n_gp=%s mode=%s",
            d.name,
            i + 1,
            len(dirs),
            local,
            cached.get("n_gen_positions_cached"),
            (cached.get("trunc") or {}).get("mode"),
        )
        clear_cuda()

    logger.info("done ok=%d fail=%d", n_ok, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
