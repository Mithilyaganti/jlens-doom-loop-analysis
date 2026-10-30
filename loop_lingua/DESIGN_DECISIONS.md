# Design decisions

Each choice, the alternative that was turned down, and why.

## Package boundary

**Choice.** New package `loop_lingua/`. Paper-1 (`jspace/`, `scripts/`, root README, `PROJECT_SPEC.md`) is not edited.

**Alternative.** Patch `TokenState` in place.

**Why.** The offset bug is real, and the paper-1 file is the baseline being measured. A patch would make the 149/250 count unreproducible. The copy of the Antidoom sampler is tested against `jspace.detection.find_inner_repetition`.

## D0 vs D1

**Choice.** D0 is the original sampler: 16-character fingerprints every 128 characters, threshold 4 repeats and 60 code points. D1 is the same numbers with an exact scan.

**Alternative.** Treat "code-point" as a different threshold, or replace D0 with the exact scan.

**Why.** Python `len` on the Antidoom string is already code points, so a second numeric rule would be a rename. The sampler can miss a loop that sits between sample points (there is a unit test for that). Keeping D0 identical to paper-1 is what makes a later language comparison comparable to the existing English pipeline. D1 is the control for sampler misses, which will not be uniform across scripts.

## D2 graphemes

**Choice.** Threshold of 60 extended grapheme clusters, using a UAX #29 subset: Mn/Me/Mc stay with the base, and a virama ties the next Indic consonant (so `క్ష` and `क्ष` are one cluster).

**Alternative.** Depend on the `regex` module's `\X`, or count code points only.

**Why.** Telugu and Hindi vowel signs are extra code points. A 60-code-point Telugu loop is fewer aksharas than 60 English letters, so D0/D1 will fire more easily on those scripts. D2 is the comparison that holds visual letters roughly fixed. A full UAX #29 implementation is not required for these scripts and would be an extra dependency. Hangul and emoji flags are not in the prompt set; that limit is stated in `graphemes.py`.

## D3 token threshold

**Choice.** Detect on the token-id sequence. The minimum span is `round(60 / reference_chars_per_token)`, and at least 4. The default reference is 4.0 characters per token.

**Alternative.** Always use 60 tokens, or retokenize with a whitespace splitter.

**Why.** The Antidoom rule is a character budget. English BPE spends fewer tokens per character than Telugu byte fallback, so a raw character rule is a longer generation for Telugu. D3 holds the token budget at the English-equivalent of 60 characters. 4.0 is a scale anchor so unit tests do not need the Qwen tokenizer. The study should pass the English fertility measured by `fertility.py` instead of leaving 4.0 in place. D3 does not run without token ids or token strings.

## D4 fertility scaling

**Choice.** `min_chars = round(60 * chars_per_token_lang / chars_per_token_english)`, searched on code points. The caller passes `chars_per_token`. There is no default.

**Alternative.** Infer fertility from the English item's model score, or from the same string being classified.

**Why.** Using the string under test to set its own threshold makes the threshold depend on how repetitive that string already is. Using an English accuracy to pick the threshold would violate "never selected by English outcome" in spirit, even though this is a detector rather than a sampler. A missing rate is an error, so a silent 4.0 cannot leak into D4.

## Who owns a split character

**Choice.** The token that contains the character's first byte owns the whole character. Continuation-only tokens get an empty span.

**Alternative.** Trust `tokenizer.decode` of each prefix, or edit `fix_mojibake`.

**Why.** Prefix decode drops or replaces an unfinished UTF-8 sequence, so the strings are not prefixes and the lengths lie. That is the bug. Empty spans mean `char_to_token_index` returns one token. On MGSM and on the Hindi MMLU questions this matched the tokenizer offset map's first covering token.

## Prompt source

**Choice.** 300 items from MMLU-ProX-Lite **test**, parallel in English, Hindi, and Telugu, stratified by category with largest remainders and a fixed seed (`20261030`). Within a category, ids are shuffled with that seed. Validation ids 0–69 stay out.

**Alternative.** Mix MGSM into the 300, or translate MGSM into Hindi, or pick items the model gets right in English.

**Why.** MGSM has no Hindi (confirmed from the upstream file list and the TSVs). Inventing Hindi would break the "matched" claim. The sampler's type, `EligibleItem`, has id, source, category, and the set of languages. It has no gold letter and no score. A test builds two pools that differ only in a fake `english_correct` field and checks that the ids match. Gold is written to a sidecar after selection.

## Romanized Telugu and Tenglish

**Choice.** Romanized prompts are `transliterate_telugu` of the Telugu prompt (ISO 15919, inherent vowel `a`). Unmapped Telugu-block characters raise. Tenglish is 100 empty slots, round-robin across categories, `prompt: null`.

**Alternative.** Ask a model to romanize or to code-mix.

**Why.** A model romanization is a generation, and it would confound the thing the study is trying to measure. The 300 romanized prompts contain no Telugu-block characters (checked after the build). Code-mixing by a script would not be the "hand-edited" subset the study asked for, so those slots stay empty.

## Instructions

**Choice.** One fixed instruction line per language. The romanized line is the transliteration of the Telugu line.

**Alternative.** Translate the instruction with a model, or omit instructions.

**Why.** The instruction has to be in the same script as the item or the drift measure will blame the prompt. Tying the romanized line to the Telugu line keeps them from diverging. The wording was written for this package and has not had a native-speaker review.

## Licenses

**Choice.** Record what the upstream files said on 2026-10-30 and ship `corpus/DATA_NOTICES.md`.

**Alternative.** Trust a catalog snippet that called MGSM CC-BY-SA.

**Why.** The MGSM `LICENSE` file in `google-research/url-nlp` is CC-BY-4.0. The MMLU-ProX-Lite card says MIT and the repo listing had no separate LICENSE file. Upstream MMLU-Pro is Apache-2.0. Both are named so a reviewer can see the conflict instead of a single invented label.

## Power

**Choice.** Equal-n two-proportion z test. Report uncorrected alpha 0.05 and Bonferroni across 3 contrasts (English vs Hindi, Telugu, and romanized Telugu). Planned n is 300. The 0.229 base rate is labeled as Liquid's Antidoom-mix figure, not a measurement here.

**Alternative.** Simulate a full study, or include Tenglish as a fourth contrast.

**Why.** There are no loop outcomes yet, so a simulation would be theater. Tenglish has no prompts, so it is not a planned contrast until the slots are filled.

## Pilot

**Choice.** 24 items × 4 languages = 96 generations. Same ids in each language, round-robin over categories. Planning cap for the full study is 2048 new tokens, pilot cap 512, budget 8 hours, health floor 15 tokens/second. Decision is `pending` without a rate, `go` / `replan` / `no-go` with a rate, and `basis=assumed` when the rate was typed in.

**Alternative.** Run the 96 generations in this environment, or treat 15 tokens/second as a published T4 speed.

**Why.** This environment is CPU-only and was told not to run a GPU job. 15 tokens/second is a floor for "the 4B model is actually on the GPU," not a benchmark. The hour estimate assumes a constant rate and a full 2048-token cap; both overstate the cost or the speed, and the caveats say so. Decoding in the adapter is greedy (`do_sample=False`). Liquid's pipeline used temperature 0.01. That difference is untested.

## Corpus path

**Choice.** Files live in `loop_lingua/corpus/`.

**Alternative.** `loop_lingua/data/`.

**Why.** The repo `.gitignore` ignores every `data/` directory. Putting the corpus there would have dropped it from the commit. `.gitignore` was not edited, because it belongs to paper-1.
