# What This Project Is About — A Plain-Language Guide

## The problem we study

Sometimes an AI language model gets stuck repeating itself. You ask it a math
question and it answers — and then keeps going: "...the answer is 4. the answer
is 4. the answer is 4..." hundreds of times, forever. Researchers call this a
**doom loop**. It matters because modern reasoning models work by thinking in
long steps before answering, and one loop kills the whole chain of thought.

Liquid AI recently showed you can fix this by *retraining* the model. Our
question is different: **can we see the failure building up inside the model,
and switch it off while it runs — with no retraining at all?**

## The tool we use

A language model is not a black box. Think of it as about 30 stages (called
layers) that a piece of text passes through. Researchers built a kind of
translator — we call it the lens — that takes the model's internal numbers at
each stage and tells you "which word is the model leaning toward right now."

The middle stages work like a scratchpad: the model holds the words it is
planning to use next. We call this region the **workspace**. Every word has a
"location" (a direction) in this workspace, like a seat in a theater.

## The three experiments

### 1. Do "trigger words" live in a special place?

When a loop starts, there is a first word of the repeated chunk — " given",
" So", " try". We call it the **trigger word**. We collect these from real
loops by running hundreds of problems, then compare where trigger words sit in
the workspace against three fair comparison groups: everyday words used just
as often, other transition words like "But" and "So", and random words.

**Expecting:** trigger words sit somewhat closer to the workspace's center of
action than the comparison words, especially in the middle stages. It may be a
weak effect — we measure carefully and report honestly either way.

### 2. Is there a warning sign before the loop starts?

A loop doesn't appear from nowhere — the model is already "deciding" to loop a
few steps before it happens. We record the workspace during the last ~5 steps
before each loop begins and ask: is it still considering many possible words
(healthy), or has it locked onto one word early (about to loop)? We compare
against carefully matched moments from normal responses — including moments
where the same trigger word appeared once and the model simply moved on.

**Expecting:** before a loop, the workspace narrows down and locks onto the
trigger word several steps early — like a seismograph before an earthquake.

### 3. If we remove that direction, does the loop stop? (The decisive test)

We take prompts we know loop, and re-run each one. At the moment the warning
sign appears, we surgically remove the trigger word's direction from the
workspace — like muting one instrument in a mix. We compare against: doing
nothing, removing random directions, removing unimportant words' directions,
removing at the wrong stages, and a brute-force method that simply blocks the
trigger word from being chosen (this one works by construction — it proves our
setup can detect loop prevention if it exists). We also verify the model still
solves problems correctly while the direction is removed.

**Expecting:** an honest coin flip. If loops drop, the headline is "a
training-free fix." If they don't, the headline is "the map can describe the
failure but cannot steer it" — still a real finding, because almost nobody has
tested this properly.

## Why two models

We run everything on **Qwen3.5-4B** (loops often — good for statistics) and
repeat the key analysis on **LFM2-2.6B** (already run; we re-crunch its
existing results with better statistics). Findings that hold in both are much
more trustworthy.

## What we get out of it, either way

A public table of trigger words, loop-rate measurements, the warning-sign
test, and the causal answer — packaged as an arXiv paper. Even a negative
outcome contributes, because this field is new and a properly-run negative
tells everyone something real.

## Tiny glossary

- **Token** — a word piece; models read and write tokens, roughly words.
- **Layer** — one of the ~30 processing stages inside the model.
- **Workspace** — the middle layers, where the model holds words it plans to
  use; the lens lets us read it.
- **Direction** — a word's "seat" in the workspace.
- **Ablation** — temporarily removing one direction while the model runs.
- **Loop rate** — the fraction of prompts that end in a doom loop.
