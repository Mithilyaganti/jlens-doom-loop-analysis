# Exp 3 Analysis Plan (pre-registered)

1. Primary outcome: loop rate = fraction of prompts with find_inner_repetition hit.
2. Conditions (paired per prompt):
   - baseline
   - ablate_trigger (workspace band)
   - ablate_random (workspace band, norm-matched unit vectors)
   - ablate_control (frequency-matched content token directions)
   - ablate_trigger_sensory
   - ablate_trigger_motor
3. Statistical tests: McNemar on paired loop/no-loop baseline vs each intervention.
4. Effect size: Cohen's h on loop-rate proportions.
5. Secondary: mean generation length; small GSM8K-style eval accuracy (sanity that ablation is surgical).
6. Dose-response: ablate top-1 / top-3 / top-5 trigger directions (workspace).
