# Agent Mod Requests

## Scaffold Guardrails for R2-Style Reasoner
1. **Verify-on-Edit Nudge**: Run corresponding test scripts immediately after any single edit to prevent compounding errors.
2. **State Compression**: Summarize intermediate context states every 20 steps to prevent tokenizer overflow and context bloat.
3. **Loop Detection**: Intercept execution if the exact same actions are repeated 3 times without a change in logs.
