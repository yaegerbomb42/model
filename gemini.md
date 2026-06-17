# Global Agent Instructions & Scaffold Rules

These rules apply to the AI coding agent at all times during codebase modifications and executions.

---

## 1. The "Verify-on-Edit" Nudge
* Immediately after editing any code file, the agent must run the corresponding test script (e.g., `test_agent.py` or `train_rl.py`) to verify changes.
* Do not edit multiple files without verification.
* If a test fails, feed the raw terminal error log back into the context as a distinct `[SYSTEM: CRITIQUE]` block for self-correction.

## 2. State Compression (Summary Truncation)
* Every 20 steps, the agent must summarize the progress made so far.
* The agent will clear the middle of the context window, passing only the summary of past attempts and the current codebase state to prevent token bloat.

## 3. Loop Detection Heuristics
* The agent must monitor its own actions.
* If the agent executes the exact same terminal command or modifies the same lines of code 3 times in a row without a change in the error output, it must intercept its own loop and output:
  `[SYSTEM: You are stuck in a loop. Step back, read your previous 5 attempts, and try a completely different logical approach.]`
