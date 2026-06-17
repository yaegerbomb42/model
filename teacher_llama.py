"""
teacher_llama.py
----------------
Metal-accelerated local teacher using llama-cpp-python pointed at the
Ollama-cached gemma4:12b Q4_K_M GGUF blob.

Returns full next-token logit vectors for knowledge distillation —
something Ollama's REST API cannot do (it only returns sampled tokens).

Usage
-----
    from teacher_llama import LlamaCppTeacher
    teacher = LlamaCppTeacher()
    logits = teacher.logits(input_ids_tensor)  # (batch, seq, vocab)
"""

import os
import numpy as np
import torch

# gemma3:4b — lightweight local teacher (~2.5 GB, laptop-friendly)
_GEMMA3_4B_MODELFILE_CMD = "ollama show gemma3:4b --modelfile"

# gemma4:12b — fallback if 4b not available
_GEMMA4_12B_BLOB = (
    "/Users/yaeger/.ollama/models/blobs/"
    "sha256-1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606"
)

def _get_ollama_blob(model_tag: str) -> str | None:
    """
    Ask Ollama for the primary GGUF blob path for a given model tag.
    Returns None if the model is not found.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["ollama", "show", model_tag, "--modelfile"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.splitlines():
            if line.startswith("FROM ") and "blobs" in line:
                path = line.split(" ", 1)[1].strip()
                if os.path.exists(path):
                    return path
    except Exception:
        pass
    return None

def _find_largest_blob(blobs_dir: str) -> str:
    blobs = [
        os.path.join(blobs_dir, f)
        for f in os.listdir(blobs_dir)
        if not f.endswith(".json")
    ]
    return max(blobs, key=os.path.getsize)


def _resolve_model_path() -> tuple[str, str]:
    """
    Returns (gguf_path, model_tag) for the best available local teacher.
    Priority: gemma3:4b  →  gemma4:12b  →  largest blob in Ollama cache.
    """
    # 1. Try lightweight gemma3:4b first (laptop-friendly)
    path = _get_ollama_blob("gemma3:4b")
    if path:
        return path, "gemma3:4b"

    # 2. Fall back to gemma4:12b (heavier but higher quality)
    if os.path.exists(_GEMMA4_12B_BLOB):
        return _GEMMA4_12B_BLOB, "gemma4:12b"

    # 3. Last resort: largest file in the Ollama blobs dir
    blobs_dir = os.path.expanduser("~/.ollama/models/blobs")
    path = _find_largest_blob(blobs_dir)
    return path, "unknown-fallback"


class LlamaCppTeacher:
    """
    Thin wrapper around llama_cpp.Llama that exposes a logits() method
    compatible with the distillation loop in server.py.

    Parameters
    ----------
    model_path : str | None
        Path to the GGUF file. Defaults to the gemma4:12b Ollama blob.
    n_gpu_layers : int
        Number of transformer layers to offload to Metal (-1 = all).
    n_ctx : int
        Context window. 512 is plenty for our seq_len=16 distillation batches.
    vocab_size : int
        Must match the tokenizer vocab size (gemma4 = 262144).
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 512,
        vocab_size: int = 262144,
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed. Run:\n"
                "  CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python"
            )

        tag = "custom"
        if model_path is None:
            model_path, tag = _resolve_model_path()

        print(f"[LlamaCppTeacher] Using teacher: {tag}")
        print(f"[LlamaCppTeacher] Loading {model_path} "
              f"(n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}) ...")

        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=True,   # ← required to get logits for every position
            verbose=False,
        )
        self.vocab_size = vocab_size
        print("[LlamaCppTeacher] Ready ✅")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute teacher logits for a batch of token sequences.

        Parameters
        ----------
        input_ids : torch.Tensor  shape (batch, seq_len)
            Integer token ids on any device.

        Returns
        -------
        torch.Tensor  shape (batch, seq_len, vocab_size)  on CPU (fp32)
        """
        ids_cpu = input_ids.cpu().numpy().astype(np.int32)
        batch_size, seq_len = ids_cpu.shape
        all_logits = []

        for b in range(batch_size):
            tokens = ids_cpu[b].tolist()
            # Reset KV-cache and run eval
            self._llm.reset()
            self._llm.eval(tokens)
            # _llm.eval_logits is shape (seq_len, vocab_size) after logits_all=True
            raw = np.array(self._llm.eval_logits, dtype=np.float32)  # (seq, vocab)
            if raw.shape[0] < seq_len:
                # Pad if needed (shouldn't happen but be safe)
                pad = np.zeros((seq_len - raw.shape[0], raw.shape[1]), dtype=np.float32)
                raw = np.concatenate([raw, pad], axis=0)
            all_logits.append(raw[:seq_len])

        stacked = np.stack(all_logits, axis=0)  # (batch, seq, vocab)
        return torch.from_numpy(stacked)  # CPU fp32

    def __call__(self, input_ids: torch.Tensor) -> "FakeOutput":
        """Mimic HuggingFace model(input_ids).logits interface."""
        return FakeOutput(self.logits(input_ids))


class FakeOutput:
    """Tiny shim so LlamaCppTeacher can be used as a drop-in for HF models."""
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
