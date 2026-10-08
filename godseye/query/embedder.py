"""Text embeddings for semantic matching: the OWL-ViT text tower, i.e. the text side of the same CLIP space the detector's
reused image embeddings live in (no unrelated embedding model). Loaded lazily."""
from __future__ import annotations

import numpy as np


class OwlTextEmbedder:
    def __init__(self, model_name: str = "google/owlvit-base-patch32", device: str = "cpu"):
        self.model_name, self.device = model_name, device
        self._m = self._tok = None
        self._cache: dict[str, np.ndarray] = {}

    def _load(self):
        if self._m is None:
            import torch
            from transformers import AutoTokenizer, OwlViTModel
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._m = OwlViTModel.from_pretrained(self.model_name).to(self.device).eval()
            self._torch = torch

    def embed(self, texts: list[str]) -> np.ndarray:
        """-> (N, 512) float32, L2-normalised."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            self._load()
            with self._torch.no_grad():
                enc = self._tok([f"a photo of {t}" for t in todo], padding="max_length", max_length=16, truncation=True,
                                return_tensors="pt").to(self.device)
                f = self._m.get_text_features(**enc)
                f = (f if hasattr(f, "shape") else f.pooler_output).float()
                f = f / f.norm(dim=-1, keepdim=True)
            for t, v in zip(todo, f.cpu().numpy()):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts]).astype(np.float32)
