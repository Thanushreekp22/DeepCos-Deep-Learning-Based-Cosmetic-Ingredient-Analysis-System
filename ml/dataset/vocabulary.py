"""
Ingredient vocabulary: ingredient name <-> integer token (Module 2).

The LSTM consumes integer token ids, exactly like the DeepCos design:

    Water            -> 12
    Glycerin         -> 45
    Niacinamide      -> 81
    Hyaluronic Acid  -> 103
    Salicylic Acid   -> 57

Index 0 is reserved for padding and index 1 for unknown ingredients, so the
vocabulary keeps a stable layout between training and inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ml import config
from ml.knowledge_base import normalize_ingredient_name


class Vocabulary:
    """Bidirectional ingredient-name <-> token-index mapping."""

    def __init__(self, token2idx: dict[str, int], idx2token: list[str]) -> None:
        self.token2idx: dict[str, int] = dict(token2idx)
        self.idx2token: list[str] = list(idx2token)
        if self.idx2token[:2] != [config.PAD_TOKEN, config.UNK_TOKEN]:
            raise ValueError("Vocabulary must start with <pad> and <unk>")

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, names: Iterable[str], extra_aliases: dict[str, str] | None = None) -> "Vocabulary":
        """Build a vocabulary from canonical ingredient names."""
        token2idx: dict[str, int] = {config.PAD_TOKEN: config.PAD_INDEX, config.UNK_TOKEN: config.UNK_INDEX}
        idx2token: list[str] = [config.PAD_TOKEN, config.UNK_TOKEN]
        for name in names:
            key = normalize_ingredient_name(name)
            if not key or key in token2idx:
                continue
            token2idx[key] = len(idx2token)
            idx2token.append(name)
        if extra_aliases:
            # alias -> canonical key; aliases never get their own index
            for alias, canonical in extra_aliases.items():
                alias_key = normalize_ingredient_name(alias)
                canonical_key = normalize_ingredient_name(canonical)
                if alias_key and canonical_key in token2idx:
                    token2idx[alias_key] = token2idx[canonical_key]
        vocab = cls(token2idx, idx2token)
        return vocab

    # -- encoding ----------------------------------------------------------
    def token_index(self, name: str) -> int:
        return self.token2idx.get(normalize_ingredient_name(name), config.UNK_INDEX)

    def encode(self, names: Iterable[str], seq_len: int | None = None) -> np.ndarray:
        """Encode ingredient names into a padded int32 token sequence."""
        seq_len = int(seq_len or config.SEQ_LEN)
        indices = [self.token_index(name) for name in names][:seq_len]
        if len(indices) < seq_len:
            indices.extend([config.PAD_INDEX] * (seq_len - len(indices)))
        return np.asarray(indices, dtype="int32")

    def encode_batch(
        self, sequences: Sequence[Sequence[str]], seq_len: int | None = None
    ) -> np.ndarray:
        return np.stack([self.encode(seq, seq_len) for seq in sequences]).astype("int32")

    def decode(self, indices: Iterable[int]) -> list[str]:
        out: list[str] = []
        for idx in indices:
            idx = int(idx)
            if idx == config.PAD_INDEX:
                continue
            if 0 <= idx < len(self.idx2token):
                out.append(self.idx2token[idx])
            else:
                out.append(config.UNK_TOKEN)
        return out

    def coverage(self, names: Iterable[str]) -> dict:
        names = list(names)
        known = sum(1 for name in names if self.token_index(name) != config.UNK_INDEX)
        return {
            "total": len(names),
            "known": known,
            "unknown": len(names) - known,
            "coverage": round(known / len(names), 4) if names else 0.0,
        }

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": "1.0.0",
            "seq_len": config.SEQ_LEN,
            "pad_index": config.PAD_INDEX,
            "unk_index": config.UNK_INDEX,
            "size": len(self.idx2token),
            "idx2token": self.idx2token,
            "token2idx": self.token2idx,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Vocabulary":
        idx2token = payload["idx2token"]
        token2idx = payload.get("token2idx")
        if not token2idx:
            token2idx = {token: i for i, token in enumerate(idx2token)}
        return cls(token2idx, idx2token)

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path or config.INGREDIENT_VOCAB_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Vocabulary":
        path = Path(path or config.INGREDIENT_VOCAB_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Vocabulary not found at {path}. Run 'python -m ml.train_ingredient_model' first."
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # -- dunder ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.idx2token)

    def __contains__(self, name: object) -> bool:
        return normalize_ingredient_name(str(name)) in self.token2idx

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Vocabulary(size={len(self)}, seq_len={config.SEQ_LEN})"
