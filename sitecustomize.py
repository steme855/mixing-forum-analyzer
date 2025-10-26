"""Runtime helpers to provide graceful fallbacks for optional dependencies.

This module is imported automatically by Python (see :mod:`site`) which lets
us register minimal stub implementations for heavy optional packages when they
are not installed in the current environment. The goal is to keep the demo
usable and the test-suite green without enforcing massive installs such as
PyTorch on the evaluation CI.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _install_sentence_transformers_stub() -> None:
    try:
        importlib.import_module("sentence_transformers")
        return
    except ModuleNotFoundError:
        pass

    import re
    from collections import Counter
    from typing import Iterable, Sequence

    import numpy as np

    module = ModuleType("sentence_transformers")
    module.__doc__ = (
        "Lightweight stub for sentence_transformers used for local testing. "
        "Provides a minimal SentenceTransformer API that falls back to a "
        "bag-of-words embedding."
    )

    _word_pattern = re.compile(r"\w+", re.UNICODE)

    def _tokenize(text: str) -> list[str]:
        return _word_pattern.findall(text.lower())

    class SentenceTransformer:
        """Tiny replacement that mimics the encode API with bag-of-words."""

        def __init__(self, model_name: str | None = None, device: str = "cpu", **_: object) -> None:
            self.model_name = model_name or "stub-bow-model"
            self.device = device
            self._token_index: dict[str, int] = {}

        def _ensure_iterable(self, sentences: str | Sequence[str] | Iterable[str]) -> list[str]:
            if isinstance(sentences, str):
                return [sentences]
            return list(sentences)

        def encode(
            self,
            sentences: str | Sequence[str] | Iterable[str],
            *,
            normalize_embeddings: bool = False,
            **_: object,
        ) -> np.ndarray:
            rows = self._ensure_iterable(sentences)
            if not rows:
                return np.zeros((0, len(self._token_index)), dtype=np.float32)

            docs = [Counter(_tokenize(r)) for r in rows]
            for doc in docs:
                for token in doc:
                    if token not in self._token_index:
                        self._token_index[token] = len(self._token_index)

            if not self._token_index:
                return np.zeros((len(rows), 0), dtype=np.float32)

            matrix = np.zeros((len(rows), len(self._token_index)), dtype=np.float32)
            for row_idx, doc in enumerate(docs):
                for token, count in doc.items():
                    col_idx = self._token_index[token]
                    matrix[row_idx, col_idx] = float(count)

            if normalize_embeddings and matrix.size:
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                matrix = matrix / norms

            return matrix

    module.SentenceTransformer = SentenceTransformer
    module.__all__ = ["SentenceTransformer"]

    sys.modules["sentence_transformers"] = module


def _install_spacy_stub() -> None:
    try:
        importlib.import_module("spacy")
        return
    except ModuleNotFoundError:
        pass

    from typing import Iterable, Iterator

    module = ModuleType("spacy")
    module.__doc__ = "Ultra-light spaCy stub used for environments without the real package."
    module.__version__ = "0.0-stub"
    module.__file__ = __file__

    class Token:
        def __init__(self, text: str) -> None:
            self.text = text
            self.lemma_ = text.lower()
            self.pos_ = "X"

        def __repr__(self) -> str:
            return f"Token(text={self.text!r})"

    class Doc:
        def __init__(self, text: str) -> None:
            self.text = text
            self._tokens = [Token(tok) for tok in text.split()]

        def __iter__(self) -> Iterator[Token]:
            return iter(self._tokens)

        def __len__(self) -> int:
            return len(self._tokens)

    class Language:
        def __init__(self, name: str = "stub") -> None:
            self.name = name

        def __call__(self, text: str) -> Doc:
            return Doc(text)

        def pipe(self, texts: Iterable[str]) -> Iterator[Doc]:
            for text in texts:
                yield self(text)

    def load(name: str) -> Language:
        return Language(name)

    def blank(name: str) -> Language:
        return Language(name)

    util_module = ModuleType("spacy.util")
    util_module.is_package = lambda _: False  # type: ignore[attr-defined]

    module.Token = Token
    module.Doc = Doc
    module.Language = Language
    module.load = load
    module.blank = blank
    module.util = util_module
    module.__all__ = ["Token", "Doc", "Language", "load", "blank", "util"]

    sys.modules["spacy"] = module
    sys.modules["spacy.util"] = util_module


def _install_torch_stub() -> None:
    try:
        importlib.import_module("torch")
        return
    except ModuleNotFoundError:
        pass

    from typing import Any

    import numpy as np

    module = ModuleType("torch")
    module.__doc__ = (
        "Minimal PyTorch stub backed by NumPy for sum/isfinite/allclose calls. "
        "Only the tiny subset used in smoke tests is implemented."
    )

    class Tensor:
        """Very small tensor wrapper over numpy arrays."""

        def __init__(self, data: Any, dtype: Any | None = None) -> None:
            if isinstance(data, Tensor):
                array = data._array.copy()
            else:
                array = np.array(data, dtype=dtype)
            if dtype is not None and array.dtype != dtype:
                array = array.astype(dtype)
            self._array = array

        def sum(self) -> "Tensor":
            return Tensor(self._array.sum())

        def all(self) -> bool:
            return bool(self._array.all())

        def numpy(self) -> np.ndarray:
            return self._array.copy()

        def astype(self, dtype: Any) -> "Tensor":
            return Tensor(self._array.astype(dtype))

        @property
        def dtype(self) -> np.dtype:
            return self._array.dtype

        @property
        def shape(self) -> tuple[int, ...]:
            return self._array.shape

        def __array__(self, dtype: Any | None = None) -> np.ndarray:
            return np.asarray(self._array, dtype=dtype)

        def __iter__(self):
            return iter(self._array)

        def __repr__(self) -> str:
            return f"Tensor({self._array!r})"

    def tensor(data: Any, dtype: Any | None = None) -> Tensor:
        return Tensor(data, dtype=dtype)

    def isfinite(value: Any) -> Tensor:
        arr = np.asarray(value)
        return Tensor(np.isfinite(arr))

    def allclose(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        arr_a = np.asarray(a)
        arr_b = np.asarray(b)
        return bool(np.allclose(arr_a, arr_b, rtol=rtol, atol=atol))

    def float32(value: Any) -> np.float32:
        return np.float32(value)

    module.Tensor = Tensor
    module.tensor = tensor
    module.isfinite = isfinite
    module.allclose = allclose
    module.float32 = np.float32
    module.__all__ = ["Tensor", "tensor", "isfinite", "allclose", "float32"]

    sys.modules["torch"] = module


_install_sentence_transformers_stub()
_install_spacy_stub()
_install_torch_stub()
