"""Runtime-Stubs für optionale Abhängigkeiten.

Wird von Python automatisch beim Start importiert (via :mod:`site`) wenn
dieses Verzeichnis im sys.path liegt. Registriert leichtgewichtige Stub-
Implementierungen für schwere Pakete (torch, sentence_transformers, spaCy),
sodass Demo und Test-Suite ohne vollständige ML-Installation funktionieren.

Hinweis: In pytest wird diese Datei über conftest.py explizit importiert,
da sitecustomize.py nur im sys.path-Root automatisch geladen wird.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _install_sentence_transformers_stub() -> None:
    """Installiert einen BoW-basierten Stub wenn sentence_transformers fehlt."""
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
        "Leichtgewichtiger sentence_transformers-Stub für lokale Tests. "
        "Bildet die encode()-API nach mit Bag-of-Words-Vektoren."
    )

    _word_pattern = re.compile(r"\w+", re.UNICODE)

    def _tokenize(text: str) -> list[str]:
        return _word_pattern.findall(text.lower())

    class SentenceTransformer:
        """Minimaler Ersatz – mimikt encode() mit BoW (sentence-transformers >=3.x API)."""

        def __init__(
            self,
            model_name: str | None = None,
            device: str = "cpu",
            **_: object,
        ) -> None:
            self.model_name = model_name or "stub-bow-model"
            self.device = device
            self._token_index: dict[str, int] = {}

        def _ensure_iterable(
            self, sentences: str | Sequence[str] | Iterable[str]
        ) -> list[str]:
            if isinstance(sentences, str):
                return [sentences]
            return list(sentences)

        def encode(
            self,
            sentences: str | Sequence[str] | Iterable[str],
            *,
            normalize_embeddings: bool = False,
            show_progress_bar: bool = False,
            # convert_to_numpy wurde in >=3.x entfernt – ignorieren via **kwargs
            **_kwargs: object,
        ) -> np.ndarray:
            """Gibt ein np.ndarray zurück (kompatibel mit sentence-transformers >=3.x)."""
            rows = self._ensure_iterable(sentences)
            if not rows:
                return np.zeros((0, max(len(self._token_index), 1)), dtype=np.float32)

            docs = [Counter(_tokenize(r)) for r in rows]
            for doc in docs:
                for token in doc:
                    if token not in self._token_index:
                        self._token_index[token] = len(self._token_index)

            if not self._token_index:
                return np.zeros((len(rows), 1), dtype=np.float32)

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

    module.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    module.__all__ = ["SentenceTransformer"]
    sys.modules["sentence_transformers"] = module


def _install_spacy_stub() -> None:
    """Installiert einen minimalen spaCy-Stub wenn spaCy fehlt."""
    try:
        importlib.import_module("spacy")
        return
    except ModuleNotFoundError:
        pass

    from typing import Iterable, Iterator

    module = ModuleType("spacy")
    module.__doc__ = "Ultra-leichter spaCy-Stub für Umgebungen ohne echtes spaCy."
    module.__version__ = "0.0-stub"  # type: ignore[attr-defined]
    module.__file__ = __file__  # type: ignore[attr-defined]

    class Token:
        is_stop: bool = False
        is_punct: bool = False
        like_num: bool = False

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

    module.Token = Token  # type: ignore[attr-defined]
    module.Doc = Doc  # type: ignore[attr-defined]
    module.Language = Language  # type: ignore[attr-defined]
    module.load = load  # type: ignore[attr-defined]
    module.blank = blank  # type: ignore[attr-defined]
    module.util = util_module  # type: ignore[attr-defined]
    module.__all__ = ["Token", "Doc", "Language", "load", "blank", "util"]  # type: ignore[attr-defined]

    sys.modules["spacy"] = module
    sys.modules["spacy.util"] = util_module


def _install_torch_stub() -> None:
    """Installiert einen minimalen PyTorch-Stub wenn torch fehlt."""
    try:
        importlib.import_module("torch")
        return
    except ModuleNotFoundError:
        pass

    from typing import Any

    import numpy as np

    module = ModuleType("torch")
    module.__doc__ = (
        "Minimaler PyTorch-Stub auf NumPy-Basis für Smoke-Tests. "
        "Nur das kleine Subset für sum/isfinite/allclose ist implementiert."
    )

    class Tensor:
        """Kleiner Tensor-Wrapper über np.ndarray."""

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

    module.Tensor = Tensor  # type: ignore[attr-defined]
    module.tensor = tensor  # type: ignore[attr-defined]
    module.isfinite = isfinite  # type: ignore[attr-defined]
    module.allclose = allclose  # type: ignore[attr-defined]
    module.float32 = np.float32  # type: ignore[attr-defined]
    module.__all__ = ["Tensor", "tensor", "isfinite", "allclose", "float32"]  # type: ignore[attr-defined]
    sys.modules["torch"] = module


_install_sentence_transformers_stub()
_install_spacy_stub()
_install_torch_stub()
