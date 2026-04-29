from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .phoneme_map import SILENCE, OBSTRUENT, SONORANT  # noqa: F401

_DEFAULT_MODEL_PATH = Path(__file__).parent / "model.joblib"

_CACHED_CLASSIFIER: GateClassifier | None = None


class GateClassifier:
    """Wraps the trained gate model for inference."""

    def __init__(self, model_path: Path = _DEFAULT_MODEL_PATH) -> None:
        payload = joblib.load(model_path)
        self._model = payload["model"]
        self.features: list[str] = payload["features"]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return an array of 'silence'/'obstruent'/'sonorant' for each row."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise KeyError(f"Gate classifier expects columns not found in DataFrame: {missing}")
        X = df[self.features].to_numpy(dtype=float)
        return self._model.predict(X)


def load_classifier(model_path: Path = _DEFAULT_MODEL_PATH) -> GateClassifier:
    """Load the gate classifier, caching it for the lifetime of the process."""
    global _CACHED_CLASSIFIER
    if _CACHED_CLASSIFIER is None:
        _CACHED_CLASSIFIER = GateClassifier(model_path)
    return _CACHED_CLASSIFIER


def apply_gate(df: pd.DataFrame, segment_col: str = "segment_class") -> pd.DataFrame:
    """Add a segment_class column ('silence'/'obstruent'/'sonorant') to df."""
    clf = load_classifier()
    out = df.copy()
    out[segment_col] = clf.predict(df)
    return out
