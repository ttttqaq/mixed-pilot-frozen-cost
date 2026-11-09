"""Scoring interface extracted from MTRouter (see UPSTREAM.json)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import numpy as np

def validate_model_names(
    checkpoint_model_names: list[str],
    expected_model_names: list[str] | None = None,
    *,
    source: str = "checkpoint",
) -> None:
    """Validate model names match between checkpoint and expected.

    This ensures that the model index mapping is consistent between
    training and inference, preventing silent routing errors.

    Args:
        checkpoint_model_names: Model names from checkpoint.
        expected_model_names: Expected model names (if None, skip validation).
        source: Source description for error message.

    Raises:
        ValueError: If model names don't match (including order).
    """
    if expected_model_names is None:
        return

    if checkpoint_model_names != expected_model_names:
        raise ValueError(
            f"Model names mismatch between {source} and expected.\n"
            f"  {source}: {checkpoint_model_names}\n"
            f"  expected: {expected_model_names}\n"
            "These must match (including order) for correct routing."
        )


class QFunction(ABC):
    """Abstract interface for Q-function estimators.

    A Q-function estimates the value Q(x, a) of selecting model a
    given state x (conversation history).

    Implementations can use:
    - Neural networks (MLP, Transformer)
    - Linear models
    - Simple heuristics for baselines
    """

    @abstractmethod
    def predict(
        self,
        history: str | list[dict],
        available_models: list[str],
    ) -> np.ndarray:
        """Predict Q-values for each available model.

        Args:
            history: Conversation history (text or messages).
            available_models: List of model names.

        Returns:
            Q-value array, shape=(len(available_models),)
        """
        ...

    def predict_with_features(
        self,
        features: dict[str, Any],
        available_models: list[str],
    ) -> np.ndarray:
        """Predict Q-values from pre-computed features.

        Override this for more efficient inference when features
        are already computed.

        Args:
            features: Pre-computed state features.
            available_models: List of model names.

        Returns:
            Q-value array.
        """
        # Default: extract history and use predict()
        history = features.get("history_text", "")
        return self.predict(history, available_models)

    def load(self, path: Path | str) -> None:
        """Load model weights from path.

        Args:
            path: Path to weights file.
        """
        raise NotImplementedError("Subclass must implement load()")

    def save(self, path: Path | str) -> None:
        """Save model weights to path.

        Args:
            path: Path to save weights.
        """
        raise NotImplementedError("Subclass must implement save()")


