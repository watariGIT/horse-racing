"""Model implementations for horse racing prediction."""

from typing import Any

from src.model_training.models.base import BaseModel
from src.model_training.models.ensemble import EnsembleModel
from src.model_training.models.lgbm_classifier import LGBMClassifierModel
from src.model_training.models.lgbm_ranker import LGBMRankerModel

# Factory registry for model creation by name
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "lgbm_classifier": LGBMClassifierModel,
    "lgbm_ranker": LGBMRankerModel,
    "ensemble": EnsembleModel,
}


def create_model(model_type: str, **kwargs: Any) -> BaseModel:
    """Factory function to create a model by name.

    Args:
        model_type: Model type key from MODEL_REGISTRY.
        **kwargs: Parameters passed to the model constructor.

    Returns:
        An instance of the requested model.

    Raises:
        ValueError: If model_type is not registered.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)


__all__ = [
    "BaseModel",
    "LGBMClassifierModel",
    "LGBMRankerModel",
    "EnsembleModel",
    "MODEL_REGISTRY",
    "create_model",
]
