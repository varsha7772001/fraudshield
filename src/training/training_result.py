from dataclasses import dataclass

from sklearn.base import BaseEstimator


@dataclass
class TrainingResult:
    model: BaseEstimator
    training_time: float