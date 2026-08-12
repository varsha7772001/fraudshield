from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray