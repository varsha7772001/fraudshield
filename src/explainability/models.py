from dataclasses import dataclass


@dataclass
class FeatureImportance:
    feature_name: str
    importance: float
    rank: int