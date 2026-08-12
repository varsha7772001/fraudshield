from dataclasses import dataclass


@dataclass
class FeatureDecision:
    feature_name: str
    keep: bool
    reason: str


@dataclass
class FeatureSelectionReport:
    decisions: list[FeatureDecision]

    def columns_to_drop(self) -> list[str]:
        return [
            decision.feature_name
            for decision in self.decisions
            if not decision.keep
        ]

    def columns_to_keep(self) -> list[str]:
        return [
            decision.feature_name
            for decision in self.decisions
            if decision.keep
        ]