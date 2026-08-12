from dataclasses import dataclass
import pandas as pd


@dataclass
class FeatureMetadata:
    name: str
    role: str
    business_group: str
    storage_type: str
    missing_percentage: float


@dataclass
class FeatureCatalog:
    target: str
    identifiers: list[str]
    features: list[FeatureMetadata]


class FeatureCatalogBuilder:

    GROUP_PREFIXES = {
        "Transaction": "Transaction",
        "Product": "Product",
        "card": "Card",
        "addr": "Address",
        "dist": "Distance",
        "P_email": "Email",
        "R_email": "Email",
        "C": "Count",
        "D": "Time",
        "M": "Matching",
        "V": "Engineered",
        "id_": "Identity",
        "Device": "Device",
    }

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str = "isFraud",
        identifiers: list[str] | None = None,
    ) -> None:

        self.dataframe = dataframe
        self.target = target
        self.identifiers = identifiers or ["TransactionID"]

    def build(self) -> FeatureCatalog:

        features = []

        missing_percentages = (
            self.dataframe.isnull().mean() * 100
        )

        for column in self.dataframe.columns:

            role = self._get_role(column)

            business_group = self._get_business_group(column)

            storage_type = str(self.dataframe[column].dtype)

            missing_percentage = float(round(
                missing_percentages[column],
                2,
            ))

            metadata = FeatureMetadata(
                name=column,
                role=role,
                business_group=business_group,
                storage_type=storage_type,
                missing_percentage=missing_percentage,
            )

            features.append(metadata)

        return FeatureCatalog(
            target=self.target,
            identifiers=self.identifiers,
            features=features,
        )

    def _get_role(self, column: str) -> str:

        if column == self.target:
            return "Target"

        if column in self.identifiers:
            return "Identifier"

        return "Feature"

    def _get_business_group(self, column: str) -> str:

        for prefix, group in self.GROUP_PREFIXES.items():

            if column.startswith(prefix):
                return group

        return "Other"