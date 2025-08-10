from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

from config import ImputationConfig


class Imputation(BaseEstimator, TransformerMixin):
    """
   Imputation class for credit risk datasets.
    - Multiple imputation strategies (zero, median, mode, group-based)
    - Missing value flagging for high-missingness features
    """

    def __init__(self,
                 features_to_impute_zero: Optional[List[str]] = None,
                 imputation_strategies: Optional[Dict[str, str]] = None,
                 grouping_columns: Optional[Dict[str, List[str]]] = None,
                 missing_flag_threshold: float = 0.3):
        """
        Initialize the Imputation class.
        """
        self.features_to_impute_zero = features_to_impute_zero or []
        self.imputation_strategies = imputation_strategies or {}
        self.grouping_columns = grouping_columns or {}
        self.missing_flag_threshold = missing_flag_threshold

        # Store fitted parameters
        self.imputation_values_ = {}
        self.features_to_flag_ = []
        self.group_medians_ = {}
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Imputation':
        """
        Fit the imputation parameters on training data.
        """
        X = X.copy()

        self.features_to_flag_ = self._identify_features_to_flag_missing(X)

        for feature, strategy in self.imputation_strategies.items():
            if feature not in X.columns:
                warnings.warn(f"Feature '{feature}' not found in data. Skipping.")
                continue

            if strategy == "median":
                self.imputation_values_[feature] = X[feature].median()
            elif strategy == "mode":
                mode_val = X[feature].mode()
                if len(mode_val) > 0:
                    self.imputation_values_[feature] = mode_val.iloc[0]
                else:
                    self.imputation_values_[feature] = X[feature].iloc[0]  # fallback
            elif strategy == "group_median":
                if feature in self.grouping_columns:
                    group_cols = self.grouping_columns[feature]
                    if all(col in X.columns for col in group_cols):
                        self.group_medians_[feature] = (
                            X.groupby(group_cols)[feature]
                            .median()
                            .to_dict()
                        )
                    else:
                        warnings.warn(f"Grouping columns for '{feature}' not found. Using overall median.")
                        self.imputation_values_[feature] = X[feature].median()
            elif strategy not in ["mean_across_features", "zero"]:
                self.imputation_values_[feature] = strategy

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation to data using fitted parameters.
        """
        if not self.is_fitted_:
            raise ValueError("Imputation has not been fitted yet. Call fit() first.")

        X_transformed = X.copy()

        # Flag missing values for high-missingness features
        for feature in self.features_to_flag_:
            if feature in X_transformed.columns:
                X_transformed[f'FLAG_MISS_{feature}'] = X_transformed[feature].isnull().astype(int)

        # Apply zero imputation
        for feature in self.features_to_impute_zero:
            if feature in X_transformed.columns:
                X_transformed[feature] = X_transformed[feature].fillna(0)


        for feature, strategy in self.imputation_strategies.items():
            if feature not in X_transformed.columns:
                continue

            if strategy in ["median", "mode"] and feature in self.imputation_values_:
                X_transformed[feature] = X_transformed[feature].fillna(self.imputation_values_[feature])

            elif strategy == "group_median" and feature in self.group_medians_:
                X_transformed = self._apply_group_median_imputation(X_transformed, feature)

            elif strategy == "mean_across_features":
                if feature in self.grouping_columns:
                    feature_list = self.grouping_columns[feature]
                    X_transformed = self._apply_mean_across_features_imputation(
                        X_transformed, feature, feature_list
                    )

            elif strategy not in ["zero", "median", "mode", "group_median", "mean_across_features"]:
                X_transformed[feature] = X_transformed[feature].fillna(strategy)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit imputation parameters and transform data in one step.
        """
        return self.fit(X, y).transform(X)

    def _identify_features_to_flag_missing(self, X: pd.DataFrame) -> List[str]:
        """
        Identify features with missing values above threshold.
        """
        missing_stats = X.isnull().mean()
        high_missing_features = missing_stats[missing_stats >= self.missing_flag_threshold].index.tolist()
        return high_missing_features

    def _apply_group_median_imputation(self, X: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Apply group-based median imputation using stored group medians.
        """
        if feature not in self.group_medians_:
            return X

        group_cols = self.grouping_columns[feature]
        X_copy = X.copy()

        X_copy[f'{feature}_IMPUTED'] = X_copy[feature].copy()

        for group_key, median_val in self.group_medians_[feature].items():
            if isinstance(group_key, tuple):
                mask = True
                for i, col in enumerate(group_cols):
                    if col in X_copy.columns:
                        mask = mask & (X_copy[col] == group_key[i])
            else:
                mask = X_copy[group_cols[0]] == group_key

            missing_mask = X_copy[feature].isnull()
            X_copy.loc[mask & missing_mask, f'{feature}_IMPUTED'] = median_val

        still_missing = X_copy[f'{feature}_IMPUTED'].isnull()
        if still_missing.any() and feature in self.imputation_values_:
            X_copy.loc[still_missing, f'{feature}_IMPUTED'] = self.imputation_values_[feature]

        return X_copy

    def _apply_mean_across_features_imputation(self, X: pd.DataFrame,
                                               imputed_feature: str,
                                               features_list: List[str]) -> pd.DataFrame:
        """
        Apply mean across features imputation.
        """
        X_copy = X.copy()

        # Calculate mean across specified features
        available_features = [f for f in features_list if f in X_copy.columns]
        if available_features:
            X_copy[imputed_feature] = X_copy[available_features].mean(axis=1, skipna=True)

            # Create imputed versions of original features
            for col in available_features:
                X_copy[f'{col}_IMPUTED'] = X_copy[col].fillna(X_copy[imputed_feature])

        return X_copy

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted imputation parameters.
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}

        return {
            "is_fitted": self.is_fitted_,
            "features_flagged_for_missing": self.features_to_flag_,
            "imputation_values": self.imputation_values_,
            "group_imputation_features": list(self.group_medians_.keys()),
            "zero_imputation_features": self.features_to_impute_zero,
            "missing_flag_threshold": self.missing_flag_threshold
        }


def create_imputation_pipeline():
    """Create a configured imputation pipeline for credit risk data."""
    config = ImputationConfig()

    imputer = Imputation(
        features_to_impute_zero=config.FEATURES_TO_IMPUTE_ZERO_WHEN_MISSING,
        imputation_strategies=config.VALUE_TO_IMPUTE_DICT_BY_FEATURE,
        grouping_columns=config.GROUPING_COLUMNS,
        missing_flag_threshold=0.3
    )

    return imputer

