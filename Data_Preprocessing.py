import os
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from Cleanup_and_impute_missing_values import Imputation, create_imputation_pipeline
from Feature_Engineering import create_new_sensible_features
from config import DATA_DIRECTORY


def scale_the_dataset(X_train: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """Transforms the dataset to"""
    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numerical_cols = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_cols,
            ),
            ("num", StandardScaler(), numerical_cols),
        ]
    )
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    return X_train_processed, preprocessor


def oversample_minority_class(X_train, y_train):
    X_train_smote, y_train_smote = SMOTE(random_state=1234).fit_resample(
        X_train, y_train
    )
    smote_value_counts = y_train_smote.value_counts()
    print(
        "Fraudulent transactions are %.2f%% of the test set."
        % (smote_value_counts[0] * 100 / len(y_train_smote))
    )
    return X_train_smote, y_train_smote


if __name__ == "__main__":
    application_train_data = pd.read_csv(
        os.path.join(DATA_DIRECTORY, "application_train.csv")
    )
    imputer = create_imputation_pipeline()
    y_train = application_train_data["TARGET"]
    X_train = application_train_data.drop(columns=["TARGET"])
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_feature_engineered = create_new_sensible_features(X_train_imputed)
    X_train_scaled, column_transformer = scale_the_dataset(X_train_feature_engineered)
    X_train_smote, y_train_smote = oversample_minority_class(X_train_scaled, y_train)
