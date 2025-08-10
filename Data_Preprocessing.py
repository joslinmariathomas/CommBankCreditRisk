import os

import pandas as pd

from Cleanup_and_impute_missing_values import Imputation, create_imputation_pipeline
from Feature_Engineering import create_new_sensible_features
from config import DATA_DIRECTORY


if __name__ == "__main__":
    application_train_data = pd.read_csv(os.path.join(DATA_DIRECTORY, 'application_train.csv'))
    imputer = create_imputation_pipeline()
    X_train_imputed = imputer.fit_transform(application_train_data)
    X_train_final = create_new_sensible_features(X_train_imputed)

