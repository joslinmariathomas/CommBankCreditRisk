DATA_DIRECTORY = "./data/"

CREDIT_BUREAU_COLS_LIST = [
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]
SOCIAL_CIRCLE_COLS_LIST = [
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
]
EXTRA_COLUMNS_TO_ADD_AS_MISSING_FLAG = ["EXT_SOURCE_2"]
COLUMNS_TO_REMOVE = ["CODE_GENDER", "DAYS_EMPLOYED"]


class ImputationConfig:
    """Configuration class for common imputation setups."""

    # Features to impute with zero (common for financial data)
    FEATURES_TO_IMPUTE_ZERO_WHEN_MISSING = (
        CREDIT_BUREAU_COLS_LIST + SOCIAL_CIRCLE_COLS_LIST
    )

    # Imputation strategies for different features
    VALUE_TO_IMPUTE_DICT_BY_FEATURE = {
        "CNT_FAM_MEMBERS": "mode",
        "EXT_SOURCE_3": "mode",
        "NAME_TYPE_SUITE": "Unaccompanied",
        "DAYS_LAST_PHONE_CHANGE": "median",
        "AMT_GOODS_PRICE": "median",
        "AMT_ANNUITY": "median",
        "DAYS_EMPLOYED": "group_median",
    }

    # Grouping columns for group-based imputation
    GROUPING_COLUMNS = {
        "DAYS_EMPLOYED": ["NAME_EDUCATION_TYPE", "CODE_GENDER"],
    }
