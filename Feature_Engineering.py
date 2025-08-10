import pandas as pd

from config import COLUMNS_TO_REMOVE


def create_new_sensible_features(df: pd.DataFrame) -> pd.DataFrame:
    create_income_credit_features(
        df=df,
    )
    create_age_employment_features(df=df)
    family_income_debt_features(df=df)
    df.drop(columns=COLUMNS_TO_REMOVE, inplace=True)
    return df


def create_income_credit_features(df: pd.DataFrame):
    """
    Create commonly used debt and income metrics.
    """

    df["debt_to_income"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["payment_to_income_ratio"] = (df["AMT_ANNUITY"] * 12) / df["AMT_INCOME_TOTAL"]
    df["residual_income"] = df["AMT_INCOME_TOTAL"] - (df["AMT_ANNUITY"] * 12)
    return df


def create_age_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Employment stability indicators for risk profiling."""
    df["career_stage"] = pd.cut(
        df["age_years"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["Early", "Establishing", "Peak", "Senior", "Retirement"],
    )
    df["employment_stability"] = df["employment_years"] / df["age_years"]

    df["years_to_retirement"] = 65 - df["age_years"]
    df["young_high_credit"] = (
        (df["age_years"] < 30) & (df["debt_to_income"] > 5)
    ).astype(int)
    return df


def family_income_debt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Household  and socoeconomic risk indicators."""
    df["children_ratio"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"]
    df["income_per_family_member"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["large_family"] = (df["CNT_FAM_MEMBERS"] >= 5).astype(int)
    df["single_parent"] = (
        (df["CNT_CHILDREN"] > 0) & (df["NAME_FAMILY_STATUS"] == "Single / not married")
    ).astype(int)
    df["low_income_large_family"] = (
        (df["AMT_INCOME_TOTAL"] < df["AMT_INCOME_TOTAL"].quantile(0.25))
        & (df["CNT_FAM_MEMBERS"] >= 4)
    ).astype(int)
    df["credit_per_family_member"] = df["AMT_CREDIT"] / df["CNT_FAM_MEMBERS"]
    return df
