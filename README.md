# Credit Risk Data Preprocessing Pipeline
#### Sample data cleaning and feature engineering pipeline for credit application data, for credit risk modeling workflow.

## Overview
- The dataset used is from [Kaggle Home Credit Risk ](https://www.kaggle.com/competitions/home-credit-default-risk/rules)
- Scope: application_train dataset (307,511 loan applications, 122 features)
- Business Context: Consumer credit risk assessment for loan approval decisions
- Objective: Develop data preprocessing pipeline for downstream risk modeling
- Note: Dataset is not added to the github repository.
## Key Features
### Imputation strategy
- Housing-Related Features (>70% Missing) : Created binary flags for features with >30% missing values,
  as the absence itself carries predictive signal. 
- Credit Bureau Enquiries (`AMT_REQ_CREDIT_BUREAU_*`): Missing values logically represent zero enquiries,
  indicating lower risk. Imputed with 0 based on business logic.
- Social Circle Variables (`OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE`):
  Missing values indicate no observations in social circle, imputed with 0 following 
  the same risk-based reasoning.
- Standard Imputation Approach:
  `CNT_FAM_MEMBERS`: Median imputation
  `NAME_TYPE_SUITE`: Mode imputation (NaN assumed as "Unaccompanied")
  `DAYS_LAST_PHONE_CHANGE`: Median imputation
- Sophisticated Imputation for External Sources:
  For `EXT_SOURCE_3`, created composite score averaging available external sources rather than 
  using simple statistical measures, preserving the predictive relationship between
  external scoring systems.
- Ethical AI considerations : Removed `CODE_GENDER` for fair modelling irrespective of gender

### Feature Engineering Strategy
#### Financial Health Indicators
- debt_to_income_ratio(`AMT_CREDIT`/`AMT_INCOME_TOTAL`): Core creditworthiness metric
- payment_to_income_ratio(`AMT_ANNUITY`* 12 /`AMT_INCOME_TOTAL`): Affordability assessment
- residual_income(`AMT_INCOME_TOTAL` -`AMT_ANNUITY`* 12) : Disposable income after credit payments

#### Life Stage Risk Factors
- career_stage: Age-based risk categorization
- employment_stability(`employment_years`/`age_years`): Employment duration relative to age
- years_to_retirement(65 -`age_years`) : Income stability timeline
- young_high_credit(`age_years` < 30 & `debt_to_income` > 5): Early-career high-risk flag

#### Household Composition Risks
- children_ratio(`CNT_CHILDREN`/`CNT_FAM_MEMBERS`): Family dependency burden
- single_parent(`CNT_CHILDREN`>0 & `NAME_FAMILY_STATUS` == `Single / not married`): Single-income household risk
- low_income_large_family(`AMT_INCOME_TOTAL`<25th percentile income & `CNT_FAM_MEMBERS`>=4): 
  Compound vulnerability indicator
- credit_per_family_member(`AMT_CREDIT`/`CNT_FAM_MEMBERS`): Household debt burden
