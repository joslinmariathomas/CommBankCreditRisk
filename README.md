# Credit Risk Data Preprocessing Pipeline
#### Data cleaning and feature engineering pipeline for credit application data, for credit risk modeling workflow.

## Overview
- The dataset used is from [Kaggle Home Credit Risk ](https://www.kaggle.com/competitions/home-credit-default-risk/rules)
- Scope: application_train dataset (307,511 loan applications, 122 features)
- Business Context: Consumer credit risk assessment for loan approval decisions
- Objective: Develop production-ready data preprocessing pipeline for downstream risk modeling

## Key Features
### Real-World Data Challenges Addressed
- Systematic missing data patterns in housing-related features (70% missingness)
- Data quality issues requiring domain expertise for resolution
- Ethical AI considerations
Feature Engineering Strategy

### Feature Engineering Strategy
#### Financial Health Indicators
- debt_to_income_ratio: Core creditworthiness metric
- payment_to_income_ratio: Affordability assessment
- residual_income: Disposable income after credit payments
- credit_per_family_member: Household debt burden

#### Life Stage Risk Factors
- career_stage: Age-based risk categorization
- employment_stability: Employment duration relative to age
- years_to_retirement: Income stability timeline
- young_high_credit: Early-career high-risk flag

#### Household Composition Risks
- children_ratio: Family dependency burden
- single_parent: Single-income household risk
- low_income_large_family: Compound vulnerability indicator
