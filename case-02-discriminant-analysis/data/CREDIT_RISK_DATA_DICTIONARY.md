# Credit Risk Assessment Data Dictionary  
LendSmart - Loan Application Dataset  
*Version: 2025-11-03*

---

## Dataset Overview
- **File:** credit_risk_data.csv  
- **Observations:** 2,500 loan applications  
- **Period:** 2022-01-01 to 2024-12-29 (3 years)  
- **Good Loan Rate:** 73.44% (1,836 loans)  
- **Default Rate:** 26.56% (664 defaults)

---

# 1. Identification Variables

| Variable | Type | Description | Values |
|---------|------|-------------|--------|
| application_id | String | Unique application identifier | APP_001 to APP_2500 |
| application_date | Date | Loan application submission date | YYYY-MM-DD |
| loan_amount | Float | Requested loan amount | \$5,000 – \$500,000 |

---

# 2. Financial Indicators

## Income & Employment

| Variable | Type | Scale | Description |
|----------|------|--------|-------------|
| annual_income | Float | \$15,000–\$149,930 | Annual gross income |
| employment_years | Float | 0–19.3 | Years in current employment |
| job_stability_score | Float | 0.011–0.999 | Higher = more stable employment |

---

## Credit History

| Variable | Type | Scale | Description |
|----------|------|--------|-------------|
| credit_score | Integer | 334–850 | FICO credit score |
| credit_utilization | Float | 0.004–0.998 | Ratio of credit used to credit available |
| payment_history_score | Float | 0.029–1.000 | Payment history quality |
| open_credit_lines | Integer | 0–11 | Active credit accounts |

---

## Debt & Assets

| Variable | Type | Scale | Description |
|---------|------|--------|-------------|
| debt_to_income_ratio | Float | 0.009–0.979 | Total debt payments / income |
| savings_ratio | Float | 0.000–0.893 | Savings / annual income |
| asset_value | Float | \$551–\$1,000,000 | Total value of assets |

---

# 3. Demographic Variables

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| age | Integer | Applicant age | 18–75 |
| education_level | Categorical | Highest completed education | HS (444), Associates (596), Bachelors (834), Masters (442), Doctorate (184) |
| marital_status | Categorical | Marital status | Single (580), Married (1311), Divorced (439), Widowed (170) |
| residential_stability | Float | Years at current address | 0.0–16.4 |

---

# 4. Outcome Variable

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| loan_status | Binary | Loan performance outcome | 0 = Good, 1 = Default (90+ days past due) |

---

# 5. Data Quality Notes

- **Missing Data:** None (100% complete)  
- **Total Variables:** 18  
- **Distribution Notes:**  
  - Credit scores ~ normal (mean 681.7)  
  - Income right–skewed  
  - Loan amounts vary widely  
- **Default Rate:** 26.56%  
- **Group Differences:**  
  - Significant differences in credit score, DTI, employment stability  
- **Covariance:** Groups have different covariance structures  
- **Normality:** Approx. multivariate normal after transformation  

---

# 6. Discriminant Analysis Suitability

- Sample size sufficient (N = 2,500)  
- Acceptable class imbalance (73% good / 27% default)  
- Predictor mix: 15 continuous, 2 categorical  
- Strong group separation expected  

---

# 7. Recommended Analytical Workflow

1. Exploratory Data Analysis  
2. Assumption Testing  
3. LDA and QDA modeling  
4. Stepwise variable selection  
5. Cross-validation & ROC analysis  

---

# 8. Business Context Variables

- **Industry:** Fintech lending (P2P and direct)  
- **Loan Types:** Personal, consolidation, small business  
- **Risk Framework:** Traditional scoring + ML enhancements  
- **Regulation:** Consumer lending / fair lending  

---

# 9. File Format

- **Encoding:** UTF-8  
- **Delimiter:** Comma  
- **Missing values:** Empty cells  
- **Header:** Included  
- **Index:** Not included — use application_id  

---

# 10. Note
This dataset is *synthetic* for educational purposes but reflects realistic financial patterns in credit risk modeling.
