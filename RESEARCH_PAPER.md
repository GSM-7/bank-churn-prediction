# Predictive Modeling and Risk Scoring for Bank Customer Churn

## Abstract
Retention actions at retail banks are often reactive and broad. This study builds a predictive churn-scoring system on a 10,000-customer European retail-banking dataset (20.4% churn). Five models were compared (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost) using a stratified 80/20 split and 5-fold cross-validation. Gradient Boosting achieved the best discrimination (ROC-AUC 0.869, CV-AUC 0.864, accuracy 86.9%, precision 78.7%). Explainability analysis (permutation importance, SHAP, partial dependence) shows that product count, age, geography (Germany) and engagement are the dominant churn drivers. A Streamlit application exposes churn probabilities, feature importance and a what-if simulator for retention teams.

## 1. Introduction
Churn reduces customer lifetime value, revenue stability and cross-sell potential. Descriptive churn analysis explains why customers left; this project instead scores customers *before* they leave so that retention campaigns can be targeted and cost-efficient. Primary objectives: predict churn accurately, produce probability scores, identify key drivers. Secondary objectives: limit false positives, improve interpretability, enable scenario analysis.

## 2. Data and Exploratory Analysis
The dataset has 10,000 customers and 14 columns, with no missing values and no duplicates. Non-informative fields (Year, CustomerId, Surname) were removed. The target is `Exited` (1 = churned); 20.37% of customers churned, a moderate class imbalance handled with stratified splitting.

Key EDA findings (churn rate by segment):
| Segment | Churn rate |
|---|---|
| France / Spain / **Germany** | 16.2% / 16.7% / **32.4%** |
| Female / Male | 25.1% / 16.5% |
| Inactive / Active members | 26.9% / 14.3% |
| 1 / 2 / 3 / 4 products | 27.7% / **7.6%** / 82.7% / 100% |
| Age 30-40 / 40-50 / 50-60 | 12.1% / 34.0% / **56.2%** |

Two patterns stand out: customers with two products are by far the most loyal, while customers with 3+ products churn at extreme rates (small groups: 266 and 60 customers); and churn rises sharply from age 40, peaking in the 50-60 band.

## 3. Methodology
**Preprocessing.** One-hot encoding (Geography, Gender), standard scaling of numeric features inside a scikit-learn Pipeline (so no leakage into the test set).
**Feature engineering.** Balance-to-Salary ratio; Product density (products / (tenure+1)); Engagement-product interaction (IsActiveMember x NumOfProducts); Age-tenure interaction.
**Split.** Stratified 80/20 train-test split (random_state=42) plus stratified 5-fold CV on the training set.
**Models.** Logistic Regression (interpretable baseline), Decision Tree, Random Forest, Gradient Boosting, XGBoost.
**Metrics.** Accuracy, Precision, Recall, F1, ROC-AUC (threshold 0.5).

## 4. Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.869** | **0.787** | 0.489 | 0.603 | **0.869** | **0.864** |
| Random Forest | 0.844 | 0.612 | 0.629 | **0.621** | 0.865 | 0.856 |
| XGBoost | 0.868 | 0.773 | 0.494 | 0.603 | 0.864 | 0.861 |
| Decision Tree | 0.762 | 0.451 | **0.784** | 0.573 | 0.842 | 0.830 |
| Logistic Regression | 0.710 | 0.383 | 0.698 | 0.494 | 0.776 | 0.766 |

Tree ensembles clearly beat the linear baseline (ROC-AUC +0.09), indicating non-linear effects (e.g., age and product count). Gradient Boosting was selected on ROC-AUC. Its precision of 78.7% means roughly four in five flagged customers truly churn, which supports the secondary goal of fewer false alarms. The trade-off is recall (48.9% at the 0.5 threshold); lowering the threshold raises recall at the cost of precision, and the app lets analysts tune it. Random Forest gives the best F1 and a more balanced profile if recall matters more than campaign cost.

## 5. Explainability
Permutation importance (drop in ROC-AUC) ranks **Age** and **NumOfProducts** far above all other features. SHAP (mean |value|) gives the same top two (NumOfProducts 0.75, Age 0.72), followed by Geography_Germany (0.24), Gender_Male (0.21), the engagement-product interaction (0.19), IsActiveMember (0.14) and Balance-to-Salary ratio (0.14). The engineered engagement and balance-ratio features add signal beyond the raw columns. Partial dependence plots (figures/partial_dependence.png) show churn probability rising steeply with age through the 50s and a non-monotonic effect of product count. These plots and the SHAP summary (figures/shap_summary.png) give transparent, auditable reasons per prediction, supporting regulatory and business trust.

## 6. Application
The Streamlit dashboard provides: (i) a customer churn risk calculator with adjustable threshold and risk band; (ii) predicted-probability distributions for churned vs retained customers; (iii) feature importance, SHAP and partial dependence views; (iv) a what-if simulator for engagement, product count, balance and tenure, with sensitivity sweeps; (v) a model comparison table.

## 7. Recommendations
1. **Prioritise 40-60 year-old customers**, the highest-risk age band, with relationship-manager outreach.
2. **Promote the two-product bundle.** Two-product customers churn at 7.6%; single-product customers churn at 27.7%. Investigate why 3-4 product customers churn almost universally (possible mis-selling, fees or service issues) before pushing a third product.
3. **Re-activate inactive members** with engagement campaigns; inactive customers churn at nearly twice the active rate.
4. **Run a Germany-specific retention review**; churn there is double that of France and Spain.
5. **Use probability bands, not a single cut-off:** high-touch offers for the top risk decile, low-cost digital nudges for the middle band.

## 8. Limitations and Future Work
Data is a single snapshot with no transactional or complaint history; churn timing is unknown. Fairness checks across gender and geography should be completed before deployment because both appear as drivers. Future work: probability calibration, cost-sensitive threshold optimisation using campaign cost and CLV, and time-series behavioural features.

## 9. Conclusion
A gradient-boosted model scores churn risk with ROC-AUC 0.87 and 79% precision, and the drivers are consistent across three explainability methods. Engagement and product utilisation, not just demographics, offer the clearest levers for retention strategy.
