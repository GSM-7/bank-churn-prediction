# Predictive Modeling & Risk Scoring for Bank Customer Churn

Churn probability scoring, explainability and a what-if simulator built with scikit-learn + Streamlit.

## Run locally
```
pip install -r requirements.txt
# put Churn_Modelling.csv inside /data
python train.py
streamlit run app.py
```
## Contents
- `train.py` – preprocessing, feature engineering, stratified split, 5-fold CV, 4–5 models, evaluation, permutation importance
- `app.py` – risk calculator, probability distribution, feature importance, what-if simulator, model comparison
- Engineered features: BalanceSalaryRatio, ProductDensity, EngagementProduct, AgeTenure

Live app: <paste Streamlit link>  |  Paper: RESEARCH_PAPER.md  |  Video: <paste link>
