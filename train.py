"""Train churn models, evaluate, and save artifacts. Run: python train.py"""
import os, json
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

DATA = "data/Churn_Modelling.csv"
OUT = "models"
CAT = ["Geography", "Gender"]

def add_features(df):
    df = df.copy()
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["EngagementProduct"] = df["IsActiveMember"] * df["NumOfProducts"]
    df["AgeTenure"] = df["Age"] * df["Tenure"]
    return df

def load_xy(path=DATA):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ["RowNumber", "Year", "CustomerId", "Surname"] if c in df])
    df = df.dropna().drop_duplicates()
    y = df.pop("Exited")
    X = add_features(df)
    num = [c for c in X.columns if c not in CAT]
    return X.astype({c: float for c in num}), y

def build_pipeline(model, X):
    numeric = [c for c in X.columns if c not in CAT]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), numeric),
    ])
    return Pipeline([("pre", pre), ("clf", model)])

def main():
    os.makedirs(OUT, exist_ok=True)
    X, y = load_xy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          eval_metric="logloss", random_state=42)
    except ImportError:
        pass
    rows, fitted = [], {}
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for name, m in models.items():
        pipe = build_pipeline(m, X).fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        cv_auc = cross_val_score(build_pipeline(m, X), Xtr, ytr, cv=cv, scoring="roc_auc").mean()
        rows.append(dict(Model=name, Accuracy=accuracy_score(yte, pred),
                         Precision=precision_score(yte, pred, zero_division=0),
                         Recall=recall_score(yte, pred), F1=f1_score(yte, pred),
                         ROC_AUC=roc_auc_score(yte, p), CV_AUC=cv_auc))
        fitted[name] = pipe
        print(rows[-1])
    res = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
    res.to_csv(f"{OUT}/metrics.csv", index=False)
    best = res.iloc[0]["Model"]
    pipe = fitted[best]
    joblib.dump({"pipe": pipe, "name": best, "columns": list(X.columns)}, f"{OUT}/best_model.joblib")
    # model-agnostic permutation importance on raw input columns
    pi = permutation_importance(pipe, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42)
    (pd.DataFrame({"Feature": X.columns, "Importance": pi.importances_mean})
       .sort_values("Importance", ascending=False).to_csv(f"{OUT}/feature_importance.csv", index=False))
    pd.DataFrame({"prob": pipe.predict_proba(Xte)[:, 1], "actual": yte.values}).to_csv(f"{OUT}/test_scores.csv", index=False)
    # --- explainability artifacts: SHAP + partial dependence ---
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay
    os.makedirs("figures", exist_ok=True)
    try:
        import shap
        pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
        Z = pre.transform(Xte.sample(1000, random_state=42))
        Z = Z.toarray() if hasattr(Z, "toarray") else Z
        names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
        if hasattr(clf, "feature_importances_"):
            sv = shap.TreeExplainer(clf).shap_values(Z)
        else:
            sv = shap.LinearExplainer(clf, Z).shap_values(Z)
        sv = sv[1] if isinstance(sv, list) else sv
        shap.summary_plot(sv, Z, feature_names=names, show=False)
        plt.tight_layout(); plt.savefig("figures/shap_summary.png", dpi=150); plt.close()
        pd.DataFrame({"Feature": names, "MeanAbsSHAP": np.abs(sv).mean(0)}) \
          .sort_values("MeanAbsSHAP", ascending=False).to_csv(f"{OUT}/shap_importance.csv", index=False)
    except Exception as e:
        print("SHAP skipped:", e)
    top = ["Age", "NumOfProducts", "Balance", "IsActiveMember", "CreditScore", "Tenure"]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    PartialDependenceDisplay.from_estimator(pipe, Xte, top, ax=ax.ravel())
    plt.tight_layout(); plt.savefig("figures/partial_dependence.png", dpi=130); plt.close()
    json.dump({"best": best, "churn_rate": float(y.mean())}, open(f"{OUT}/summary.json", "w"))
    print("Best model:", best)

if __name__ == "__main__":
    main()
