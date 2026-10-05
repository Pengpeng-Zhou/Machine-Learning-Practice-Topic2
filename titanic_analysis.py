"""题目二：Kaggle Titanic 生存预测。

Run from this directory with ``python titanic_analysis.py``.  All learned
preprocessing steps are inside sklearn Pipelines, so cross-validation does not
fit imputers, scalers, or encoders on the validation fold.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"


def configure_plot() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
    })


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic features from passenger information."""
    df = frame.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df.loc[~df["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Title"] = "Rare"
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = np.where(df["FamilySize"] == 1, "Yes", "No")
    df["HasCabin"] = np.where(df["Cabin"].notna(), "Yes", "No")
    df["FareLog"] = np.log1p(df["Fare"])
    df["TicketPrefix"] = (
        df["Ticket"].str.replace(r"\d", "", regex=True)
        .str.replace(r"[\W_]+", "", regex=True)
        .str.upper()
        .replace("", "NONE")
    )
    df["AgeBand"] = pd.cut(
        df["Age"], bins=[-np.inf, 5, 12, 18, 35, 60, np.inf],
        labels=["Child", "School", "Teen", "Adult", "Mature", "Senior"],
    ).astype(object)
    df["FamilySizeGroup"] = pd.cut(
        df["FamilySize"], bins=[0, 1, 4, 7, np.inf],
        labels=["Alone", "Small", "Medium", "Large"],
    ).astype(object)
    return df


def make_one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse=False)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop")


def make_pipeline(model, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocess", make_preprocessor(numeric, categorical)),
        ("model", model),
    ])


def annotate_bars(ax, bars, values, fmt="{:.4f}") -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.006,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=8,
        )


def plot_eda(train: pd.DataFrame, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Survival by sex and passenger class.
    ax = axes[0, 0]
    grouped = train.groupby(["Pclass", "Sex"], observed=False)["Survived"].agg(["mean", "count"]).unstack()
    classes = [1, 2, 3]
    x = np.arange(len(classes))
    width = 0.34
    for i, sex in enumerate(["female", "male"]):
        means = [grouped["mean"][sex].get(c, np.nan) for c in classes]
        counts = [grouped["count"][sex].get(c, 0) for c in classes]
        bars = ax.bar(x + (i - 0.5) * width, means, width, label=sex.title())
        for bar, value, n in zip(bars, means, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"n={int(n)}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate")
    ax.set_title("(a) Survival by Sex and Class")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    # (b) Age distributions.
    ax = axes[0, 1]
    for survived, color in [(0, "#777777"), (1, "#4472C4")]:
        values = train.loc[train["Survived"] == survived, "Age"].dropna()
        ax.hist(values, bins=20, density=True, alpha=0.52, color=color,
                label=f"Survived = {survived}")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Density")
    ax.set_title("(b) Age Distribution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    # (c) Fare distributions on a logarithmic x-axis.
    ax = axes[1, 0]
    for survived, color in [(0, "#777777"), (1, "#ED7D31")]:
        values = train.loc[train["Survived"] == survived, "Fare"].replace(0, np.nan).dropna()
        ax.hist(values, bins=25, density=True, alpha=0.52, color=color,
                label=f"Survived = {survived}")
    ax.set_xscale("log")
    ax.set_xlabel("Fare (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("(c) Fare Distribution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    # (d) Family size and survival.
    ax = axes[1, 1]
    family = df.groupby("FamilySize", observed=False)["Survived"].agg(["mean", "count"]).reset_index()
    ax.plot(family["FamilySize"], family["mean"], marker="o", color="#70AD47", lw=1.5)
    for _, row in family.iterrows():
        ax.text(row["FamilySize"], row["mean"] + 0.035, f"n={int(row['count'])}",
                ha="center", fontsize=7)
    ax.set_xlabel("Family Size")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("(d) Family Size and Survival")
    ax.grid(alpha=0.2)

    fig.suptitle("Titanic Exploratory Data Analysis", y=0.995, fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_eda.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    order = metrics.sort_values("roc_auc_mean")["model"].tolist()
    plot_df = metrics.set_index("model").loc[order].reset_index()
    y = np.arange(len(plot_df))
    for ax, metric, label, color in [
        (axes[0], "accuracy_mean", "Accuracy", "#4472C4"),
        (axes[1], "roc_auc_mean", "ROC-AUC", "#ED7D31"),
    ]:
        bars = ax.barh(y, plot_df[metric], xerr=plot_df[metric.replace("_mean", "_std")],
                       color=color, alpha=0.86, capsize=3)
        annotate_bars(ax, bars, plot_df[metric])
        ax.set_yticks(y, plot_df["model"])
        ax.set_xlabel(label)
        ax.set_xlim(max(0, plot_df[metric].min() - 0.08), 1.0)
        ax.grid(axis="x", alpha=0.2)
        ax.set_title(label)
    fig.suptitle("Stratified 5-Fold Model Comparison", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_evaluation(y: pd.Series, oof_pred: np.ndarray, oof_prob: np.ndarray,
                          fold_curves: list[tuple[np.ndarray, np.ndarray, float]],
                          best_name: str) -> dict:
    cm = confusion_matrix(y, oof_pred, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / total:.1%})",
                    ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Not Survived", "Survived"])
    ax.set_yticks([0, 1], ["Not Survived", "Survived"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"(a) Confusion Matrix: {best_name}")

    ax = axes[1]
    for i, (fpr, tpr, fold_auc) in enumerate(fold_curves, 1):
        ax.plot(fpr, tpr, lw=1.0, alpha=0.45, label=f"Fold {i} (AUC={fold_auc:.3f})")
    pooled_fpr, pooled_tpr, _ = roc_curve(y, oof_prob)
    pooled_auc = roc_auc_score(y, oof_prob)
    ax.plot(pooled_fpr, pooled_tpr, color="black", lw=2.4, label=f"OOF pooled (AUC={pooled_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"(b) Out-of-Fold ROC: {best_name}")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(alpha=0.2)
    fig.suptitle("Best Model Evaluation", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_model_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"confusion_matrix": cm.tolist(), "oof_accuracy": float(accuracy_score(y, oof_pred)),
            "oof_auc": float(pooled_auc)}


def plot_feature_importance(fitted: Pipeline, best_name: str) -> pd.DataFrame:
    preprocess = fitted.named_steps["preprocess"]
    estimator = fitted.named_steps["model"]
    names = preprocess.get_feature_names_out()
    names = [name.replace("num__", "").replace("cat__", "") for name in names]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        importance_type = "Tree feature importance"
    else:
        values = np.abs(estimator.coef_.ravel())
        importance_type = "Absolute standardized coefficient"
    importance = pd.DataFrame({"feature": names, "importance": values})
    importance = importance.sort_values("importance", ascending=False).head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    bars = ax.barh(importance["feature"], importance["importance"], color="#70AD47")
    annotate_bars(ax, bars, importance["importance"], fmt="{:.4f}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top 15 Features: {best_name}\n({importance_type})")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return importance.sort_values("importance", ascending=False)


def main() -> None:
    configure_plot()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8")
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8")
    train_features = add_features(train.drop(columns=["Survived"]))
    test_features = add_features(test)
    all_train = add_features(train)
    plot_eda(train, all_train)

    numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "FareLog"]
    categorical = ["Sex", "Embarked", "Title", "IsAlone", "HasCabin", "TicketPrefix", "AgeBand", "FamilySizeGroup"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Sex baseline": make_pipeline(LogisticRegression(max_iter=2000, random_state=42), [], ["Sex"]),
        "Logistic regression": make_pipeline(LogisticRegression(max_iter=2000, random_state=42), numeric, categorical),
        "Random forest": make_pipeline(RandomForestClassifier(
            n_estimators=400, max_depth=7, min_samples_leaf=2, random_state=42, n_jobs=-1,
        ), numeric, categorical),
        "Gradient boosting": make_pipeline(GradientBoostingClassifier(
            n_estimators=160, learning_rate=0.04, max_depth=2, random_state=42,
        ), numeric, categorical),
    }
    metrics_rows = []
    for name, model in models.items():
        scores = cross_validate(model, train_features if name != "Sex baseline" else train_features,
                                train["Survived"], cv=cv, scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"},
                                return_train_score=False, n_jobs=1)
        metrics_rows.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(ddof=1),
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(ddof=1),
        })
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(ROOT / "model_metrics.csv", index=False, encoding="utf-8-sig")
    plot_model_comparison(metrics)

    best_name = metrics.sort_values(["roc_auc_mean", "accuracy_mean"], ascending=False).iloc[0]["model"]
    best_model = clone(models[best_name])
    oof_pred = cross_val_predict(best_model, train_features, train["Survived"], cv=cv, method="predict")
    oof_prob = cross_val_predict(best_model, train_features, train["Survived"], cv=cv, method="predict_proba")[:, 1]
    fold_curves = []
    for fold_train, fold_valid in cv.split(train_features, train["Survived"]):
        fold_model = clone(models[best_name])
        fold_model.fit(train_features.iloc[fold_train], train["Survived"].iloc[fold_train])
        prob = fold_model.predict_proba(train_features.iloc[fold_valid])[:, 1]
        fpr, tpr, _ = roc_curve(train["Survived"].iloc[fold_valid], prob)
        fold_curves.append((fpr, tpr, auc(fpr, tpr)))
    evaluation = plot_model_evaluation(train["Survived"], oof_pred, oof_prob, fold_curves, best_name)

    fitted = clone(models[best_name]).fit(train_features, train["Survived"])
    importance = plot_feature_importance(fitted, best_name)
    importance.to_csv(ROOT / "feature_importance.csv", index=False, encoding="utf-8-sig")

    test_pred = fitted.predict(test_features).astype(int)
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
    submission.to_csv(ROOT / "submission.csv", index=False)

    summary = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_survivors": int(train["Survived"].sum()),
        "train_survival_rate": float(train["Survived"].mean()),
        "test_predicted_survivors": int(test_pred.sum()),
        "test_predicted_survival_rate": float(test_pred.mean()),
        "best_model": best_name,
        "metrics": metrics.to_dict(orient="records"),
        "evaluation": evaluation,
        "new_features": {
            "Title": "从 Name 提取称谓并将低频称谓合并为 Rare。",
            "FamilySize": "SibSp + Parch + 1，表示同行家庭规模。",
            "IsAlone": "FamilySize 是否为 1。",
            "HasCabin": "Cabin 是否有记录，代理客舱/社会阶层信息。",
            "FareLog": "log(1 + Fare)，减弱票价右偏。",
            "TicketPrefix": "从 Ticket 提取字母前缀，表示票号类别。",
            "AgeBand": "将年龄划分为 Child/School/Teen/Adult/Mature/Senior。",
            "FamilySizeGroup": "将家庭规模分为 Alone/Small/Medium/Large。",
        },
    }
    (ROOT / "titanic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_model": best_name,
        "metrics": metrics.round(4).to_dict(orient="records"),
        "predicted_survivors": int(test_pred.sum()),
        "predicted_survival_rate": float(test_pred.mean()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
