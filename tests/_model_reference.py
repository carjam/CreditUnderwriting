from __future__ import annotations

from dataclasses import dataclass

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ModelRun:
    roc_auc: float
    accuracy: float
    n_test: int


def run_reference_model() -> ModelRun:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return ModelRun(
        roc_auc=float(roc_auc_score(y_test, proba)),
        accuracy=float((pred == y_test).mean()),
        n_test=int(y_test.shape[0]),
    )
