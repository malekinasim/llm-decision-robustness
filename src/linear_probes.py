import numpy as np
from typing import Dict, Any
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from src.metrics import fisher_score, roc_auc, youden_optimal_threshold, confusion_from_scores
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit

def _standardize(X: np.ndarray):
    """Standardize features per layer (optional but helps LDA/LogReg/SVM)."""
    sc = StandardScaler(with_mean=True, with_std=True)
    return sc.fit_transform(X), sc

def _stratified_cap(X, y, max_n=5000, seed=42):
    if len(y) <= max_n: return X, y
    rng = np.random.RandomState(seed)
    pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    k_pos = min(len(pos), max_n//2); k_neg = max_n - k_pos
    pos_sel = rng.choice(pos, k_pos, replace=False)
    neg_sel = rng.choice(neg, k_neg, replace=False)
    sel = np.concatenate([pos_sel, neg_sel])
    return X[sel], y[sel]
def tune_probes_on_layer(X: np.ndarray, y: np.ndarray,method="lda", random_state: int = 42)-> Dict[str, Dict[str, Any]]:
    """
    Run a light CV-tuning on a single layer (X,y) and return best params for each method.
    Methods: 'lda', 'logreg', 'linsvm'
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    scoring = 'roc_auc' 

    best_params = {}

     # ---- LDA ----
    if(method=="lda"):
        lda_pipe = Pipeline([
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LinearDiscriminantAnalysis())
        ])
        lda_param_grid = [
        {"clf__solver": ["svd"], "clf__shrinkage": [None]},            
        {"clf__solver": ["lsqr"], "clf__shrinkage": ["auto", None]}, 
    ]
        lda_cv = GridSearchCV(lda_pipe, lda_param_grid, cv=cv, scoring="roc_auc",
                        n_jobs=-1, refit=True, error_score="raise")  # یا 'warn' / 'nan'
        lda_cv.fit(X, y)
        best_params["lda"] = lda_cv.best_params_

    # ---- Logistic Regression ----
    if(method=="logreg"):
        logreg_pipe = Pipeline([
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))  
        ])
        logreg_grid = {
            "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "clf__solver": ["liblinear", "lbfgs"],  
            "clf__penalty": ["l2"],
            "clf__tol": [1e-4, 1e-3]
        }
        logreg_cv = GridSearchCV(logreg_pipe, logreg_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        logreg_cv.fit(X, y)
        best_params["logreg"] = logreg_cv.best_params_

    # ---- Linear SVM ----
    if method == "linsvm":
        # X, y: ویژگی های TRAIN فقط برای یک لایه (مثلا mid_li)
        X_tune = X.astype(np.float32, copy=False)
        y_tune = y.astype(int, copy=False)
        # زیرنمونه کوچک فقط برای Grid
        X_small, y_small = _stratified_cap(X_tune, y_tune, max_n=4000)

        linsvm_pipe = Pipeline([
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LinearSVC(class_weight="balanced", dual=(X_small.shape[0] < X_small.shape[1])))
        ])
        linsvm_grid = {
            "clf__C":  [0.25, 0.5, 1.0],   # کوچکتر و سریع‌تر
            "clf__tol":[1e-3]              # 1e-4 کند می‌کند
        }
        cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
        linsvm_cv = GridSearchCV(
            linsvm_pipe, linsvm_grid, cv=cv,
            scoring="roc_auc", n_jobs=-1,
            refit=False,            # مهم: فقط params را پیدا کن
            pre_dispatch="2*n_jobs"
        )
        linsvm_cv.fit(X_small, y_small)
        best_params["linsvm"] = linsvm_cv.best_params_

    return best_params


def fit_eval_probes_per_layer(
    X_train_layers: Dict[int, np.ndarray], y_train: np.ndarray,
    X_test_layers: Dict[int, np.ndarray],  y_test: np.ndarray,method="lda",
    regularize_lda: float = 1e-3,best_params: Dict[str, Dict[str, Any]] | None = None
) -> Dict[str, Dict[int, dict]]:
    
    results = {}

    for li, Xtr in X_train_layers.items():
        Xte = X_test_layers[li]
        # Standardize both train & test on train stats
        Xtr_std, sc = _standardize(Xtr)
        Xte_std = sc.transform(Xte)

        # ===== LDA =====
        if method=="lda":
            
            try:
                p = (best_params or {}).get("lda", {})
                lda = LinearDiscriminantAnalysis(solver=p.get("clf__solver", "svd"),
                    shrinkage=p.get("clf__shrinkage", None))  # svd is fine; shrinkage optional
                lda.fit(Xtr_std, y_train)
                s_lda = lda.decision_function(Xte_std) if hasattr(lda, "decision_function") else lda.predict_proba(Xte_std)[:,1]
                au = roc_auc(y_test,s_lda)
                fs = fisher_score(s_lda, y_test)
                cm0 = confusion_from_scores(s_lda, y_test, thr=0.0)
                thr_star, _ = youden_optimal_threshold(s_lda, y_test)
                cm_star = confusion_from_scores(s_lda, y_test, thr=thr_star)
                results.setdefault('lda', {})
                results['lda'][li] = {'acc': cm0['acc'], 'auroc': au, 'fisher': fs,
                                    'thr0': 0.0, 'thr_star': thr_star,
                                    'cm_thr0': cm0, 'cm_thr_star': cm_star,'scores':s_lda}
            
            except Exception as e:
                results.setdefault('lda', {})
                results['lda'][li] = {'acc': 0.0, 'auroc': 0.5, 'fisher': 0.0, 'error': str(e)}

        # ===== Logistic Regression =====
        if method=="logreg":
            try:
                p = (best_params or {}).get("logreg", {})
                lr = LogisticRegression(
                    C=p.get("clf__C", 1.0),
                    solver=p.get("clf__solver", "liblinear"),
                    penalty=p.get("clf__penalty", "l2"),
                    tol=p.get("clf__tol", 1e-4),
                    max_iter=5000,
                    class_weight="balanced"
                )
                lr.fit(Xtr_std, y_train)
                s_lr = lr.decision_function(Xte_std)
                au = roc_auc( y_test,s_lr)
                fs = fisher_score(s_lr, y_test)
                cm0 = confusion_from_scores(s_lr, y_test, thr=0.0)
                thr_star, _ = youden_optimal_threshold(s_lr, y_test)
                cm_star = confusion_from_scores(s_lr, y_test, thr=thr_star)
                results.setdefault('logreg', {})
                results['logreg'][li] = {'acc': cm0['acc'], 'auroc': au, 'fisher': fs,
                                        'thr0': 0.0, 'thr_star': thr_star,
                                        'cm_thr0': cm0, 'cm_thr_star': cm_star,'scores':s_lr}
            except Exception as e:
                results.setdefault('logreg', {})
                results['logreg'][li] = {'acc': 0.0, 'auroc': 0.5, 'fisher': 0.0, 'error': str(e)}

        # ===== Linear SVM =====
        if method=="linsvm":
            try:
                p = (best_params or {}).get("linsvm", {})
                n, d = Xtr_std.shape
                svm = LinearSVC(
                    C=p.get("clf__C", 1.0),
                    tol=p.get("clf__tol", 1e-4),
                    max_iter=10000,
                    class_weight="balanced",
                    dual=(n < d)
                )
                svm.fit(Xtr_std, y_train)
                # LinearSVC returns distance to boundary in decision_function
                s_svm = svm.decision_function(Xte_std)
                au = roc_auc(y_test,s_svm )
                fs = fisher_score(s_svm, y_test)
                cm0 = confusion_from_scores(s_svm, y_test, thr=0.0)
                thr_star, _ = youden_optimal_threshold(s_svm, y_test)
                cm_star = confusion_from_scores(s_svm, y_test, thr=thr_star)
                results.setdefault('linsvm', {})
                results['linsvm'][li] = {'acc': cm0['acc'], 'auroc': au, 'fisher': fs,
                                        'thr0': 0.0, 'thr_star': thr_star,
                                        'cm_thr0': cm0, 'cm_thr_star': cm_star ,'scores':s_svm}
            except Exception as e:
                results.setdefault('linsvm', {})
                results['linsvm'][li] = {'acc': 0.0, 'auroc': 0.5, 'fisher': 0.0, 'error': str(e)}
    return  results.get(method, {})

def scores_for_probe(method: str, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """
    Return decision scores s = f(x) for the chosen linear probe.
    method in {"massmean","lda","logreg","linsvm"}
    """
    if method == "massmean":
        mu_pos = Xtr[ytr==1].mean(axis=0); mu_neg = Xtr[ytr==0].mean(axis=0)
        w = mu_pos - mu_neg
        w = w / (np.linalg.norm(w) + 1e-8)
        return Xte @ w

    # standardized train/test for the learned probes
    sc = StandardScaler().fit(Xtr)
    Xtr_std = sc.transform(Xtr); Xte_std = sc.transform(Xte)

    if method == "lda":
        lda = LinearDiscriminantAnalysis(solver="svd").fit(Xtr_std, ytr)
        return lda.decision_function(Xte_std) if hasattr(lda, "decision_function") else lda.predict_proba(Xte_std)[:,1]

    if method == "logreg":
        lr = LogisticRegression(max_iter=2000).fit(Xtr_std, ytr)
        return lr.decision_function(Xte_std)

    if method == "linsvm":
        svm = LinearSVC().fit(Xtr_std, ytr)
        return svm.decision_function(Xte_std)

    raise ValueError(f"unknown probe method: {method}")
