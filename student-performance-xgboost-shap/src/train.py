import os
import pickle
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV,
    cross_val_score, StratifiedKFold, KFold
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, label_binarize
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    classification_report, mean_squared_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.ensemble import (
    StackingClassifier, StackingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from imblearn.over_sampling import SMOTE
from data_loader import DatasetLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_and_evaluate(self, dataset_name: str, X: pd.DataFrame,
                           y: pd.Series, task_type: str = 'regression'):
        logger.info(f"\n{'='*60}")
        logger.info(f"  Stacking Ensemble Pipeline — {dataset_name.upper()}")
        logger.info(f"{'='*60}")

        raw_features = list(X.columns)
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        cat_classes = {}
        for col in cat_cols:
            cat_classes[col] = sorted(X[col].dropna().astype(str).unique().tolist())

        # ---- 1. Imputation (fit on all — safe, no label leakage) ----------
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        if num_cols:
            X[num_cols] = num_imputer.fit_transform(X[num_cols])
        if cat_cols:
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

        # ---- 2. Ordinal Encoding (fit on all — safe, no label leakage) ----
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        if cat_cols:
            X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))

        # ---- 3. SMOTE class distribution (classification only) ------------
        class_dist_before = "N/A"
        class_dist_after  = "N/A"
        if task_type == 'classification':
            class_dist_before = str(y.value_counts().sort_index().to_dict())
            logger.info(f"Class distribution BEFORE SMOTE: {class_dist_before}")

        # ---- 4. Train / Test Split (BEFORE scaling, RFE, and SMOTE) -------
        # FIX 1: Split first so scaler and RFE never see test labels
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task_type == 'classification' else None
        )

        # ---- 5. StandardScaler (fit on train only) ------------------------
        # FIX 1 (continued): scaler fitted on X_train, applied to both
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns
        )

        # ---- 6. Feature Selection (RFE — fit on train only) ---------------
        # FIX 1 (continued): RFE fitted on X_train so test labels never
        # influence which features are selected
        logger.info("Running RFE for feature selection...")
        if task_type == 'classification':
            selector_estimator = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
            )
        else:
            selector_estimator = RandomForestRegressor(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
            )

        n_features_to_select = max(5, int(X_train_scaled.shape[1] * 0.75))
        selector = RFE(
            estimator=selector_estimator,
            n_features_to_select=n_features_to_select, step=1
        )
        selector.fit(X_train_scaled, y_train)

        X_train_selected = X_train_scaled.loc[:, selector.support_]
        X_test_selected  = X_test_scaled.loc[:,  selector.support_]
        selected_features = X_train_selected.columns.tolist()
        logger.info(
            f"RFE complete: {X_train_scaled.shape[1]} → {len(selected_features)} features retained"
        )

        # ---- 7. SMOTE (applied to training fold only) ---------------------
        # FIX 2: SMOTE runs after the split so synthetic samples never
        # contaminate the test set
        if task_type == 'classification':
            try:
                n_neighbors = min(5, pd.Series(y_train).value_counts().min() - 1)
                smote = SMOTE(random_state=42, k_neighbors=max(1, n_neighbors))
                X_train_selected, y_train = smote.fit_resample(X_train_selected, y_train)
                class_dist_after = str(
                    pd.Series(y_train).value_counts().sort_index().to_dict()
                )
                logger.info(f"Class distribution AFTER SMOTE:  {class_dist_after}")
            except Exception as e:
                logger.warning(f"SMOTE skipped: {e}")

        # ---- 8. Build Stacking Ensemble -----------------------------------
        logger.info("Building Stacking Ensemble...")
        model, cv_strategy, scoring = self._build_model(task_type, X_train_selected.shape[0])

        # ---- 9. 5-Fold Cross-Validation -----------------------------------
        logger.info("Running 5-Fold Cross-Validation...")
        cv_scores = cross_val_score(
            model, X_train_selected, y_train,
            cv=cv_strategy, scoring=scoring, n_jobs=-1
        )
        logger.info(
            f"CV {scoring}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        # ---- 10. Fit on full training set ----------------------------------
        logger.info("Fitting ensemble on full training set...")
        model.fit(X_train_selected, y_train)

        # ---- 11. Evaluate on test set -------------------------------------
        self._evaluate_model(
            dataset_name, model, X_test_selected, y_test,
            task_type, cv_scores, class_dist_before, class_dist_after, scoring
        )

        # ---- 12. SHAP KernelExplainer -------------------------------------
        logger.info("Building SHAP KernelExplainer (using k-means background)...")
        background = shap.kmeans(X_train_selected, min(50, len(X_train_selected)))
        predict_fn = model.predict if task_type == 'regression' else model.predict_proba
        explainer = shap.KernelExplainer(predict_fn, background)

        # ---- 13. Save artifacts -------------------------------------------
        self._save_artifacts(
            dataset_name, model, explainer,
            num_imputer, cat_imputer, encoder, scaler, selector,
            raw_features, num_cols, cat_cols, cat_classes, selected_features
        )
        logger.info(f"All artifacts saved for {dataset_name.upper()}.\n")

    def _build_model(self, task_type: str, n_train: int):
        if task_type == 'regression':
            base_models = [
                ('rf', RandomForestRegressor(
                    # FIX 3: tighter regularisation for small UCI dataset
                    n_estimators=300, max_depth=4,
                    min_samples_leaf=10, max_features='sqrt',
                    random_state=42, n_jobs=-1
                )),
                ('gb', GradientBoostingRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42
                )),
            ]
            # FIX 3: replace tuned XGBoost meta-learner with Ridge —
            # XGBoost overfits the stacking layer on ~320 training samples
            from sklearn.linear_model import Ridge
            model = StackingRegressor(
                estimators=base_models,
                final_estimator=Ridge(alpha=1.0),
                cv=5
            )
            cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'

        else:
            base_models = [
                ('rf', RandomForestClassifier(
                    n_estimators=300, max_depth=6,
                    min_samples_leaf=2, class_weight='balanced',
                    random_state=42, n_jobs=-1
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42
                )),
            ]
            xgb_meta = xgb.XGBClassifier(
                objective='multi:softprob', random_state=42,
                eval_metric='mlogloss', verbosity=0,
                use_label_encoder=False
            )
            param_dist = {
                'n_estimators':    [200, 300, 400],
                'max_depth':       [3, 4, 5, 6],
                'learning_rate':   [0.01, 0.05, 0.1],
                'subsample':       [0.7, 0.8, 0.9],
                'colsample_bytree':[0.7, 0.8, 1.0],
                'min_child_weight':[1, 3, 5],
            }
            tuned_meta = RandomizedSearchCV(
                xgb_meta, param_dist, n_iter=20,
                cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=0
            )
            model = StackingClassifier(
                estimators=base_models, final_estimator=tuned_meta, cv=3
            )
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'

        return model, cv_strategy, scoring

    def _evaluate_model(self, dataset_name, model, X_test, y_test,
                        task_type, cv_scores, dist_before, dist_after, scoring):
        y_pred = model.predict(X_test)
        report_path = self.models_dir / f"{dataset_name}_report.txt"

        with open(report_path, "w") as f:
            f.write(f"--- Academic Performance Report: {dataset_name.upper()} ---\n\n")
            f.write(
                f"5-Fold Cross Validation ({scoring.upper()}): "
                f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n\n"
            )

            if task_type == 'classification':
                f.write(f"Class Distribution Before SMOTE: {dist_before}\n")
                f.write(f"Class Distribution After SMOTE:  {dist_after}\n\n")

            if task_type == 'regression':
                mse  = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2   = r2_score(y_test, y_pred)
                f.write(f"Test Set RMSE:     {rmse:.4f}\n")
                f.write(f"Test Set MSE:      {mse:.4f}\n")
                f.write(f"Test Set R² Score: {r2:.4f}\n")
                logger.info(f"[{dataset_name.upper()}] RMSE={rmse:.4f}  R²={r2:.4f}")
            else:
                report = classification_report(y_test, y_pred)
                f.write("Test Set Classification Report:\n")
                f.write(report)
                logger.info(f"[{dataset_name.upper()}]\n{report}")

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(7, 6))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues, ax=ax)
                ax.set_title(f"Confusion Matrix — {dataset_name.upper()}", fontsize=13)
                plt.tight_layout()
                plt.savefig(self.models_dir / f"{dataset_name}_cm.png",
                            dpi=150, bbox_inches='tight')
                plt.close(fig)

                classes = np.unique(y_test)
                n_classes = len(classes)
                if n_classes >= 2:
                    y_score  = model.predict_proba(X_test)
                    y_test_bin = label_binarize(y_test, classes=classes)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
                    class_labels = {
                        0: 'Low / Fail',
                        1: 'Medium / Withdrawn',
                        2: 'High / Pass',
                        3: 'Distinction'
                    }
                    for i, cls in enumerate(classes):
                        col_idx = i if y_test_bin.shape[1] > 1 else 1
                        fpr, tpr, _ = roc_curve(
                            y_test_bin[:, col_idx],
                            y_score[:, i] if y_score.ndim > 1 else y_score
                        )
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, lw=2.5,
                                color=colors[i % len(colors)],
                                label=f"{class_labels.get(int(cls), cls)} (AUC={roc_auc:.2f})")

                    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
                    ax.set_xlabel('False Positive Rate', fontsize=12)
                    ax.set_ylabel('True Positive Rate', fontsize=12)
                    ax.set_title(f'ROC Curve — {dataset_name.upper()}', fontsize=13)
                    ax.legend(loc='lower right', fontsize=10)
                    plt.tight_layout()
                    plt.savefig(self.models_dir / f"{dataset_name}_roc.png",
                                dpi=150, bbox_inches='tight')
                    plt.close(fig)

    def _save_artifacts(self, dataset_name, model, explainer,
                        num_imputer, cat_imputer, encoder, scaler, selector,
                        raw_features, num_cols, cat_cols, cat_classes, selected_features):
        pipeline_dict = {
            'num_imputer':       num_imputer,
            'cat_imputer':       cat_imputer,
            'encoder':           encoder,
            'scaler':            scaler,
            'selector':          selector,
            'raw_features':      raw_features,
            'num_cols':          num_cols,
            'cat_cols':          cat_cols,
            'cat_classes':       cat_classes,
            'selected_features': selected_features,
        }
        with open(self.models_dir / f"{dataset_name}_model.pkl",     "wb") as f:
            pickle.dump(model, f)
        with open(self.models_dir / f"{dataset_name}_explainer.pkl", "wb") as f:
            pickle.dump(explainer, f)
        with open(self.models_dir / f"{dataset_name}_pipeline.pkl",  "wb") as f:
            pickle.dump(pipeline_dict, f)
        logger.info(
            f"Saved: {dataset_name}_model.pkl, _explainer.pkl, _pipeline.pkl"
        )


if __name__ == "__main__":
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(Path(__file__).parent))

    loader  = DatasetLoader(data_dir=str(PROJECT_ROOT / "data"))
    trainer = ModelTrainer(models_dir=str(PROJECT_ROOT / "models"))

    datasets = [
        ("uci",   loader.load_uci_data,   'regression'),
        ("xapi",  loader.load_xapi_data,  'classification'),
        ("oulad", loader.load_oulad_data, 'classification'),
    ]

    for name, load_func, task in datasets:
        logger.info(f"\nLoading {name.upper()} dataset...")
        X, y = load_func()
        trainer.train_and_evaluate(name, X, y, task_type=task)

    logger.info("\n All pipelines completed successfully.")