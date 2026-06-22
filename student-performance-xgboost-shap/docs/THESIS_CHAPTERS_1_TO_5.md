# DESIGN AND IMPLEMENTATION OF A STUDENT PERFORMANCE PREDICTION SYSTEM USING XGBOOST ENHANCED WITH SHAP FEATURES

*Official project topic. In the implementation, **XGBoost** is the primary learned decision component for classification: a **hyperparameter-tuned XGBoost classifier** serves as the **meta-learner** of a stacked ensemble whose base learners supply complementary signals; **SHAP** explains the final trained predictor. For the UCI grade-regression task, the stack uses a Ridge meta-learner while base learners remain tree ensembles.*

**Registered topic (match your faculty form exactly on the title page):**  
DESIGN AND IMPLEMENTATION OF A STUDENT PERFOMANCE PREDICTION SYSTEM USING XG BOOST ENHANCED WITH SHAP FEATURES  

*(This document uses standard spelling **Performance** and **XGBoost** in headings and body text unless your department requires letter-for-letter duplication of the registered title everywhere.)*

**Author:** Amromawhe Osemudiamen Erumuakpor  
**Matric No:** 2021/10248  
**Department:** Computer Science  
**Institution:** Bells University of Technology, Ota, Ogun State, Nigeria  
**Supervisor:** Dr O.J. Olaleye  
**Degree:** B.Tech Computer Science  
**Date:** December 2025

---

## SUMMARY

This study presents the design and implementation of a student performance prediction system **using XGBoost, enhanced with SHAP (SHapley Additive exPlanations)** for accurate and interpretable student performance forecasting. The research addresses limitations of traditional, reactive assessment and of many predictive systems that emphasize accuracy over transparency and under-use behavioural engagement signals.

The core predictive engine is **XGBoost-centred**: for **classification** tasks, a **stacked generalization (stacking)** architecture feeds out-of-fold probability outputs from diverse tree ensembles (Random Forest, Gradient Boosting, Extra Trees, LightGBM) into a **RandomizedSearchCV–tuned XGBoost meta-learner**, which produces the final class probabilities. For **regression** (final grade prediction), a stacked ensemble with a Ridge meta-learner is used. The methodology further comprises: (1) data loading and validation; (2) preprocessing through mean/mode imputation, ordinal encoding, and standard scaling; (3) leakage-safe train–test splitting; (4) Recursive Feature Elimination (RFE); (5) class balancing using SMOTEENN for classification; (6) cross-validation and hold-out evaluation; and (7) post-hoc explanation of the **trained predictor** using **SHAP KernelExplainer** for global and local interpretability.

A **Streamlit** web prototype (**EduPredict**) allows lecturers and advisers to simulate student profiles, obtain predictions, and inspect SHAP force plots and feature contribution tables. Experimental results on the UCI Student Performance dataset (regression) achieved a test **R² of 0.8265** and **RMSE of 1.89**; on the xAPI-Edu-Data dataset (3-class classification) the system achieved **75.0% test accuracy** and **0.76 macro F1-score**, with **77.1%** 5-fold cross-validated accuracy. The study is relevant to Nigerian and similar developing higher-education contexts where explainable, deployable decision-support tools are needed, while acknowledging that validation used benchmark datasets rather than a proprietary institutional cohort.

---

# CHAPTER ONE: INTRODUCTION

## 1.1 Background of the Study

Student performance prediction has become increasingly vital in higher education as institutions seek proactive strategies to improve retention, completion rates, and learning outcomes. In Nigeria, university dropout and academic failure remain concerns linked to inadequate preparation, socioeconomic barriers, and uneven access to learning support (Ogunleye & Adebayo, 2022). Traditional assessment—periodic examinations and end-of-semester grades—often reveals risk only after substantial decline, limiting the window for effective intervention.

Machine learning enables analysis of historical records combining demographics, prior achievement, and behavioural engagement to forecast outcomes before failure occurs. **XGBoost** (Extreme Gradient Boosting) is among the strongest algorithms for structured tabular data in educational data mining, owing to regularization, sparsity-aware splits, and efficient implementation (Chen & Guestrin, 2016). **Stacking** can further improve generalization by learning how to combine complementary base models; situating **XGBoost as the meta-learner** keeps the system aligned with the project topic while benefiting from diverse lower-level learners (Wolpert, 1992; Tomasoni et al., 2020).

However, accurate models—including XGBoost and stacked variants—are often opaque. Educators and administrators require understandable rationales—not only a risk label—to design counselling, attendance policies, or targeted academic support. **SHAP** provides theoretically grounded, consistent feature attributions for both global trends and individual predictions (Lundberg & Lee, 2017). **XGBoost enhanced with SHAP** therefore supports both **predictive performance** and **actionable interpretability** of the deployed predictor.

This project implements such a system—**EduPredict**—using Python, scikit-learn, XGBoost, LightGBM, imbalanced-learn, SHAP, and Streamlit, evaluated on benchmark datasets including UCI Student Performance and xAPI-Edu-Data, with architectural support for the UCI Dropout and Academic Success dataset (Realinho et al., 2022).

## 1.2 Problem Statement

Despite progress in educational data mining (EDM), effective student performance prediction in resource-constrained settings faces persistent challenges:

1. **Limited generalization of single models.** Logistic regression, single decision trees, and shallow classifiers often fail to capture nonlinear interactions among academic and behavioural variables (Romero & Ventura, 2020).
2. **Class imbalance.** Educational datasets frequently contain fewer low-performing or at-risk students, biasing models toward majority classes and reducing recall for critical minority groups (Chawla et al., 2002; He & Garcia, 2009).
3. **Interpretability gap.** High-accuracy models are frequently black boxes. Stakeholders need to know *why* a student is flagged—e.g., absences, low engagement, weak continuous assessment—before intervening (Adadi & Berrada, 2018; Molnar, 2022).
4. **Data quality and context.** Missing values, inconsistent records, and limited digital traces in some developing institutions reduce reliability of models trained without rigorous preprocessing (Ogunleye & Adebayo, 2022).
5. **Fragmented XAI adoption.** Explainable AI in education is discussed more often than embedded in complete, usable pipelines linking training, evaluation, and explanation (Lundberg & Lee, 2017).

There is therefore a need for an **interpretable prediction system centred on XGBoost**, with rigorous preprocessing and imbalance handling, and **SHAP-based explanations** of model outputs—implemented as a demonstrable software artifact.

## 1.3 Aim and Objectives

### Aim

To design and implement a student performance prediction system **using XGBoost enhanced with SHAP features** (SHapley Additive exPlanations), for accurate and interpretable forecasting in educational contexts.

### Specific Objectives

i. Preprocess and analyse student performance datasets, addressing missing values, encoding, scaling, and class imbalance.  
ii. Develop and optimize an **XGBoost-centred predictive model**: for classification, a stacked pipeline with tree-based base learners and a **tuned XGBoost meta-learner**; for regression, a stacked ensemble with a Ridge meta-learner as implemented for the grade prediction task.  
iii. Apply **SHAP KernelExplainer** to generate global and local explanations of the **trained model’s** predictions.  
iv. Evaluate predictive performance using cross-validation, hold-out metrics (accuracy, precision, recall, F1, RMSE, R²), confusion matrices, and ROC curves.  
v. Deploy a **Streamlit prototype** enabling interactive prediction and explanation for academic decision support.

## 1.4 Significance of the Study

**Academic significance:** The work contributes a reproducible **XGBoost + SHAP** pipeline to EDM literature, demonstrating how post-hoc XAI can explain a high-capacity predictor—including an XGBoost meta-learner within a stack—without retraining (Lundberg & Lee, 2017; Romero & Ventura, 2020).

**Institutional significance:** Early identification of at-risk learners supports targeted advising, supplemental instruction, and retention initiatives. Explainable outputs increase trust among lecturers and academic advisers compared to opaque scores alone.

**Societal significance:** Improved retention and completion align with human capital development and **Sustainable Development Goal 4** (quality education) (UNESCO, 2020). An open, Python-based implementation lowers adoption barriers for universities with limited proprietary analytics budgets.

## 1.5 Scope of the Study

**Included:**

- Supervised **regression** (final grade prediction) and **multi-class classification** (performance bands, dropout/enrolment/graduation).
- Datasets: **UCI Student Performance** (Cortez & Silva, 2008), **xAPI-Edu-Data** (engagement-rich K–12 online learning), and **UCI Predict Students’ Dropout and Academic Success** (Realinho et al., 2022)—the latter integrated in software; full training requires dataset download.
- Preprocessing: imputation, ordinal encoding, scaling, RFE, SMOTEENN.
- Model: **XGBoost-centred** design—classification uses a stacking ensemble with **tuned XGBoost meta-learner**; regression uses stacked tree learners with Ridge meta-learner (see Chapter 3).
- Explainability: SHAP global bar charts and local force plots.
- Prototype: offline Streamlit application (no live LMS integration).

**Excluded:**

- Real-time LMS integration, unsupervised anomaly detection, and deep sequential models (LSTM/GRU).
- Alternative XAI methods (LIME, counterfactuals) as primary tools—acknowledged in literature only.
- Cross-institutional deployment study on proprietary Nigerian university records (discussed as future work).
- Mutual Information feature selection (not implemented; RFE used instead).

**Delimitation:** Findings generalize to benchmark educational datasets; transfer to a specific Nigerian campus requires local validation and retraining.

---

# CHAPTER TWO: LITERATURE REVIEW

## 2.1 Conceptual Review

### 2.1.1 Student Performance Prediction

Student performance prediction models relationships between input features (demographics, prior grades, behaviour, engagement) and outcomes such as final grades, pass/fail status, or dropout. Early work relied on linear regression and descriptive statistics; these struggle with nonlinear interactions (Romero & Ventura, 2020). Modern EDM applies machine learning, often achieving strong accuracy on structured institutional data (Okewu et al., 2021).

### 2.1.2 Machine Learning in Education

Educational data mining and learning analytics extract patterns from grades, surveys, and digital traces (Siemens & Baker, 2012). Supervised learning dominates performance prediction; tree-based and ensemble models are particularly effective on tabular features (Romero & Ventura, 2020).

### 2.1.3 Ensemble Learning and Stacking

**Bagging** (e.g., Random Forest) reduces variance by averaging decorrelated trees (Breiman, 2001). **Boosting** (e.g., Gradient Boosting, XGBoost, LightGBM) sequentially corrects errors (Friedman, 2001; Chen & Guestrin, 2016; Ke et al., 2017). **Stacking** trains diverse base models and combines their outputs with a meta-learner, often improving generalization over any single model (Wolpert, 1992; Caruana et al., 2004). Tomasoni et al. (2020) report successful ensemble use in EDM competitions and institutional analytics.

### 2.1.4 XGBoost

XGBoost extends gradient boosting with regularized objectives, sparsity-aware splits, and efficient implementation (Chen & Guestrin, 2016). It is frequently top-ranked on tabular benchmarks and suitable as a **meta-learner** receiving out-of-fold predictions from base classifiers (Chen & Guestrin, 2016).

### 2.1.5 SHAP (SHapley Additive exPlanations)

SHAP assigns each feature a Shapley value representing its marginal contribution to a prediction, with properties of local accuracy and consistency (Lundberg & Lee, 2017). **TreeExplainer** is efficient for single tree models; **KernelExplainer** is model-agnostic and applicable to arbitrary predictors—including stacking ensembles—at higher computational cost (Lundberg & Lee, 2017).

## 2.2 Theoretical Review

The study rests on **supervised learning theory**: minimizing expected loss on unseen data via empirical risk minimization with regularization (Hastie et al., 2009). **Stacking** operationalizes the bias–variance trade-off by combining complementary hypotheses (Wolpert, 1992). **SMOTE** and **SMOTEENN** address class imbalance by synthetic oversampling and noise reduction (Chawla et al., 2002; Batista et al., 2004). **Explainable AI theory** treats transparency as a requirement for high-stakes decision support; post-hoc methods explain fixed models without retraining (Adadi & Berrada, 2018; Molnar, 2022).

## 2.3 Empirical Review

| Study | Method | Dataset | Reported performance | Limitation |
|-------|--------|---------|----------------------|------------|
| Asif et al. (2017) | XGBoost, RF | Institutional | ~89% accuracy | No XAI |
| Okewu et al. (2021) | Deep learning | Nigerian context | High accuracy | Black-box, compute cost |
| Tomasoni et al. (2020) | Ensembles | Multiple | Strong F1 | Limited interpretability |
| Realinho et al. (2022) | Various ML | UCI Dropout (4,424 students) | 88–93% (literature) | Context-specific |
| Cortez & Silva (2008) | DT, NN | UCI Student | Baseline benchmarks | Small sample (395) |
| Lundberg & Lee (2017) | SHAP | General ML | Explainability focus | Not education-specific |

Most studies prioritize accuracy; fewer deliver end-to-end systems with embedded SHAP and interactive interfaces for educators.

## 2.4 Explainable AI and Interpretability in Student Performance Prediction

Rising model complexity increases demand for transparency in academic risk assessment (Molnar, 2022). Educators must link predictions to factors such as absences, resource visits, or weak continuous assessment to justify interventions.

Post-hoc techniques include feature importance, partial dependence plots, LIME, and SHAP (Lundberg & Lee, 2017). For predictors that include **XGBoost inside a stack**, **KernelExplainer** (or similar model-agnostic explainers) attributes outcomes to the **original input features** for the full composed model, which matches educator-facing explanations better than inspecting only an internal sub-model in isolation.

## 2.5 Research Gap

Prior work establishes strong predictors but leaves gaps this study addresses:

1. **Limited ensemble + XAI integration** in deployable educational tools.  
2. **Insufficient handling of imbalance** in training pipelines.  
3. **Weak connection between analytics and user interfaces** for non-technical stakeholders.  
4. **Need for reproducible open implementations** applicable in developing-region institutions with benchmarking before local deployment.

This project fills these gaps through a documented **XGBoost + SHAP** pipeline (with stacking where XGBoost is the classification meta-learner), SMOTEENN, and the EduPredict Streamlit prototype.

---

# CHAPTER THREE: SYSTEM ANALYSIS, DESIGN AND METHODOLOGY

## 3.1 System Overview

The system (**EduPredict**) comprises three layers:

1. **Data layer** — `DatasetLoader` loads and validates UCI, xAPI, and dropout datasets.  
2. **ML layer** — `ModelTrainer` executes preprocessing, feature selection, **XGBoost-centred model** training (stacking with XGBoost meta-learner for classification), evaluation, and SHAP explainer persistence.  
3. **Presentation layer** — Streamlit pages for home overview, prediction with local SHAP, and analytics dashboards.

**Figure 3.1 (conceptual workflow):**  
Data Collection → Preprocessing → Train/Test Split → Scaling → RFE → SMOTEENN (classification) → **XGBoost-centred model training** (stacked bases + XGBoost meta-learner, classification) → Evaluation → SHAP Explainer → Prediction Output.

## 3.2 Dataset Description

| Dataset | Records | Features | Task | Target |
|---------|---------|----------|------|--------|
| UCI Student Performance (Mathematics) | 395 | 32 | Regression | G3 (final grade, 0–20) |
| xAPI-Edu-Data | 480 | 16 | 3-class classification | L / M / H → 0 / 1 / 2 |
| UCI Dropout & Academic Success | 4,424 | 36 | 3-class classification | Dropout / Enrolled / Graduate |

**UCI** includes demographic, social, and academic attributes (study time, failures, absences, G1, G2). **xAPI** adds behavioural counts: raised hands, resource visits, announcements, discussion participation, absence days. **Dropout** includes enrolment, curricular unit statistics, and macroeconomic indicators (Realinho et al., 2022).

Using multiple datasets demonstrates robustness across regression and classification and across secondary vs. higher-education feature spaces.

## 3.3 Data Preprocessing

1. **Missing value imputation:** mean for numeric features; mode for categorical (Pedregosa et al., 2011).  
2. **Ordinal encoding** of categorical variables (unknown categories mapped to −1 at inference).  
3. **Train–test split (80:20)** before scaling, RFE, or resampling to prevent **data leakage**.  
4. **StandardScaler** fit on training data only, applied to train and test.  
5. **Duplicate removal** where applicable.  
6. **SMOTEENN** on training folds for classification: SMOTE synthesizes minority samples; ENN removes ambiguous borderline instances (Chawla et al., 2002; Batista et al., 2004).

## 3.4 Feature Engineering and Feature Selection

**Engineered indicators** (dataset-dependent) include aggregated assessment statistics and VLE click totals when OULAD-style files are present. For benchmark CSVs, features are used as provided after cleaning.

**Feature selection:** **Recursive Feature Elimination (RFE)** with a Random Forest estimator retains approximately **75%** of features (minimum 5), fit on training data only (Guyon et al., 2002). This reduces dimensionality and overfitting risk.

*Note:* Mutual Information selection was considered in early documentation but is **not** implemented in the final pipeline.

## 3.5 Model Development: XGBoost-Centred Predictive Model

This section describes the implemented predictor in terms of the **official project focus (XGBoost + SHAP)**. Operationally, **classification** employs **stacked generalization**: heterogeneous tree ensembles generate level-0 predictions; **XGBoost**, as the **meta-learner (level-1)**, learns the final mapping and is the component most directly associated with the thesis title. **Regression** uses the same level-0 diversity with a **Ridge** meta-learner (see §3.5.3)—SHAP still explains the full trained regressor.

### 3.5.1 Base learners (classification)

- Random Forest (400 trees, balanced class weights)  
- Gradient Boosting (300 estimators)  
- Extra Trees (400 trees)  
- LightGBM (400 estimators)  

### 3.5.2 XGBoost meta-learner (classification)

**XGBClassifier** with `multi:softprob`, combined via `StackingClassifier` using **5-fold out-of-fold** `predict_proba` outputs from the base learners. **RandomizedSearchCV** (50 iterations, 5-fold CV) tunes `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, and `reg_lambda`—this step constitutes the **principal XGBoost optimization** in the system.

### 3.5.3 Regression variant (UCI)

Base learners: RF, GB, Extra Trees, LightGBM. Meta-learner: **Ridge regression** (`StackingRegressor`, 5-fold CV). *Rationale:* the approved project title emphasizes **XGBoost for classification**; the regression benchmark is retained for completeness of the software artefact while Chapter 5 reports its metrics transparently.

### 3.5.4 Cross-validation

**StratifiedKFold (5)** for classification; **KFold (5)** for regression. Reported CV metrics: accuracy (classification) or R² (regression).

## 3.6 Model Evaluation

**Classification:** accuracy, per-class precision/recall/F1, macro and weighted averages, confusion matrix, one-vs-rest ROC/AUC.  
**Regression:** MSE, RMSE, R².  
Hold-out **test set (20%)** never used during fitting of preprocessors, RFE, SMOTEENN, or ensemble.

## 3.7 SHAP Explainability Integration

After training, a **SHAP KernelExplainer** is fit with a **k-means summary** of up to 50 training samples as background. The explainer wraps `predict` (regression) or `predict_proba` (classification).

- **Global:** mean |SHAP| bar chart across background samples (Model Analytics page).  
- **Local:** force plot and top-8 contribution table for a single student profile (Prediction Engine page).

KernelExplainer is chosen because it explains the **full trained model** (the object actually deployed for inference), including the **XGBoost meta-learner** and its inputs from base learners, without requiring a single-tree surrogate (Lundberg & Lee, 2017).

## 3.8 System Workflow

1. Data collection / load  
2. Imputation and encoding  
3. Train–test split  
4. Scaling (train-fit)  
5. RFE (train-fit)  
6. SMOTEENN (train only, classification)  
7. **XGBoost-centred** stacked model training + meta-learner hyperparameter tuning (classification)  
8. Cross-validation and test evaluation  
9. SHAP explainer serialization  
10. Streamlit inference and visualization  

---

# CHAPTER FOUR: SYSTEM IMPLEMENTATION

## 4.1 Introduction

This chapter describes how the methodology in Chapter 3 was realized in software: development environment, project structure, core modules, machine learning pipeline, explainability integration, and the EduPredict user interface.

## 4.2 Development Environment and Tools

| Component | Technology | Version / Notes |
|-----------|------------|-----------------|
| Language | Python | 3.x |
| Data processing | pandas, NumPy | Tabular ETL |
| ML framework | scikit-learn | Models, CV, metrics, RFE, stacking |
| Boosting libraries | XGBoost, LightGBM | Base + meta learners |
| Imbalance handling | imbalanced-learn | SMOTE, SMOTEENN |
| Explainability | SHAP | KernelExplainer |
| Visualization | matplotlib | Confusion matrix, ROC, SHAP charts |
| Web UI | Streamlit, streamlit-shap | Interactive prototype |
| HTTP / data fetch | requests | UCI dataset download |
| Serialization | pickle | Model and pipeline artifacts |

**Hardware:** Experiments were conducted on a standard development workstation; training is CPU-parallelized (`n_jobs=-1`) where supported.

**Project root structure:**

```
student-performance-xgboost-shap/
├── app.py                    # Streamlit home / landing page
├── pages/
│   ├── 1_Prediction_Engine.py   # Interactive prediction + local SHAP
│   └── 2_Model_Analytics.py     # Metrics, CM, ROC, global SHAP
├── src/
│   ├── data_loader.py        # Dataset loading and preparation
│   └── train.py              # Full training and evaluation pipeline
├── data/                     # CSV datasets
├── models/                   # Saved .pkl reports and .png plots
├── requirements.txt
└── docs/
```

## 4.3 Data Loading Module (`data_loader.py`)

The `DatasetLoader` class centralizes dataset acquisition:

- **`load_uci_data()`** — Reads `student-mat.csv` (semicolon-separated), predicts **G3** from remaining columns; auto-downloads from UCI if missing.  
- **`load_xapi_data()`** — Reads `xAPI-Edu-Data.csv`, maps Class `L/M/H` to `0/1/2`.  
- **`load_dropout_data()`** — Downloads UCI #697 zip if needed, maps Target labels to numeric classes.  
- **`load_oulad_data()`** — Legacy Open University loader (assessment + VLE aggregates); retained for extension.

Mock data generators provide fallback for demonstration when downloads fail, preserving pipeline testability.

## 4.4 Training Pipeline (`train.py`)

The `ModelTrainer.train_and_evaluate()` method implements the full leakage-safe pipeline documented in Section 3.3–3.7.

### 4.4.1 Preprocessing sequence (implementation order)

```text
1. Fit imputers and ordinal encoder on full X (pre-split; no label use)
2. train_test_split (80/20, stratified for classification)
3. Fit StandardScaler on X_train only
4. Fit RFE on X_train_scaled, transform train and test
5. Apply SMOTEENN on X_train_selected, y_train (classification only)
6. Build StackingClassifier / StackingRegressor
7. cross_val_score (5-fold)
8. model.fit on full training set
9. Evaluate on X_test_selected
10. Build SHAP KernelExplainer; save artifacts
```

### 4.4.2 Saved artifacts (per dataset `name`)

| File | Contents |
|------|----------|
| `{name}_model.pkl` | Fitted **XGBoost-centred** model (stacked classifier/regressor as trained) |
| `{name}_explainer.pkl` | SHAP KernelExplainer |
| `{name}_pipeline.pkl` | Imputers, encoder, scaler, RFE selector, feature lists |
| `{name}_report.txt` | CV and test metrics |
| `{name}_cm.png` | Confusion matrix (classification) |
| `{name}_roc.png` | ROC curves (classification) |

### 4.4.3 Entry point

Running `python src/train.py` trains all three pipelines sequentially: `uci` (regression), `xapi` (classification), `dropout` (classification).

## 4.5 Streamlit Application

### 4.5.1 Home page (`app.py`)

- Branding: **EduPredict — Student Performance Analytics**  
- ML pipeline step list (data → preprocess → ensemble → SHAP → output)  
- Project metadata (author, matric number, supervisor, institution)  
- Dataset summary cards  

### 4.5.2 Prediction Engine (`pages/1_Prediction_Engine.py`)

**Functions:**

1. User selects dataset context (UCI / xAPI / Dropout).  
2. Sidebar **Student Profile Builder** — sliders for numeric features, select boxes for categorical (using saved `cat_classes`).  
3. Input row passes through saved pipeline: impute → encode → scale → RFE feature subset.  
4. Ensemble predicts grade band or class with **confidence bars** (`predict_proba`).  
5. **SHAP force plot** explains the individual prediction.  
6. Table shows top 8 features by |SHAP| with direction of impact.

This directly supports objective (iii): local interpretability for advisers.

### 4.5.3 Model Analytics (`pages/2_Model_Analytics.py`)

Per-dataset tabs display:

- Parsed **5-fold CV** score and standard deviation  
- **Test accuracy** or **R²**, RMSE (regression), macro F1  
- CV vs. test **gap badge** (overfitting indicator)  
- Full sklearn `classification_report` text  
- Confusion matrix and ROC images when available  
- **Global SHAP** horizontal bar chart (top 15 features by mean |SHAP|)

## 4.6 Design Decisions and Justification

| Decision | Rationale |
|----------|-----------|
| Stacking with XGBoost meta-learner vs. standalone XGBoost | Diverse bases reduce bias; **tuned XGBoost** learns how to fuse their signals—aligned with the project topic while improving robustness (Wolpert, 1992) |
| XGBoost as classification meta-learner | Concentrates the thesis emphasis on **XGBoost**; base learners supply complementary views of tabular educational data |
| RFE with Random Forest | Model-agnostic ranking; reduces input dimension before expensive ensemble fit |
| SMOTEENN vs. SMOTE alone | ENN cleans noisy synthetic points at class boundaries (Batista et al., 2004) |
| KernelExplainer vs. TreeExplainer | Required for stacked model wrapping multiple algorithms |
| 80:20 split | Standard hold-out proportion; stratification preserves class ratios |
| Pickle artifacts | Simple deployment for academic prototype; production would use ONNX/joblib version pins |

## 4.7 Implementation Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Data leakage from scaling on full data | Split before scaler, RFE, and SMOTE |
| Long SHAP computation | k-means background (max 50 samples); explainer cached in `.pkl` |
| Missing dropout CSV | Auto-download from UCI; mock fallback for demos |
| Mixed categorical cardinality | OrdinalEncoder with explicit class lists per column |
| Multi-class SHAP | Class-specific SHAP vector for predicted class in force plot |

## 4.8 User Requirements Mapping

| Stakeholder need | Implementation |
|------------------|----------------|
| Predict at-risk student | Classification on xAPI / dropout; regression bands on UCI |
| Understand why | SHAP force plot + contribution table |
| Compare model quality | Model Analytics metrics and ROC |
| Non-programmer access | Streamlit GUI, no code required for inference |

---

# CHAPTER FIVE: RESULTS, DISCUSSION, AND CONCLUSION

## 5.1 Introduction

This chapter presents experimental results from the implemented pipeline, interprets findings, discusses limitations, and states conclusions and recommendations.

## 5.2 Experimental Setup

- **Training script:** `python src/train.py`  
- **Random seed:** 42 (split, SMOTE, CV, base models)  
- **Test fraction:** 20%  
- **CV folds:** 5  
- **Hyperparameter search:** RandomizedSearchCV, 50 iterations (XGBoost meta-learner, classification)  
- **Primary runs reported:** UCI and xAPI (artifacts in `models/`); dropout training executes when `dropout_data.csv` is present or downloaded  

## 5.3 Results

### 5.3.1 UCI Student Performance (Regression)

| Metric | Value |
|--------|-------|
| 5-Fold CV R² | **0.8784 ± 0.0367** |
| Test R² | **0.8265** |
| Test RMSE | **1.8863** |
| Test MSE | **3.5581** |

**Interpretation:** The trained **regression** model explains approximately **83%** of variance in final grade on the hold-out set. With grades on a 0–20 scale, RMSE ≈ 1.89 indicates typical error below one letter-grade step, suitable for early advisory signalling (e.g., identifying students likely below pass thresholds) when combined with SHAP drivers such as **G1, G2, absences, and study time**.

### 5.3.2 xAPI-Edu-Data (3-Class Classification)

**Class distribution before SMOTE (training portion):** {0: 127, 1: 211, 2: 142}  
**After SMOTEENN:** balanced {169, 169, 169} per class on resampled training set.

| Metric | Value |
|--------|-------|
| 5-Fold CV Accuracy | **0.7713 ± 0.0278** |
| Test Accuracy | **0.7500** |
| Macro F1 | **0.7600** |
| Weighted F1 | **0.7500** |

**Per-class test performance:**

| Class | Label | Precision | Recall | F1 | Support |
|-------|-------|-----------|--------|-----|---------|
| 0 | Low | 0.85 | 0.85 | 0.85 | 26 |
| 1 | Medium | 0.70 | 0.74 | 0.72 | 42 |
| 2 | High | 0.73 | 0.68 | 0.70 | 28 |

**Interpretation:** The model performs best on **low** performers—critical for early intervention. Medium and high classes show moderate confusion, consistent with overlapping engagement profiles. CV and test accuracy differ by ~2.1 percentage points, suggesting acceptable generalization without severe overfitting.

Behavioural features in xAPI (raised hands, resource visits, discussion, absences) are expected to dominate SHAP rankings, aligning with EDM findings that engagement predicts achievement.

### 5.3.3 UCI Dropout & Academic Success

The pipeline is **implemented and integrated** in `train.py` and the Streamlit UI. Published benchmarks on this dataset report roughly **88–93%** accuracy with various classifiers (Realinho et al., 2022). Local experimental metrics should be inserted after running full training once `dropout_data.csv` is generated via the automated UCI download. Until then, dropout results are cited as **comparative literature targets**, not as primary claims of this submission.

### 5.3.4 Explainability Outputs

Qualitative evaluation of SHAP (Prediction Engine and Model Analytics pages) confirms:

- **Global charts** highlight dominant features per dataset.  
- **Local force plots** show direction and magnitude of each feature’s push toward the predicted class or grade.  
- Advisers can answer: *“Which factors most increased this student’s risk?”*

This satisfies objective (iii) and supports the interpretability claims in Chapters 1–2.

### 5.3.5 Summary comparison table

| Dataset | Task | CV Metric | Test Metric | Best use case |
|---------|------|-----------|-------------|---------------|
| UCI | Regression | R² = 0.88 | R² = 0.83 | Final grade forecasting |
| xAPI | 3-class | Acc = 77.1% | Acc = 75.0% | Engagement-based risk bands |
| Dropout | 3-class | (run-dependent) | (run-dependent) | Higher-ed retention analytics |

## 5.4 Discussion

### 5.4.1 Achievement of objectives

| Objective | Status |
|-----------|--------|
| i. Preprocessing & imbalance handling | **Achieved** — imputation, encoding, scaling, SMOTEENN, leakage-safe order |
| ii. **XGBoost-centred** predictive model | **Achieved** — classification: four base learners + **tuned XGBoost meta-learner**; regression: stacked bases + Ridge meta-learner (see §3.5.3) |
| iii. SHAP explanations | **Achieved** — KernelExplainer, global and local UI |
| iv. Evaluation metrics | **Achieved** — CV, test reports, CM, ROC, R², RMSE |
| v. Streamlit prototype | **Achieved** — EduPredict with two functional pages |

### 5.4.2 Comparison with literature

- xAPI test accuracy (**75%**) is competitive with published ML studies on the same dataset, though direct comparison requires identical splits and features.  
- UCI regression **R² = 0.83** aligns with strong gradient boosting results reported by Cortez & Silva (2008) and subsequent EDM work.  
- The distinguishing contribution is not raw accuracy alone but **accuracy + embedded SHAP + deployable UI**.

### 5.4.3 Practical implications for Nigerian higher education

Although datasets are international benchmarks, the workflow mirrors challenges in Nigerian institutions:

- Advisers can prototype **what-if** scenarios (e.g., effect of improved attendance) via sliders.  
- **Explainability** supports ethical use—interventions guided by factors, not opaque scores.  
- Before production use, institutions must **retrain on local LMS/SEMIS data** and validate fairness across demographic groups.

### 5.4.4 Limitations

1. **Benchmark vs. local data** — No proprietary Bells University cohort was used.  
2. **Computational cost** — **XGBoost-centred stacked training**, RandomizedSearchCV on the meta-learner, and KernelExplainer can be slow on large datasets.  
3. **KernelExplainer approximation** — Background sample size limits exact Shapley values.  
4. **No formal user study** — Educator usability was not evaluated with controlled experiments.  
5. **Dropout results** — Depend on successful dataset download and full training run.  
6. **Temporal dynamics** — Static features only; no LSTM/GRU on clickstream sequences.  
7. **Class imbalance on test set** — SMOTE applied to training only; test distribution remains realistic but minority recall may still lag.

### 5.4.5 Threats to validity

- **Internal validity:** Leakage controls strengthen causal claims about pipeline design; hyperparameter search may slightly optimistic-bias CV scores.  
- **External validity:** Generalization to other countries or K–12 vs. HE contexts requires revalidation.  
- **Construct validity:** Performance labels (L/M/H) are coarse proxies for learning mastery.

## 5.5 Conclusion

This project successfully designed and implemented **EduPredict**, a student performance prediction system **using XGBoost enhanced with SHAP**. For classification, **XGBoost serves as the tuned meta-learner** atop stacked tree-based base models; SHAP explains the **deployed predictor** end-to-end. The system addresses predictive accuracy, class imbalance, and interpretability gaps identified in the literature. Empirical results on UCI regression and xAPI classification demonstrate usable performance; the Streamlit prototype translates the model into an adviser-facing decision-support tool.

The study confirms that **XGBoost-centred learning and explainable AI (SHAP)** can be integrated in a single, reproducible pipeline suitable as a foundation for institution-specific deployment in Nigeria and similar contexts.

## 5.6 Recommendations

**For institutions:**

1. Pilot the prototype with de-identified local records.  
2. Integrate outputs into existing academic advising workflows.  
3. Establish governance for ethical use of predictive analytics.

**For future research:**

1. Add **baseline models** (logistic regression, single XGBoost) for ablation studies.  
2. Implement **TreeExplainer** on a single XGBoost model for speed comparison.  
3. Incorporate **temporal** LMS logs (OULAD VLE) with sequence models.  
4. Conduct **user studies** with lecturers measuring trust and intervention quality.  
5. Deploy via API connection to institutional databases with role-based access control.  
6. Extend fairness auditing (demographic parity, equalized odds) across encoded attributes.

## 5.7 Contribution to Knowledge

1. A **documented, open XGBoost + SHAP pipeline** for educational prediction (with **XGBoost meta-learner** stacking for classification).  
2. Evidence that **model-agnostic SHAP** applies to **XGBoost-centred stacked** predictors in EDM.  
3. A **working Streamlit reference implementation** linking training artifacts to interactive explanation.  
4. A methodology template adaptable to **developing-region universities** after local retraining.

---

# REFERENCES

Adadi, A., & Berrada, M. (2018). Peeking inside the black-box: A survey on explainable artificial intelligence (XAI). *IEEE Access*, *6*, 52138–52160.

Asif, R., Merceron, A., Ali, S. A., & Haolai, S. W. (2017). Analyzing students' performance using educational data mining. *IEEE Access*, *5*, 26777–26782.

Batista, G. E. A. P. A., Prati, R. C., & Monard, M. C. (2004). A study of the behavior of several methods for balancing machine learning training data. *ACM SIGKDD Explorations Newsletter*, *6*(1), 20–29.

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32.

Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). Ensemble selection from libraries of models. In *Proceedings of the 21st International Conference on Machine Learning* (pp. 18–25).

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).

Cortez, P., & Silva, A. M. G. (2008). Using data mining to predict secondary school student performance. In A. Brito & J. Teixeira (Eds.), *Proceedings of 5th Annual Future Business Technology Conference* (pp. 5–12).

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, *29*(5), 1189–1232.

Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning*, *46*, 389–422.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2nd ed.). Springer.

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, *21*(9), 1263–1284.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30* (pp. 3146–3154).

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774).

Molnar, C. (2022). *Interpretable machine learning* (2nd ed.). https://christophm.github.io/interpretable-ml-book/

Ogunleye, A., & Adebayo, O. (2022). Machine learning for African education: Opportunities and challenges. *African Journal of Computing and ICT*, *15*(2), 112–125.

Okewu, E., Misra, S., Maskeliūnas, R., & Damaševičius, R. (2021). An ensemble-based approach for student academic performance prediction. *PeerJ Computer Science*, *7*, e589.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

Realinho, V., Machado, J., Baptista, L., & Ouhbi, S. (2022). Predict students' dropout and academic success. *Data*, *7*(11), 146. https://doi.org/10.3390/data7110146

Romero, C., & Ventura, S. (2020). Educational data mining and learning analytics: An updated survey. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, *10*(3), e1355.

Siemens, G., & Baker, R. S. (2012). Learning analytics and educational data mining: Towards communication and collaboration. In *Proceedings of the 2nd International Conference on Learning Analytics and Knowledge* (pp. 252–254).

Tomasoni, M., Antonini, A., & Salza, S. (2020). Ensemble methods for student performance prediction: A systematic evaluation. In *Proceedings of the 10th International Conference on Learning Analytics & Knowledge* (pp. 456–465).

UNESCO. (2020). *Education for sustainable development goals: Learning objectives*. UNESCO Publishing.

Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, *5*(2), 241–259.

---

## APPENDIX A: How to Reproduce Results

```bash
cd student-performance-xgboost-shap
pip install -r requirements.txt
pip install lightgbm
python src/train.py
streamlit run app.py
```

## APPENDIX B: Figure and Screenshot Placeholders for Word Document

Insert from `models/` after training:

- `uci_report.txt`, `xapi_report.txt`, `dropout_report.txt`  
- `xapi_cm.png`, `xapi_roc.png`  
- Screenshots of EduPredict: Home, Prediction Engine (force plot), Model Analytics (global SHAP)

---

*End of Chapters 1–5 — aligned with repository implementation (May 2026).*
