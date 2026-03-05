# Hospital-30-day-Readmission
# 🏥 Hospital Readmission Prediction (30-Day)

A machine learning pipeline in R to predict whether a patient will be readmitted to hospital within **30 days** of discharge. The project covers end-to-end ML development: data preprocessing, model training, performance evaluation, explainability analysis (SHAP), and fairness assessment across age groups.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Explainability](#explainability)
- [Fairness Evaluation](#fairness-evaluation)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Hospital readmissions within 30 days are a major quality indicator and cost driver in healthcare systems. This project builds a binary classification pipeline to identify high-risk patients at the point of discharge, enabling targeted interventions.

**Target variable:** `readmitted_30_days` (`Yes` / `No`)

**Key highlights:**
- Stratified train/test split with reproducible seeds
- Two competing models: Elastic Net (GLMNet) and Random Forest (ranger)
- 5-fold cross-validation with ROC-AUC as the primary metric
- DALEX-powered model explainability (SHAP, Breakdown, Ceteris Paribus)
- Fairness audit across patient age groups

---

## Dataset

The pipeline uses the `hospital_readmissions_30k` dataset (~30,000 records), which is downsampled to **5,000 stratified records** for efficient training and evaluation.

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Patient age (binned into groups) |
| `gender` | Categorical | Patient gender |
| `diabetes` | Categorical | Diabetes diagnosis flag |
| `hypertension` | Categorical | Hypertension diagnosis flag |
| `blood_pressure` | Categorical | Blood pressure classification |
| `length_of_stay` | Numeric | Inpatient days |
| `discharge_destination` | Categorical | Post-discharge setting |
| `readmitted_30_days` | Binary Target | Whether patient was readmitted within 30 days |

> **Note:** `patient_id` is dropped prior to modelling to prevent data leakage.

---

## Project Structure

```
hospital-readmission/
│
├── FINAL-_HOSPITAL_READMISSION.R   # Main analysis script
├── README.md                        # Project documentation
└── CONTRIBUTING.md                  # Contribution guidelines
```

---

## Installation

### Prerequisites

- R (≥ 4.1.0)
- RStudio (recommended)

### Install Required Packages

Run the following at the top of the script or in your R console:

```r
install.packages(c(
  "tidyverse", "caret", "pROC", "PRROC",
  "ranger", "e1071", "glmnet", "DALEX", "ingredients", "xgboost"
))
```

### Clone the Repository

```bash
git clone https://github.com/your-username/hospital-readmission.git
cd hospital-readmission
```

---

## Usage

1. Ensure the dataset `hospital_readmissions_30k` is loaded into your R environment (or update the data loading step to point to your local file path).
2. Open `FINAL-_HOSPITAL_READMISSION.R` in RStudio.
3. Run the script top-to-bottom, or source it directly:

```r
source("FINAL-_HOSPITAL_READMISSION.R")
```

Outputs produced during execution:
- Model comparison table (`results`)
- SHAP summary table (`shap_summary`)
- Breakdown plot for the highest-risk patient
- Ceteris Paribus profile for `length_of_stay`
- Fairness metrics by age group (`fairness_age`, `fairness_gaps`)

---

## Methodology

### 1. Preprocessing
- Dropped `patient_id` to avoid leakage
- Removed rows with missing values (`drop_na()`)
- Encoded categorical variables as factors
- Engineered `age_group` from continuous age (bins: ≤39, 40–59, 60–79, 80+)
- Applied one-hot dummy encoding via `dummyVars()` with full rank encoding
- Removed near-zero variance predictors via `nearZeroVar()`

### 2. Train / Test Split
- Stratified 80/20 split using `createDataPartition()` on the target variable
- `set.seed(123)` used throughout for reproducibility

### 3. Model Training
Both models were trained using **5-fold cross-validation** with `ROC` as the optimisation metric:

| Model | Method | Preprocessing |
|---|---|---|
| Elastic Net | `glmnet` | Centre + Scale |
| Random Forest | `ranger` | None required |

### 4. Model Selection
The best-performing model is selected automatically based on the highest **ROC-AUC** on the test set.

---

## Results

Model performance is evaluated using:
- **ROC-AUC** — primary metric
- **Accuracy** — secondary metric (threshold: 0.5)

Run `results` in R to view the comparison table after executing the script.

---

## Explainability

Model explainability is implemented using the **DALEX** and **ingredients** packages:

| Technique | Purpose |
|---|---|
| **SHAP** (`predict_parts`, type = `"shap"`) | Global feature importance averaged over test samples |
| **Break-Down** | Local explanation for the single highest-risk patient |
| **Ceteris Paribus** | How `length_of_stay` alone affects readmission probability |

---

## Fairness Evaluation

The model is audited for **age-group fairness** across four cohorts: `≤39`, `40–59`, `60–79`, `80+`.

Metrics computed per group:
- **Predicted Positive Rate** — proportion flagged as high-risk
- **TPR** (True Positive Rate / Recall) — sensitivity per group
- **FPR** (False Positive Rate) — specificity complement per group

Disparity gaps across groups are reported via `fairness_gaps` (TPR gap, FPR gap, Positive Rate gap).

---

## Dependencies

| Package | Version (min) | Purpose |
|---|---|---|
| `tidyverse` | 1.3.0 | Data wrangling & visualisation |
| `caret` | 6.0 | ML pipeline & cross-validation |
| `glmnet` | 4.0 | Elastic Net model |
| `ranger` | 0.14.0 | Random Forest model |
| `e1071` | 1.7 | SVM support (caret dependency) |
| `pROC` | 1.18 | ROC curve & AUC |
| `PRROC` | 1.3 | Precision-Recall curves |
| `xgboost` | 1.6 | XGBoost (available for extension) |
| `DALEX` | 2.4 | Model explainability framework |
| `ingredients` | 2.3 | SHAP & Ceteris Paribus profiles |

---

## Contributing

Contributions, issues, and feature requests are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

---

---

> **Disclaimer:** This project is intended for educational and research purposes only. It should not be used as the sole basis for clinical decision-making.
