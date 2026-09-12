# Creditworthiness Prediction System

An end-to-end machine learning system that predicts consumer creditworthiness and serves those predictions through an interactive web application. The project covers the full pipeline — data cleaning, feature engineering, model selection, hyperparameter tuning, and deployment behind a FastAPI application with user authentication and persistent storage.

**Repository:** https://github.com/dingyuquan/credit_score_system

---

## Motivation

Traditional credit review relies on slow, manual, and often inconsistent decisions. This project builds a transparent and automated credit-scoring system to help financial institutions reduce risk, shorten decision time, and make outcomes more consistent across applicants.

## Features

- **Modular Python codebase** organized into independent functional modules
- **OOP-based preprocessing** — `DataFrameProcessor` for cleaning, `FeatureEncoder` for feature engineering
- **Dedicated training & tuning module** (`ModelBuilder`) with randomized hyperparameter search
- **Feature ablation study** to select the smallest feature subset that preserves predictive power
- **Deployable artifact** — the tuned model is exported as a `.joblib` file loaded at application startup
- **Interactive web application** — registration/login, application submission, live scoring, and result display
- **Rating & interest decision logic** that converts a raw score into Rating A / B / C
- **SQLite persistence** for users and application records

## Tech Stack

| Layer | Tools |
|---|---|
| Modeling | scikit-learn, XGBoost, MLPClassifier (neural network) |
| Data | pandas, NumPy, Kaggle API |
| Serialization | joblib |
| Backend | FastAPI |
| Persistence | SQLAlchemy + SQLite |
| Frontend | HTML forms served by FastAPI |

## Dataset

[Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) (Kaggle), downloaded programmatically via the Kaggle API.

## Methodology

### 1. Exploratory Data Analysis
Distribution checks, histograms and box plots, summary statistics, correlation heatmaps, and pairwise plots. Skewed features, unexpected distribution spikes, duplicated categories, and anomalous points were flagged to guide cleaning decisions.

### 2. Data Cleaning
- Removed noisy symbols and inconsistent formats
- Converted string-encoded numbers to numeric types and standardized mixed-type columns
- Imputed missing values — median for numerical columns, mode for categorical
- Handled outliers by capping at reasonable percentiles, with manual review where needed

### 3. Feature Engineering
- One-hot encoding for categorical fields
- Parsed concatenated loan-type strings into boolean indicators
- Converted credit-age text (e.g. "5 Years and 3 Months") into total months
- Standardized / normalized continuous variables
- Enforced identical transformations across train, test, and inference

### 4. Feature Selection
For each model family, the model was first fit on the full feature set and feature importances were used to rank features. Models were then retrained incrementally on the top-*k* features (*k* = 5 … all), each evaluated with 5-fold cross-validated AUC. The selection rule kept the smallest subset with near-optimal performance:

```
choose the smallest k such that AUC(k) ≥ AUC_max − 0.005
```

This substantially reduced the feature set while retaining nearly all predictive power.

### 5. Model Optimization
Three model families were tuned independently with randomized search, each scored by cross-validated AUC:

| Model | Search space |
|---|---|
| Random Forest | `max_depth`, `n_estimators`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| XGBoost | `max_depth`, `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_lambda`, `reg_alpha` |
| Neural Network | `hidden_layer_sizes`, `activation`, `alpha`, `learning_rate_init`, `batch_size` |

Each tuned model was paired with its selected feature subset and evaluated on a held-out validation set; the highest validation AUC won.

## Results

Final performance on the held-out set:

| Model | Accuracy | AUC-ROC | F1-Score | Status |
|---|---|---|---|---|
| Random Forest | 0.79 | 0.9106 | 0.79 | Stable |
| **XGBoost** | **0.81** | **0.9299** | **0.81** | **Selected** |
| Neural Network | 0.72 | 0.8716 | 0.72 | Good potential |

Effect of hyperparameter tuning on accuracy:

| Model | Before tuning | After tuning |
|---|---|---|
| Random Forest | 0.75 | 0.79 |
| XGBoost | 0.72 | 0.81 |
| Neural Network | 0.69 | 0.72 |

**XGBoost** achieved the highest discriminative power and was selected as the final classifier.

## System Design

The application is a modular FastAPI service with the trained model wired directly into its business logic:

1. A pre-trained XGBoost model is loaded once at startup.
2. User input is one-hot encoded and scaled with the same transformations used in training.
3. Features are aligned to the model's training schema before inference.
4. The model returns a numerical credit score, which is mapped to **Rating A**, **Rating B**, or **Rating C** and a corresponding interest decision.
5. The application and its result are persisted to SQLite via SQLAlchemy.

**User flow:** Register → Log in → Submit application → Fill in financial information → View rating and decision

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/dingyuquan/credit_score_system.git
cd credit_score_system

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset (requires Kaggle API credentials in ~/.kaggle/kaggle.json)
#    Only needed if you want to retrain from scratch

# 5. Run the application
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000 in a browser.

> Adjust the module path in step 5 (`main:app`) and the dependency file name to match the repository layout.

## Known Challenges

- **Dirty data** — conflicting encodings, non-numeric financial fields, missing values in critical variables, and loan types stored as concatenated strings requiring custom parsing.
- **Engineering** — avoiding circular imports across modular OOP components, and keeping scaling/encoding/feature transformations identical between training and inference.
- **Workflow** — versioning datasets (small preprocessing changes shifted downstream results noticeably), reproducibility across computing environments, and iterating on feature subsets and hyperparameter search spaces.

## Roadmap

**Credit decision quality**
- Move from tree-based models toward deep learning architectures to capture complex non-linear patterns in high-dimensional data
- Replace the single-score approve/reject rule with a multi-stage decision engine combining scores, business rules, and dynamic strategies

**Operational capability**
- A real-time business dashboard giving risk managers and operations leads visibility into pipeline health, efficiency, and performance
- Workflow management (WFM) to track every application's status and drive it forward automatically under business logic

## Team

| Member | Contribution |
|---|---|
| Wenxuan Han (wh9772) | Feature engineering, baseline model construction, model evaluation, final model packaging |
| Xuan Xiang (xx3755) | Data exploration and analysis, hyperparameter tuning, model evaluation |
| Yuquan Ding (yd5966) | User-facing interface — front-end forms, authentication, database logic, and ML prediction integration |

## GenAI Contribution

Generative AI tools assisted with refining class structures and readability, debugging multi-file Python imports, generating the `.joblib` model export code, restructuring logic for a cleaner architecture, and polishing report language.
