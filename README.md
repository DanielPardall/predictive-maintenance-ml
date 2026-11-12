# Predictive Maintenance — Machine Learning Classification

**A notebook-based study of machine failure detection and failure-type classification using Support Vector Machines, neural networks and Random Forest.**

Developed for the Machine Learning / Aprendizagem Computacional academic project at the University of Coimbra (2023). The notebook combines exploratory data analysis, preprocessing experiments, class-imbalance handling, hyperparameter search and model comparison. Explanations in the notebook are in Portuguese.

## Objectives

The project investigates two supervised classification tasks:

| Task | Target | Goal |
| --- | --- | --- |
| Binary classification | `Target` | Classify an observation as failure or no failure |
| Multiclass classification | `Failure Type` | Identify the failure category, including no failure |

This is classification of recorded observations in a synthetic dataset. It is not a remaining-useful-life model or a validated system for forecasting future equipment failures.

## Dataset

The assignment uses the [Machine Predictive Maintenance Classification dataset on Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification).

The supplied notebook's saved outputs show **10,000 rows and 10 columns** in `predictive_maintenance.csv`:

- Identifiers: `UDI`, `Product ID`.
- Product category: `Type`.
- Measurements: `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
- Labels: `Target`, `Failure Type`.

The original binary target is strongly imbalanced: the saved counts show 9,661 no-failure observations and 339 failure observations.

**The CSV is not included with the supplied notebook.** Download the version containing the column names above and place `predictive_maintenance.csv` beside `Projeto_AC.ipynb` before execution. Related dataset variants can have different label columns and are not necessarily drop-in replacements.

## Workflow

### 1. Exploratory analysis

The notebook examines dataset shape, data types, missing values, identifier uniqueness, class distributions, feature distributions and relationships between the two labels. Visualizations use Matplotlib, Seaborn and Plotly.

### 2. Data preparation

The implementation:

- Removes nine rows labelled as failure in `Target` but `No Failure` in `Failure Type`.
- Removes 18 observations labelled `Random Failures`.
- Encodes product category and failure labels numerically.
- Removes identifier features from modelling.
- Compares scaling, oversampling, feature selection and combinations of these operations.

After the row removals, the saved outputs show **9,973 observations**. The multiclass task consequently uses five classes rather than the six in the original assignment:

| Encoded value | Class |
| --- | --- |
| 0 | No Failure |
| 1 | Power Failure |
| 2 | Tool Wear Failure |
| 3 | Overstrain Failure |
| 4 | Heat Dissipation Failure |

These removals are modelling decisions made in the notebook, not evidence that those observations should always be discarded in a real application.

### 3. Preprocessing experiments

Six experiment groups compare:

1. Baseline preprocessing.
2. Scaling with `RobustScaler` and `MinMaxScaler`.
3. SMOTE oversampling with different sampling strategies.
4. Feature selection.
5. Scaling and oversampling.
6. Scaling, oversampling and feature selection.

Random Forest is used in the preprocessing comparisons. The selected configurations combine scaling with SMOTE, using different sampling strategies for the binary and multiclass targets.

### 4. Models and evaluation

| Model | Implementation |
| --- | --- |
| Support Vector Machine | scikit-learn `SVC`, with searches over `C`, kernel and gamma |
| Neural network | Keras dense feed-forward networks trained with SGD |
| Random Forest | scikit-learn `RandomForestClassifier`, with searches over tree count, depth and criterion |

The notebook uses `GridSearchCV` for SVM and Random Forest experiments, a 70/30 split in the final modelling sections, and confusion matrices for comparison. Reported metrics include accuracy, macro F1, macro precision and macro recall. Macro averaging gives each class equal weight in the aggregate metric, which is useful when inspecting this imbalanced problem.

## Evaluation caveat

The final modelling section applies **scaling and SMOTE before the train/test split**. This allows information from the eventual test data to influence preprocessing and synthetic sample generation. The resulting scores should therefore be treated as exploratory notebook outputs, not unbiased estimates of performance on unseen data.

A corrected evaluation should split the original observations first, fit scaling and SMOTE only on the training data within each cross-validation fold, and evaluate once on an untouched test set. An `imbalanced-learn` pipeline can keep those operations inside the training folds.

Additional points to review before reproducing or extending the study:

- The Random Forest grid search in the multiclass section fits `X_train_target` and `y_train_target` instead of the multiclass training variables.
- Saved search outputs contain failed fits for some parameter combinations in the original environment.
- `GridSearchCV` uses the classifier's default score unless a scoring metric is supplied; this differs from selecting configurations explicitly by macro F1.
- Ordinary SMOTE operates on the numerically encoded product category as well as continuous features; categorical-aware resampling should be considered.
- Randomness is not controlled consistently across all models and experiments.

No corrected scores or rerun results are claimed in this README.

## Run the notebook

The notebook metadata records Python **3.9.13**, but the original dependency versions were not supplied. A compatible Python environment with the following packages is required:

- pandas, NumPy
- Matplotlib, Seaborn, Plotly
- scikit-learn, imbalanced-learn
- TensorFlow and Keras
- JupyterLab or Jupyter Notebook

A starting installation command in a dedicated virtual environment is:

```bash
python -m pip install jupyterlab pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn tensorflow keras
```

This command does not recreate a locked original environment; package compatibility and legacy API calls may need adjustment.

Place the CSV in the same directory as the notebook, open a terminal there, and launch:

```bash
jupyter lab Projeto_AC.ipynb
```

Run the cells in order. Hyperparameter searches and repeated preprocessing experiments can take substantial time. The notebook retains mutable state across cells, so a fresh kernel is preferable when checking reproducibility.

## Repository contents

| File | Purpose |
| --- | --- |
| `Projeto_AC.ipynb` | Original exploratory analysis, experiments, model training and saved outputs |
| `README.md` | Project overview, dataset requirements and evaluation notes |
| `predictive_maintenance.csv` | External input required locally; not supplied |

## Project status

Academic exploratory project. The repository contains a notebook rather than a packaged prediction service, exported model or automated test suite. This README was prepared by inspecting the notebook and assignment; training was not rerun, and the original notebook was not modified.
