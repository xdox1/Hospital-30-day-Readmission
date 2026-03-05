# Contributing to Hospital Readmission Prediction

Thank you for being so interested in contributing! Whether you're fixing a bug, improving documentation, or proposing a new feature, your input is valued. Please take a few minutes to read these guidelines before getting started.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Branching Strategy](#branching-strategy)
- [Coding Standards](#coding-standards)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)

---

## Code of Conduct

This project adheres to a standard of respectful, inclusive collaboration. All contributors are expected to maintain a welcoming environment. Harassment, discrimination, or disrespectful behaviour of any kind will not be tolerated.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/your-username/hospital-readmission.git
   cd hospital-readmission
   ```
3. Make sure you have **R (≥ 4.1.0)** and all required packages installed (see [README.md](README.md#installation)).
4. Create a new branch for your work (see [Branching Strategy](#branching-strategy)).

---

## How to Contribute

There are several ways to contribute to this project:

- 🐛 **Bug fixes** — Identify and fix errors in the analysis pipeline
- 📊 **New models** — Add additional classifiers (e.g., XGBoost, SVM, LightGBM)
- ✅ **Tests & validation** — Add checks for data integrity or model output ranges
- 📖 **Documentation** — Improve inline comments, README, or methodology descriptions
- ⚖️ **Fairness extensions** — Expand fairness analysis to cover gender or other demographic attributes
- 🔧 **Refactoring** — Improve code readability, modularity, or performance

---

## Branching Strategy

Please follow this branching convention:

| Branch prefix | Purpose | Example |
|---|---|---|
| `feature/` | New functionality | `feature/add-xgboost-model` |
| `fix/` | Bug fixes | `fix/shap-index-error` |
| `docs/` | Documentation changes | `docs/update-readme` |
| `refactor/` | Code restructuring | `refactor/modularise-preprocessing` |
| `experiment/` | Exploratory / WIP work | `experiment/lightgbm-trial` |

> All branches should be created from `main`. Do **not** commit directly to `main`.

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

---

## Coding Standards

To keep the codebase consistent and readable, please follow these R style guidelines:

### General

- Follow the [tidyverse style guide](https://style.tidyverse.org/) as the baseline convention.
- Use `<-` for assignment, not `=`.
- Keep lines under **100 characters** wide.
- Use 2-space indentation consistently.

### Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Variables | `snake_case` | `train_data2`, `high_risk_index` |
| Functions | `snake_case` | `evaluate_model()`, `predict_proba()` |
| Constants | `UPPER_SNAKE_CASE` | `SEED <- 123` |

### Functions

- Every function must have a clear, single responsibility.
- Document functions with a comment block describing purpose, arguments, and return value:

```r
# Evaluates a caret model on the test set.
# Args:
#   model: A trained caret model object.
#   name:  A character string label for the model.
# Returns:
#   A tibble with columns: Model, ROC_AUC, Accuracy.
evaluate_model <- function(model, name) {
  ...
}
```

### Reproducibility

- Always set a random seed before any stochastic operation (`set.seed()`).
- Do **not** hard-code absolute file paths. Use relative paths or project-relative references (e.g., via `here::here()`).

### Data

- Do not commit raw data files to the repository. Use `.gitignore` to exclude data directories.
- If adding a new dataset, document its source, structure, and any preprocessing assumptions in the README.

---

## Commit Message Guidelines

Write clear, descriptive commit messages using the following format:

```
<type>: <short summary in present tense>

[Optional: longer description of the change and why it was made]
```

**Types:**

| Type | Use for |
|---|---|
| `feat` | New feature or model |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code restructuring without behaviour change |
| `test` | Adding or updating tests |
| `chore` | Dependency updates, config changes |

**Examples:**

```
feat: add XGBoost model to comparison pipeline

fix: correct SHAP index alignment for test set samples

docs: add fairness metrics explanation to README
```

---

## Pull Request Process

1. Ensure your branch is **up to date** with `main` before opening a PR:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. Write a clear **PR description** that includes:
   - What the change does
   - Why it was needed
   - Any limitations or known issues
3. Link any related **issues** in the description using `Closes #<issue-number>`.
4. Make sure your code runs **end-to-end** without errors before submitting.
5. PRs require at least **one approving review** before merging.
6. Squash commits where appropriate to keep the `main` history clean.

---

## Reporting Issues

Found a bug? Please open an issue and include:

- A **clear title** describing the problem
- Steps to **reproduce** the issue
- The **expected** vs **actual** behaviour
- Your R version (`R.version`) and operating system
- Any relevant **error messages or stack traces**

Use the label `bug` when creating the issue.

---

## Feature Requests

Have an idea to improve the project? Open an issue with the label `enhancement` and describe:

- The **problem** you are trying to solve
- Your **proposed solution**
- Any **alternatives** you considered
- Whether you'd be willing to implement it yourself

---

## Questions?

If you're unsure about anything, feel free to open a **Discussion** or raise an issue tagged `question`. We're happy to help.

---

Thank you for helping make this project better! 🙌
