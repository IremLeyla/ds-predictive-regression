# Predictive Regression

This repo looks at what changes when you train a model to predict rather than just to explore. You never have all the data, so the goal is a model that generalizes: one that strikes a good balance between bias and variance. Across three notebooks you will use a train/test split to estimate performance on unseen data, expand model flexibility with polynomial features, control overfitting with regularization, and decide how to handle outliers. Linear regression is the example model throughout, but the framework applies to any supervised learning task.

## Learning Objectives

By the end of this repository, you should be able to:

- Split data into train and test sets and evaluate a model with the root mean squared error.
- Recognize underfitting (high bias) and overfitting (high variance) and explain the trade-off between them.
- Create more flexible linear models with polynomial and interaction features.
- Apply Ridge (L2) and Lasso (L1) regularization to reduce overfitting.
- Detect outliers visually and with z-scores, and judge when to remove them.

## Learning Path

Work through the notebooks in order; each one builds on the previous.

| File / Folder | Description |
|---|---|
| [**1 - Bias-Variance Trade-Off**](1_bias_variance_tradeoff.ipynb) | Train/test splits, polynomial features, and the bias-variance trade-off. |
| [**2 - Regularization**](2_regularization.ipynb) | Interaction features plus Ridge and Lasso regression to curb overfitting. |
| [**3 - Detecting Outliers**](3_detecting_outliers.ipynb) | Spotting outliers and measuring how removing them affects model performance. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Data**](data/) | The white wine quality dataset used across the notebooks. |
| [**Assets**](assets/) | Images used in the notebooks. |
| [**Solutions**](solutions/) | Reference solutions. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd ds-predictive-regression`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebooks

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## References & Further Reading

- [**Train Test Split (scikit-learn)**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html): How to split data into train and test sets.
- [**Underfitting vs. Overfitting (scikit-learn)**](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html): A worked polynomial example of the bias-variance trade-off.
- [**Linear Models: Ridge, Lasso, and ElasticNet (scikit-learn)**](https://scikit-learn.org/stable/modules/linear_model.html): The reference for the regularized regression models used in notebook 2.
- [**Outliers: To Drop or Not to Drop**](https://www.theanalysisfactor.com/outliers-to-drop-or-not-to-drop/): Practical guidance on when removing outliers is justified.
