# Predictive Regression: Cheat Sheet

A simple summary of all 3 notebooks, plus the questions an AI Project Manager should ask.

---

## The Big Idea

We don't build models just to understand old data. We build them to **predict new data** they have never seen.

A good model **generalizes**: it learned the real pattern, not the exact examples.

> A student who memorizes last year's exam fails the new one. A student who understands the topic passes both.

---

## Key Words

| Word | Simple meaning |
|---|---|
| **Feature (X)** | An input the model uses (e.g. alcohol, sugar) |
| **Target (y)** | What we want to predict (e.g. wine quality) |
| **Train set** | Data the model learns from (practice questions) |
| **Test set** | Data the model never saw, used to check it (the real exam) |
| **Coefficient** | How much weight the model gives to a feature |
| **Hyperparameter** | A setting **you** choose, the model doesn't learn it (e.g. `alpha`) |
| **Residual** | True value minus predicted value = the error for one row |
| **Noise** | Random variation in the data that nobody can predict |

---

## How We Measure a Model

### RMSE (Root Mean Squared Error)
"On average, how far off are my predictions?"
- Same unit as the target
- **Lower = better**, 0 = perfect
- Only meaningful **compared to something** (e.g. the standard deviation of y, or a simple baseline like "always guess the average")

### R² (R-squared)
"How much of the variation in y does my model explain?"
- 1 = explains everything, 0 = no better than guessing the average
- **Higher = better**

### Adjusted R²
R² with a penalty for using many features. Normal R² always goes up when you add features, even useless ones. Adjusted R² only goes up if the feature actually helps.

---

## Notebook 1: Bias-Variance Trade-Off

### The error formula
```
Total error = bias² + variance + noise
```
- **Noise** can't be removed (irreducible error). It's the "floor": on test data you can't get below it.
- We can only work on **bias** and **variance**.

### Bias vs. variance

| | Bias (too simple) | Variance (too sensitive) |
|---|---|---|
| What happens | Model misses the real pattern | Model memorizes the noise |
| Name | **Underfitting** | **Overfitting** |
| Train error | high | very low |
| Test error | high | high |
| Changes a lot with new training data? | no (stable but wrong) | **yes** (unstable) |

**Good fit:** low train error AND similar, low test error.

> **Warning sign:** big gap between train and test error means overfitting.

### The trade-off
- More flexible model = less bias, more variance
- Simpler model = more bias, less variance
- The goal is the **sweet spot** in the middle (the bottom of the U-shaped error curve).
- It's often smart to accept a little bias if it cuts variance a lot.

### Train-test split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```
- `test_size`: share of data for testing (default 0.25)
- `random_state`: fixes the shuffle so results are reproducible
- **Never judge a model on its training data.** It has already seen the answers.

### Polynomial features
Add x², x³, ... as new features so a straight line can bend into a curve.
- Still a **linear** model, because the coefficients are linear. x² is just another column.
- `fit_transform` on train, only `transform` on test. **Never learn anything from the test set.**

### What we saw
| Model | Result |
|---|---|
| Degree 1 (straight line) | Underfitting |
| Degree 2 | Better, still underfitting |
| **Degree 3** | **Sweet spot**, matches the true formula, RMSE ≈ noise level |
| Degree 8 | Overfitting: wiggly, unstable, bad at the edges where there was no training data |

### Residual plots (error analysis)
Works with any number of features.
- **True vs. predicted:** points should sit on the diagonal line.
- **Residuals vs. predicted:** should look like a **random cloud around 0**.
  - Random cloud: model captured everything it could
  - Clear pattern (wave, U-shape): model missed something, usually underfitting
  - A few extreme points: model fails in some cases

### Occam's razor
If two models perform about the same, **pick the simpler one**. It's more stable and easier to explain.

---

## Notebook 2: Regularization

### The problem
With many features (here 66 interaction features), `LinearRegression` overfits and gets huge coefficients. It has **no brake** (no hyperparameter to control complexity).

**Interaction feature:** two features multiplied, e.g. `sugar × acidity`. 11 features + 55 pairs = 66 features.

### The solution: regularization = a brake on the coefficients
The model now tries to keep **errors small AND coefficients small**.

**`alpha`** = how strong the brake is:
- alpha = 0: no brake (same as LinearRegression)
- bigger alpha: stronger brake, smaller coefficients
- alpha too big: everything is close to 0, which means **underfitting**
- The best alpha depends on the dataset, so you have to try values.

### Ridge vs. Lasso

| | **Ridge (L2)** | **Lasso (L1)** |
|---|---|---|
| What it does | Makes coefficients **small** | Makes coefficients small or **exactly 0** |
| Feature selection? | No, keeps all features | **Yes**, drops unimportant features |
| When to use | Usually the first choice | Many features + you want a simpler, easier-to-explain model |

**ElasticNet** = a mix of both.

### What we saw (test RMSE)
| Model | Test RMSE | Features used |
|---|---|---|
| LinearRegression | 0.80 | 66 |
| Ridge α=10 | 0.78 | 66 |
| **Lasso α=0.05** | **0.77** | **16** |

> The best model used only 16 of 66 features. Fewer features can mean better predictions.

---

## Notebook 3: Detecting Outliers

### What is an outlier?
A value that is very different from the others.

**Possible causes:** typos, broken measuring devices, wrong data sources, or it's **real** (a genuinely special case).

> Not every outlier is a mistake. First find out **why** it's there.

- **Univariate:** unusual in one feature
- **Multivariate:** only unusual when looking at several features together

### How to find them
**Visually:** histograms, box plots (points outside the whiskers), scatter plots

**With the z-score:** "How many standard deviations is this value from the mean?"
```
z = (value − mean) / standard deviation
```
- Common rule: |z| > 3 means outlier (sometimes 2.5 or 3.5)
- Works best when the data looks like a bell curve (normal distribution)

### The workflow
1. **Split first, then clean.** Only remove outliers from the **train** set. The test set stays like real-world data.
2. **Try a transformation first** (e.g. `log`). It squeezes big values and fixes skewed data.
3. **Calculate the data loss.** How many rows would you delete? (Here about 7–8%, which is a lot.)
4. Remove the rows from **both X_train and y_train**, so they still match.
5. **Apply every transformation to the test set too** (e.g. the same `log`), except outlier removal.
6. Compare models with and without outliers on the **same test set**.

### What we saw
Without outliers, test RMSE improved a little (0.377 → 0.366). The train RMSE dropped much more, partly because the hard cases were simply gone. **Always trust the test score.**

### Be careful
Removing data means the model learns **nothing** about those cases. Only remove outliers if it makes sense **statistically AND from a business point of view**.

---

## Golden Rules

1. **Judge models on test data, never on train data.**
2. **Big gap between train and test = overfitting.**
3. **Noise sets a limit.** No model is perfect.
4. **Simpler is often better** (Occam's razor).
5. **Split first.** Never learn from or clean the test set.
6. **Same transformations on train and test.**
7. **Outliers are a business decision**, not only a math decision.

---

## For AI Project Managers: Questions to Ask the Data Science Team

**About performance**
- Is this score on **train or test** data?
- How big is the **gap** between train and test?
- How much better are we than a **simple baseline** (e.g. always guessing the average)?
- What is realistically possible, given the **noise** in the data?

**About complexity**
- Would a **simpler model** perform almost as well?
- Can we **explain** how the model decides? (Important for customers and regulation, e.g. the EU AI Act)
- How **stable** is the model if we train it on different data?

**About data**
- Was the data **split before cleaning**? Could there be **data leakage**?
- Which data was **removed** (outliers) and why? What % of the data is that?
- Are the removed cases maybe **important for the business** (e.g. fraud, VIP customers)?

**After go-live**
- Who **monitors** model performance in production?
- What happens when the data **changes over time** (data drift)?

> You don't need to build the model. You need to ask the right questions.
