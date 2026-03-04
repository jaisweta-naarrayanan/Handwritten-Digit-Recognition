# Handwritten Digit Recognition with Multi-Class and Multi-Label Logistic Regression

## Overview
This notebook demonstrates handwritten digit recognition using logistic regression, tackling both multi-class (10-way classification) and multi-label classification tasks. It explores the effectiveness of using raw pixel features versus a hierarchical approach where the output probabilities of a multi-class model are used as features for subsequent multi-label predictions.

## Dataset
The dataset consists of 8x8 pixel images of handwritten digits (0-9). Each image is represented by 64 pixel features. The target labels include:
- `y_train`, `y_test`: The actual digit (0-9) for multi-class classification.
- `y_MLL_train`, `y_MLL_test`: Multi-label properties for each digit, specifically `is_even`, `is_greater_than_5`, and `is_prime`.

## Notebook Sections

### Setup: Load and Preprocess Data
- Loads `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `y_MLL_train.csv`, and `y_MLL_test.csv`.
- Normalizes pixel features to the range [0, 1] and then standardizes them.
- Visualizes sample digit images.

### Part 1: Multi-Class Classification with Softmax (10-way)
- Trains a multinomial (softmax) logistic regression model to classify digits 0-9.
- Plots the distribution of learned coefficients for specific classes (e.g., 0 and 7).
- Reports overall training and test accuracy.
- Calculates and displays per-class test accuracy.
- Optionally visualizes predictions for a few test images, showing true vs. predicted labels.

### Part 2: Multi-Label Properties (Baseline)
- Trains three independent binary logistic regression models.
- Each model predicts one property (`is_even`, `is_greater_than_5`, `is_prime`) directly from the raw (standardized) pixel features `X_train_scaled` and `X_test_scaled`.
- Reports training and test accuracy for each property.

### Part 3: The Hierarchical Bridge
- Computes 10-class probability vectors `p(x)` using the softmax model from Part 1. These probabilities represent the likelihood of an image belonging to each of the 10 digits.
- Constructs a new feature matrix `X_new` where rows are these `p(x)` vectors.
- Trains new independent binary logistic regression models for `is_even`, `is_greater_than_5`, and `is_prime` using `X_new` as input features.
- Reports training and test accuracy for each property and compares them to the baseline results from Part 2.

## Discussion
This section discusses why using `p(x)` (the 10-class digit probability vector) as features for predicting properties is more effective than using raw pixels. Key points include:
- **Direct Encoding:** `p(x)` directly encodes the digit identity, which is fundamentally linked to the properties.
- **Dimensionality Reduction:** `p(x)` (10 features) is a more compact and informative representation than raw pixels (64 features).
- **Hierarchical Knowledge Transfer:** The multi-label models leverage the digit recognition capabilities already learned by the softmax model.
- **Theoretical Justification:** For digit-based properties, `p(x)` provides sufficient statistics.

## Setup and Usage
To run this notebook:
1. Ensure you have Python installed along with the following libraries:
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `scikit-learn`
2. Download the `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `y_MLL_train.csv`, and `y_MLL_test.csv` files and place them in the same directory as the notebook, or adjust the file paths accordingly.
3. Run all cells in the notebook sequentially.

## Key Findings
- The softmax multi-class classifier achieves high accuracy in digit recognition.
- Models predicting properties (`is_even`, `is_greater_than_5`, `is_prime`) trained on the probability vectors `p(x)` from the softmax model significantly outperform models trained directly on raw pixel features. This highlights the power of hierarchical feature learning.
```
