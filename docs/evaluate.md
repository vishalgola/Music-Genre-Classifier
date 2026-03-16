# `evaluate.py` — Stage 5: Model Evaluation

## Code Walkthrough

This script loads the trained model, runs it on the held-out test set, and reports how well it performs.

### `load_test_data(test_map)`

Reads every `.npy` file for tracks in the test split, stacks them to 3-channel `(224, 224, 3)` arrays, and returns two NumPy arrays: `X_test` (all spectrograms) and `y_true` (integer labels).

Missing files are warned about but not fatal — the test continues with remaining tracks. This matters because a handful of tracks may have been in the skip list during extraction.

### Prediction

```python
y_probs = model.predict(X_test, batch_size=32, verbose=1)
y_pred  = np.argmax(y_probs, axis=1)
```

The model returns a probability for each of the 16 genres per track. `np.argmax` picks the highest-probability genre as the prediction.

### `plot_confusion_matrix(cm, class_names, out_path)`

Uses seaborn's `heatmap()` to visualise the confusion matrix — a 16×16 grid where cell `[i, j]` shows how many true-genre-`i` tracks were predicted as genre `j`. The diagonal represents correct predictions; off-diagonal cells are errors.

The **magma** colour map is used to match the dark theme of the project.

---

## Theory Behind It

### Why a separate test set?

The validation set is used **during training** (to monitor for overfitting and trigger callbacks). This means the model implicitly "sees" validation data through the early stopping signal — the training stops when validation performance stops improving. The test set is **fully held out and never touched until evaluation**, providing an unbiased estimate of real-world performance.

### Accuracy vs. Weighted F1-Score

**Accuracy** = (number of correct predictions) / (total predictions)

Simple, but misleading when genres have unequal representation. If 40% of test tracks are "Electronic", a model that just predicts "Electronic" every time gets 40% accuracy — not impressive, but technically possible.

**Weighted F1-Score** accounts for class imbalance:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- **Precision:** Of all tracks predicted as genre X, what fraction actually are X?
- **Recall:** Of all actual genre-X tracks, what fraction did the model find?

The "weighted" version averages F1 across all genres, weighted by how many test samples each genre has. This penalises models that ignore minority genres.

### How to read the Confusion Matrix

```
Columns = Predicted genre
Rows = True genre
```

- **Diagonal entries** (correct predictions): should be large and bright
- **Off-diagonal entries** (errors): reveal which genres the model confuses with each other
  - For example, many `Folk` tracks predicted as `Acoustic` suggests the model can't distinguish these genres well, which is often musically understandable
- **Overall brightness of the diagonal** relative to the rest indicates how good the model is

### What's a good accuracy for this task?

FMA Medium with 16 genres is a challenging benchmark. State-of-the-art models achieve **55–70% test accuracy** on this dataset. A fine-tuned MobileNetV2 on mel spectrograms typically lands in the **50–65% range**, which is competitive. Genre classification is inherently difficult because genres overlap and are subjectively defined.
