# Dempster–Shafer Fusion — README

**Repository / Module:** `notebooks/8_DempsterShafer.ipynb`
**Purpose:** Fuse predictions from three deep-learning models (LSTM, GRU, CNN) using a Dempster–Shafer evidence fusion scheme and evaluate the combined prediction on a test dataset.

---

## Overview

This script implements a fixed Dempster–Shafer fusion pipeline that:

* Loads pre-trained models (LSTM, GRU, CNN) from Google Drive (Colab) or local paths.
* Builds test sequences from a tabular test CSV.
* Obtains class probabilities and hard predictions from each model.
* Computes model-level performance metrics (accuracy, precision, recall, F1, AUC).
* Calculates model weights from those metrics.
* Converts model probabilities to Basic Probability Assignments (BPAs) including an uncertainty mass.
* Combines BPAs across models with Dempster’s rule (accounting for conflict).
* Produces final fused predictions, probabilities and uncertainty; evaluates and saves results.

The approach improves robustness by letting each model contribute evidence and by explicitly modelling uncertainty and conflict between models.

---

## Results

**Combined Model Performance**

* Accuracy: 0.9769
* Precision: 0.9629
* Recall: 0.9440
* F1-Score: 0.9533
* AUC-ROC: 0.9961

**Model Weights (normalized)**

* LSTM: 0.3321
* GRU: 0.3334
* CNN: 0.3345

---

## Required files & expected layout

**Pre-trained models:**

* `backorder_lstm_model_dataset_3_2.h5`
* `backorder_gru_model_dataset_3.h5`
* `backorder_cnn_model_dataset_3.h5`

**Test dataset CSV:**

* `Test_dataset_3.csv` — expected to contain features and a target column as the last column; sequences are constructed by a sliding window of `SEQ_LENGTH` rows.


**Output path:**
`/content/drive/MyDrive/Thesis/dempster_shafer_fixed_results.csv`

The code is arranged to run inside Google Colab and assumes models & test CSV are saved on Google Drive. If you run locally, update the file paths and remove Drive mount.

---

## Dependencies

* Python 3.8+
* numpy
* pandas
* tensorflow (compatible with saved model versions)
* scikit-learn

**Install example:**

```bash
pip install numpy pandas tensorflow scikit-learn
```

---

## How to run (Google Colab)

1. Open a Colab notebook (or a `.py` runner in Colab).
2. Mount Google Drive (the script already calls `drive.mount('/content/drive')`).
3. Ensure model files and test CSV are uploaded to the correct Drive paths.
4. Run the script cell-by-cell or execute the Python file.

**Minimal runnable snippet (Colab):**

```python
# in a Colab cell
!pip install -q numpy pandas tensorflow scikit-learn
# then run your fusion script (make sure paths are correct)
%run /content/drive/MyDrive/Thesis/dempster_shafer_fusion.py
```

---

## Important implementation details

**Sequence creation**

* `create_sequences(data, seq_length=5)` builds sequences by taking `seq_length` rows of all columns except the last as features and uses the value of the last column at time `i+seq_length` as the label.
* Confirm ordering, indexing, and whether sorting is required prior to sequence creation.

**Predictions & probabilities**

* `get_predictions` uses `model.predict(X)` returning a single probability per sample (assumed to be P(class=1)).
* It returns both:

  * `y_pred`: thresholded predictions at 0.5
  * `y_proba`: a 2-column probability array `[P(class0), P(class1)]`

**Metric-based weights**
Formula:

```
score = 0.4 * AUC + 0.3 * F1 + 0.2 * Accuracy + 0.1 * Precision
```

Then normalized across models. This gives priority to AUC and F1 while still using accuracy and precision.

**BPA (Basic Probability Assignment)**
For each model and sample:

* `believe_0 = (1 - p1) * (1 - error_rate) * weight`
* `believe_1 = p1 * (1 - error_rate) * weight`
* `uncertainty = error_rate * weight`

Normalized so `believe_0 + believe_1 + uncertainty = 1`.

**Dempster’s rule combination**

* Evidence combined sequentially across models.
* Conflict `K = mA(0)*mB(1) + mA(1)*mB(0)`
* Combine when `K < 1` using canonical Dempster formulas and renormalize.

Final fused prediction = `argmax(believe_0, believe_1)`.

---

## Output

* Printed performance for the combined model (Accuracy, Precision, Recall, F1, AUC).
* CSV saved to:
  `/content/drive/MyDrive/Thesis/dempster_shafer_fixed_results.csv`

**CSV columns include:**

* `True_Label`, `Final_Prediction`, `Probability_Class_0`, `Probability_Class_1`, `Uncertainty`
* Individual model columns: `{model}_Pred`, `{model}_Prob0`, `{model}_Prob1`

---

## Suggestions & possible improvements

* **Robustness to total conflict:** Add a fallback rule when `K == 1`.
* **Dynamic BPA mapping:** Use calibrated probabilities (e.g., temperature scaling).
* **Alternative weighting schemes:** Try stacking, Bayesian averaging, or learnable meta-weights.
* **Vectorization:** Current per-sample loops can be slow for large test sets.
* **Compatibility:** Ensure model outputs match `get_predictions` expectations.
* **Logging:** Save random seeds and a `performance_summary.json` with metrics used for weights.

---

## References

* Shafer, G. (1976). *A Mathematical Theory of Evidence*.
* Dempster, A.P. (1967). *Upper and lower probabilities induced by a multivalued mapping*. Ann. Math. Statist.
* Practical Dempster–Shafer fusion tutorials and examples in pattern recognition literature.

---

## Notes & caveats

* The script is tailored for **binary classification** (class 0 vs class 1). For multi-class, BPA and combination rules must be adapted.
* Paths are set for Colab + Google Drive. Update for local runs.
* Performance numbers reflect your dataset and preprocessing.

---

## Quick copy-paste: How to adapt for local run

**Remove Drive mount:**

```python
# comment out or remove
# from google.colab import drive
# drive.mount('/content/drive')
```

**Use local paths:**

```python
model_lstm = tf.keras.models.load_model('/path/to/backorder_lstm_model_dataset_3_2.h5')
test_data = pd.read_csv('/path/to/Test_dataset_3.csv')
```

**Run script:**

```bash
python 8_DempsterShafer.py
```
