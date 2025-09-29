# Thesis-DeepLearning-DempsterShafer
Improving Product Demand Forecasting Using Deep Learning Models and Dempster-Shafer Theory.

# Improving Product Demand Forecasting Using Deep Learning Models and Dempster-Shafer Theory

This repository contains the complete implementation of my Master's thesis project, which focuses on **enhancing product demand forecasting** using a combination of **deep learning models** (LSTM, GRU, CNN, and hybrid architectures) and the **Dempster-Shafer theory of evidence fusion**.

The work is based on the [Back Order Prediction Dataset](https://www.kaggle.com/datasets/gowthammiryala/back-order-prediction-dataset).

---

## Repository Structure

```
Thesis-Demand-Forecasting/
│
├── README.md           # Dataset description and Kaggle link                  
│
│
├── notebooks/              
│   ├── 1_data_exploration.ipynb
│   ├── 2_preprocessing.ipynb
│   ├── 3_LSTM.ipynb
│   ├── 4_GRU.ipynb
│   ├── 5_CNN.ipynb
│   ├── 6_LSTM_CNN.ipynb
│   ├── 7_GRU_CNN.ipynb
│   └── 8_DempsterShafer.ipynb
│
├── results/                
│   ├── model_comparisons.png
│   ├── performance_tables.csv
│   └── figures/            # Visualization outputs
│           
└── requirements.txt               
```

---

## Preprocessing Steps

1. Dropped `sku` column
2. Converted negative values to `NaN` (due to illogical values)
3. Removed missing values
4. Removed duplicate rows (≈593,553 detected after dropping `sku`)
5. Replaced outliers using IQR method
6. Removed new duplicates (after outlier handling)
7. Categorical encoding
8. Normalization
9. Balance dataset

---

## Model Performance

| Model                              | Accuracy (±std) | Precision (±std) | Recall (±std)   | AUC (±std)      | Loss (±std)     |
| ---------------------------------- | --------------- | ---------------- | --------------- | --------------- | --------------- |
| **LSTM**                           | 0.9716 ± 0.0007 | 0.9461 ± 0.0042  | 0.9402 ± 0.0063 | 0.9914 ± 0.0016 | 0.0857 ± 0.0064 |
| **GRU**                            | 0.9711 ± 0.0033 | 0.9411 ± 0.0212  | 0.9445 ± 0.0153 | 0.9872 ± 0.0028 | 0.1124 ± 0.0135 |
| **CNN**                            | 0.9707 ± 0.0028 | 0.9209 ± 0.0136  | 0.9662 ± 0.0055 | 0.9952 ± 0.0004 | 0.0809 ± 0.0071 |
| **LSTM+CNN**                       | 0.9738 ± 0.0009 | 0.9629 ± 0.0068  | 0.9311 ± 0.0081 | 0.9897 ± 0.0016 | 0.0888 ± 0.0085 |
| **GRU+CNN**                        | 0.9729 ± 0.0008 | 0.9639 ± 0.0073  | 0.9264 ± 0.0087 | 0.9899 ± 0.0011 | 0.0870 ± 0.0027 |
| **Dempster-Shafer (LSTM+GRU+CNN)** | **0.9769**      | **0.9629**       | **0.9440**      | **0.9961**      | —               |

 The fusion model using **Dempster-Shafer theory** outperformed all individual and hybrid models.

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/username/Thesis-Demand-Forecasting.git
   cd Thesis-Demand-Forecasting
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
4. Run the notebooks in order from `1_data_exploration.ipynb` to `8_DempsterShafer.ipynb`.

---

## Citation

If you find this work useful, please cite:

```
Torkamand, Hodais. "Improving Product Demand Forecasting Using the Combination of Deep Learning Models and Dempster-Shafer Theory." Master's Thesis, 2025.
```

---
