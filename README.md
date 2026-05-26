# SVM-MEWMA Control Chart

A Python implementation of a **one-class SVM enhanced Multivariate EWMA (MEWMA) control chart** for statistical process monitoring and anomaly detection in high-dimensional data.

Built as part of Honours research in Statistics & Data Science at the University of Pretoria.

---

## What it does

Traditional control charts signal when a process has shifted away from normal behaviour. This project improves on standard MEWMA by adding a **one-class Support Vector Machine** as a feature layer — the SVM learns the boundary of normal behaviour from historical data, and its decision score is appended to the monitoring vector before EWMA smoothing. The result is a hybrid statistical + machine learning detector that is more sensitive to subtle, complex shifts than a chart using raw process variables alone.

**In plain terms:** it watches a stream of multivariate data in real time and raises an alarm as early as possible when something goes wrong.

---

## Key features

- **Phase I model building** — fits a one-class SVM and estimates in-control mean and covariance from historical data
- **ARL₀ calibration** — uses bisection search over Monte Carlo simulations to set the control limit at a target false-alarm rate (ARL₀ = 370)
- **Phase II monitoring** — evaluates detection performance across shift sizes δ = 0.0 to 3.0
- **Full performance report** — ARL, median run length, SDRL, alarm probability, accuracy, precision, recall, F1, and ROC-AUC at each shift size
- **Visualisations** — Phase I SVM score distribution, MEWMA chart diagnostics, ARL curves, and box plots

---

## Method summary

A one-class SVM is trained on Phase I in-control observations. Its signed decision score is appended to the standardised process variables to form an augmented monitoring vector, which is then tracked by MEWMA:

```
z_t = λ·y_t + (1 − λ)·z_{t−1}
T²_t = (z_t − μ_y)ᵀ · Σ_z⁻¹ · (z_t − μ_y)
```

An alarm fires when T²_t exceeds the calibrated control limit `h`.

---

## Stack

- Python 3.x
- numpy, pandas
- scikit-learn (OneClassSVM, StandardScaler, metrics)
- matplotlib

---

## Usage

Open `svm_mewma_control_chart_notebook.ipynb` in Jupyter and run cells top to bottom. Key settings are in the **User-adjustable settings** cell at the top:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `P` | 4 | Number of process variables |
| `N_PHASE1` | 2000 | Historical in-control observations |
| `LAMBDA` | 0.05 | EWMA smoothing parameter |
| `TARGET_ARL0` | 370 | Target in-control average run length |
| `N_CALIBRATION_REPS` | 300 | Monte Carlo reps for calibration (increase to 5000+ for publication) |
| `N_PHASE2_REPS` | 500 | Monte Carlo reps for ARL curves (increase to 2000+ for publication) |

To use your own data, replace the simulated Phase I data line with:
```python
X_phase1 = pd.read_csv("your_data.csv").to_numpy()
```

---

## Research context

This notebook forms part of a broader Honours thesis exploring hybrid statistical and machine learning monitoring frameworks for complex, high-dimensional processes. The full thesis extends this work to include a CNN feature-extraction layer (SVM-CNN-EWMA) for improved performance on non-linear process shifts.

---

*University of Pretoria · BCom Honours in Statistics & Data Science · 2026*
