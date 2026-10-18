# Loan Approval Prediction — HDFC Bank Ltd.

AI Mini Project (Semester 2). A supervised machine learning **classification** model that predicts whether a loan application should be **Approved** or **Rejected**.

Repository: https://github.com/agamjotdhingra/mini-project-1.git

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 | 5-fold CV |
|---|---|---|---|---|---|
| Logistic Regression | 85.4% | 83.2% | 98.8% | 0.903 | 80.6% |
| Decision Tree | 85.4% | 85.3% | 95.3% | 0.900 | 79.3% |
| **Random Forest** | **87.0%** | **84.8%** | **98.8%** | **0.913** | **81.3%** |
| Baseline (approve all) | 69.1% | – | – | – | – |

The most important factor is **Credit_History**, followed by loan-to-income ratio and total income.

## Files
- `Loan_Approval_Prediction.ipynb` — full notebook with outputs
- `loan_approval.py` — same code as a script
- `data/loan_data.csv` — Loan Prediction dataset (Kaggle / Analytics Vidhya), 614 rows
- `outputs/` — charts

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Loan_Approval_Prediction.ipynb
```
or `python loan_approval.py`. Also works in Google Colab (upload the notebook and `data/loan_data.csv`).
