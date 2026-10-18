# Loan Approval Prediction - HDFC Bank
# AI Mini Project 1

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

APPROVED, REJECTED, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"outputs/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# load data
df = pd.read_csv("data/loan_data.csv")
print(df.head())

# basic checks
print("Shape (rows, columns):", df.shape)
print("\nData types:\n", df.dtypes)

missing = df.isnull().sum()
print("Missing values per column:\n", missing[missing > 0].sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())
print("\nTarget distribution:\n", df["Loan_Status"].value_counts())
print("\n% approved:", round((df["Loan_Status"] == "Y").mean() * 100, 1))


# cleaning
data = df.drop(columns=["Loan_ID"]).copy()

# mode for categorical columns, median for loan amount (skewed)
for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History", "Loan_Amount_Term"]:
    data[col] = data[col].fillna(data[col].mode()[0])
data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())

# convert text to numbers
data["Dependents"] = data["Dependents"].replace("3+", "3").astype(int)
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["Married"] = data["Married"].map({"Yes": 1, "No": 0})
data["Education"] = data["Education"].map({"Graduate": 1, "Not Graduate": 0})
data["Self_Employed"] = data["Self_Employed"].map({"Yes": 1, "No": 0})
data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})

print("Missing values after cleaning:", data.isnull().sum().sum())


# EDA

# target
counts = df["Loan_Status"].map({"Y": "Approved", "N": "Rejected"}).value_counts()
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(counts.index, counts.values, color=[APPROVED, REJECTED], width=0.55)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 6,
            f"{int(b.get_height())} ({b.get_height() / len(df):.0%})", ha="center")
ax.set_title("Loan Status (Target)")
ax.set_ylabel("Number of applications")
save(fig, "01_target")

# approval rate by a column
def approval_rate_plot(col, labels, title, name):
    rate = df.groupby(col)["Loan_Status"].apply(lambda s: (s == "Y").mean() * 100)
    rate.index = [labels.get(i, i) for i in rate.index]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(rate.index.astype(str), rate.values, color=APPROVED, width=0.55)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%", ha="center")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Approval rate (%)")
    ax.set_title(title)
    save(fig, name)

approval_rate_plot("Credit_History", {0.0: "Bad (0)", 1.0: "Good (1)"},
                   "Approval Rate by Credit History", "02_credit_history")
approval_rate_plot("Property_Area", {}, "Approval Rate by Property Area", "03_property_area")

# outliers
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, col, title in zip(axes, ["ApplicantIncome", "LoanAmount"],
                          ["Applicant Income", "Loan Amount (in thousands)"]):
    ax.boxplot(df[col].dropna(), widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#dbe8f8", color=APPROVED),
               medianprops=dict(color=REJECTED, linewidth=2),
               flierprops=dict(marker="o", markerfacecolor=APPROVED, markersize=4, alpha=0.5))
    ax.set_title(title)
    ax.set_xticks([])
save(fig, "04_outliers")

# income of approved vs rejected
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
fig, ax = plt.subplots(figsize=(6, 4))
for status, color, label in [("Y", APPROVED, "Approved"), ("N", REJECTED, "Rejected")]:
    ax.hist(np.log(df.loc[df["Loan_Status"] == status, "TotalIncome"]), bins=25,
            alpha=0.6, color=color, label=label, edgecolor="white")
ax.set_xlabel("log(Total Income)")
ax.set_ylabel("Applications")
ax.set_title("Total Income: Approved vs Rejected")
ax.legend(frameon=False)
save(fig, "05_income_vs_status")

# correlation
num = data[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
            "Credit_History", "Loan_Status"]]
corr = num.corr()
fig, ax = plt.subplots(figsize=(6.5, 5.2))
im = ax.imshow(corr, cmap="Blues", vmin=-0.2, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v > 0.6 else "#0b0b0b", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Correlation Heatmap")
save(fig, "06_correlation")


# new features
data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
data["TotalIncome_log"] = np.log(data["TotalIncome"])
data["LoanAmount_log"] = np.log(data["LoanAmount"])
data["EMI"] = data["LoanAmount"] / data["Loan_Amount_Term"]
data["Loan_to_Income"] = data["LoanAmount"] / data["TotalIncome"]

data = pd.get_dummies(data, columns=["Property_Area"], drop_first=True, dtype=int)
data = data.drop(columns=["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "TotalIncome"])

X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"]
print("Final features:", list(X.columns))


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# scaling only needed for logistic regression
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
}
uses_scaling = {"Logistic Regression": True, "Decision Tree": False, "Random Forest": False}


# train and evaluate
results, preds = [], {}
for name, model in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if uses_scaling[name] else (X_train, X_test)
    model.fit(Xtr, y_train)
    p = model.predict(Xte)
    preds[name] = p
    cv_X = scaler.transform(X) if uses_scaling[name] else X
    cv = cross_val_score(model, cv_X, y, cv=5, scoring="accuracy").mean()
    results.append({"Model": name,
                    "Accuracy": accuracy_score(y_test, p),
                    "Precision": precision_score(y_test, p),
                    "Recall": recall_score(y_test, p),
                    "F1-score": f1_score(y_test, p),
                    "CV Accuracy (5-fold)": cv})

baseline = (y_test == 1).mean()
results_df = pd.DataFrame(results).set_index("Model").round(3)
print(f"Baseline (approve everyone) accuracy: {baseline:.3f}\n")
print(results_df)

# comparison chart
fig, ax = plt.subplots(figsize=(8, 4.2))
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
w = 0.25
x = np.arange(len(metrics))
for i, (name, color) in enumerate(zip(results_df.index, [APPROVED, REJECTED, THIRD])):
    ax.bar(x + (i - 1) * w, results_df.loc[name, metrics], w - 0.02, color=color, label=name)
ax.axhline(baseline, color="#52514e", linestyle="--", linewidth=1, label="Baseline (approve all)")
ax.set_xticks(x, metrics)
ax.set_ylim(0, 1.05)
ax.set_title("Model Comparison (test set)")
ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.3), fontsize=9)
save(fig, "07_model_comparison")

# confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, name in zip(axes, models):
    cm = confusion_matrix(y_test, preds[name])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    ax.set_xticks([0, 1], ["Rejected", "Approved"])
    ax.set_yticks([0, 1], ["Rejected", "Approved"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(name)
save(fig, "08_confusion_matrices")


# feature importance
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(imp.index, imp.values, color=APPROVED, height=0.6)
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Importance")
save(fig, "09_feature_importance")
print(imp.sort_values(ascending=False).round(3))

lr = models["Logistic Regression"]
coef = pd.Series(lr.coef_[0], index=X.columns).sort_values(ascending=False).round(3)
print("Logistic Regression coefficients:\n", coef)


# testing on a new applicant
def predict_applicant(gender, married, dependents, education, self_employed,
                      applicant_income, coapplicant_income, loan_amount,
                      loan_term, credit_history, property_area):
    total = applicant_income + coapplicant_income
    row = {"Gender": gender, "Married": married, "Dependents": dependents,
           "Education": education, "Self_Employed": self_employed,
           "Loan_Amount_Term": loan_term, "Credit_History": credit_history,
           "TotalIncome_log": np.log(total), "LoanAmount_log": np.log(loan_amount),
           "EMI": loan_amount / loan_term, "Loan_to_Income": loan_amount / total,
           "Property_Area_Semiurban": int(property_area == "Semiurban"),
           "Property_Area_Urban": int(property_area == "Urban")}
    x = pd.DataFrame([row])[X.columns]
    prob = rf.predict_proba(x)[0, 1]
    return ("Approved" if prob >= 0.5 else "Rejected"), round(prob, 2)

print("Good credit:", predict_applicant(1, 1, 0, 1, 0, 5000, 1500, 120, 360, 1, "Semiurban"))
print("Bad credit: ", predict_applicant(1, 1, 0, 1, 0, 5000, 1500, 120, 360, 0, "Semiurban"))
