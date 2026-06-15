#!/usr/bin/env python3
"""
NLP-Based Password Strength Prediction
=======================================
Dataset : password_data.sqlite (Users table)
Labels  : 0 = Weak, 1 = Medium, 2 = Strong
Method  : TF-IDF (char-level) + Logistic Regression
"""

import sqlite3
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  NLP PASSWORD STRENGTH PREDICTOR")
print("=" * 60)

con = sqlite3.connect("password_data.sqlite")
data = pd.read_sql_query("SELECT * FROM Users", con)
con.close()

if "index" in data.columns:
    data.drop(columns=["index"], inplace=True)

print(f"\n[1] Dataset loaded: {data.shape[0]:,} rows × {data.shape[1]} cols")
print("\nClass distribution:")
label_map = {0: "Weak", 1: "Medium", 2: "Strong"}
for k, v in data["strength"].value_counts().sort_index().items():
    print(f"   {label_map[k]:>6}  ({k}): {v:>6,}  ({100*v/len(data):.1f}%)")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("\n[2] Engineering features ...")

def lower_freq(pwd):   return len([c for c in pwd if c.islower()]) / len(pwd)
def upper_freq(pwd):   return len([c for c in pwd if c.isupper()]) / len(pwd)
def digit_freq(pwd):   return len([c for c in pwd if c.isdigit()]) / len(pwd)
def special_freq(pwd): return len([c for c in pwd if not c.isalpha() and not c.isdigit()]) / len(pwd)
def entropy(pwd):
    freq = Counter(pwd)
    n = len(pwd)
    return -sum((v/n) * np.log2(v/n) for v in freq.values())

data["length"]       = data["password"].str.len()
data["lower_freq"]   = data["password"].apply(lower_freq).round(4)
data["upper_freq"]   = data["password"].apply(upper_freq).round(4)
data["digit_freq"]   = data["password"].apply(digit_freq).round(4)
data["special_freq"] = data["password"].apply(special_freq).round(4)
data["entropy"]      = data["password"].apply(entropy).round(4)

print("\nMean feature values by strength class:")
cols = ["length", "lower_freq", "upper_freq", "digit_freq", "special_freq", "entropy"]
print(data.groupby("strength")[cols].mean().rename(index=label_map).round(3).to_string())

# ── 3. TF-IDF VECTORIZATION ───────────────────────────────────────────────────
print("\n[3] Fitting TF-IDF (char-level) ...")

dataframe = data.sample(frac=1, random_state=42)
passwords = dataframe["password"].tolist()

vectorizer = TfidfVectorizer(analyzer="char")
X_tfidf = vectorizer.fit_transform(passwords)

print(f"   TF-IDF matrix shape : {X_tfidf.shape}")
print(f"   Vocabulary size     : {len(vectorizer.get_feature_names_out())}")

# Append handcrafted features
df_feats = pd.DataFrame(X_tfidf.toarray(),
                        columns=vectorizer.get_feature_names_out())
df_feats["length"]     = dataframe["length"].values
df_feats["lower_freq"] = dataframe["lower_freq"].values

X = df_feats.values
y = dataframe["strength"].values

# ── 4. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[4] Train size: {len(X_train):,} | Test size: {len(X_test):,}")

# ── 5. MODEL TRAINING ─────────────────────────────────────────────────────────
print("\n[5] Training Logistic Regression ...")
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
print("   Done.")

# ── 6. EVALUATION ─────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n[6] Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Weak", "Medium", "Strong"]))

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Weak", "Medium", "Strong"])
disp.plot(ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")

# Feature importance (coefficient magnitudes)
feat_names = list(vectorizer.get_feature_names_out()) + ["length", "lower_freq"]
mean_coef  = np.mean(np.abs(clf.coef_), axis=0)
top_n      = 15
top_idx    = np.argsort(mean_coef)[-top_n:][::-1]
top_feats  = [repr(feat_names[i]) for i in top_idx]
top_vals   = mean_coef[top_idx]

axes[1].barh(top_feats[::-1], top_vals[::-1], color="#4f8ef7")
axes[1].set_xlabel("Mean |Coefficient|")
axes[1].set_title(f"Top {top_n} Most Influential Features", fontsize=13, fontweight="bold")
axes[1].set_xlim(0, top_vals.max() * 1.1)

plt.tight_layout()
plt.savefig("password_strength_analysis.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → password_strength_analysis.png")
plt.show()

# ── 7. SAVE MODEL ─────────────────────────────────────────────────────────────
joblib.dump(clf,        "password_clf.pkl")
joblib.dump(vectorizer, "password_vectorizer.pkl")
print("\n[7] Model saved → password_clf.pkl | password_vectorizer.pkl")

# ── 8. INFERENCE DEMO ─────────────────────────────────────────────────────────
print("\n[8] Sample predictions:")
print(f"  {'Password':<30} {'Predicted':>10}  Probabilities (W / M / S)")
print("  " + "-" * 65)

def predict(pwd, clf, vectorizer):
    n   = len(pwd) if pwd else 1
    X_v = vectorizer.transform([pwd]).toarray()
    lf  = round(sum(1 for c in pwd if c.islower()) / n, 4)
    X_f = np.hstack([X_v, [[n, lf]]])
    pred  = clf.predict(X_f)[0]
    proba = clf.predict_proba(X_f)[0]
    return label_map[pred], proba

test_passwords = [
    "abc",
    "pass",
    "password123",
    "Hello2024",
    "P@ssw0rd!",
    "Tr0ub4dor&3",
    "correct horse battery staple",
    "Zx!9kQ#mP2@nR5sY",
    "P@ssw0rd!2024#Secure",
]

for pw in test_passwords:
    label, proba = predict(pw, clf, vectorizer)
    bar_chars = "▓" * int(proba[2] * 10)
    print(f"  {pw!r:<30} {label:>8}   [{proba[0]:.2f} / {proba[1]:.2f} / {proba[2]:.2f}]")

print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
