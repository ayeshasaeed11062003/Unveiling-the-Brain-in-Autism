import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline


# =========================================================
# Load Data
# =========================================================

X = np.load("c:\\Users\\ayesh\\OneDrive\\Desktop\\Unveiling-the-Brain-in-Autism\\graph_embeddings.npy")
y_dx = np.load("c:\\Users\\ayesh\\OneDrive\\Desktop\\Unveiling-the-Brain-in-Autism\\dx_labels.npy")        # 1=ASD, 2=Control
y_site = np.load("c:\\Users\\ayesh\\OneDrive\\Desktop\\Unveiling-the-Brain-in-Autism\\site_labels.npy")  # Site labels

print("===================================")
print("Data Loaded")
print("Shape of X:", X.shape)
print("Subjects:", len(y_dx))
print("===================================\n")


# =========================================================
# Utility Evaluation Function
# =========================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, task_name="Task"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n--- {task_name} ---")
    print("Accuracy:", round(acc, 4))
    print("F1 Score:", round(f1, 4))
    print(classification_report(y_test, y_pred))

    return acc, f1


# =========================================================
# 1️⃣ RANDOM SPLIT ASD (No PCA)
# =========================================================

print("===================================")
print("Random Split - ASD (No PCA)")
print("===================================")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_dx, test_size=0.2, random_state=42, stratify=y_dx
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000))
])

evaluate_model(pipeline, X_train, X_test, y_train, y_test, "ASD Logistic")


# =========================================================
# 2️⃣ RANDOM SPLIT ASD WITH PCA
# =========================================================

print("\n===================================")
print("Random Split - ASD WITH PCA")
print("===================================")

for n_comp in [100, 200, 300]:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_comp)),
        ('clf', LogisticRegression(max_iter=2000))
    ])

    evaluate_model(
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        f"ASD Logistic + PCA ({n_comp})"
    )


# =========================================================
# 3️⃣ RANDOM FOREST ASD
# =========================================================

print("\n===================================")
print("Random Split - ASD Random Forest")
print("===================================")

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

evaluate_model(rf_pipeline, X_train, X_test, y_train, y_test, "Random Forest ASD")


# =========================================================
# 4️⃣ RANDOM SPLIT SITE CLASSIFICATION
# =========================================================

print("\n===================================")
print("Random Split - Site Classification")
print("===================================")

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y_site, test_size=0.2, random_state=42, stratify=y_site
)

site_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000))
])

evaluate_model(site_pipeline, X_train_s, X_test_s, y_train_s, y_test_s, "Site Logistic")


# =========================================================
# 5️⃣ LEAVE-ONE-SITE-OUT ASD
# =========================================================

print("\n===================================")
print("Leave-One-Site-Out ASD")
print("===================================")

logo = LeaveOneGroupOut()

loso_accuracies = []
loso_f1_scores = []

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_dx, groups=y_site)):

    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y_dx[train_idx], y_dx[test_idx]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000))
    ])

    acc, f1 = evaluate_model(
        pipeline,
        X_train_fold,
        X_test_fold,
        y_train_fold,
        y_test_fold,
        f"LOSO Fold {fold+1}"
    )

    loso_accuracies.append(acc)
    loso_f1_scores.append(f1)

print("\n===================================")
print("Final LOSO Results")
print("===================================")
print("Mean Accuracy:", round(np.mean(loso_accuracies), 4))
print("Mean F1:", round(np.mean(loso_f1_scores), 4))


# =========================================================
# PHASE 1: SITE RESIDUALIZATION
# =========================================================

print("\n===================================")
print("Phase 1: Site Residualization")
print("===================================\n")

encoder = OneHotEncoder(sparse_output=False)
site_onehot = encoder.fit_transform(y_site.reshape(-1, 1))

X_residual = np.zeros_like(X)

print("Removing linear site effects...")

for i in range(X.shape[1]):
    feature_column = X[:, i]
    reg = LinearRegression()
    reg.fit(site_onehot, feature_column)
    predicted = reg.predict(site_onehot)
    X_residual[:, i] = feature_column - predicted

print("Residualization complete.")


# =========================================================
# ASD AFTER RESIDUALIZATION (Random Split)
# =========================================================

print("\n===================================")
print("ASD Classification AFTER Site Removal")
print("===================================")

X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(
    X_residual, y_dx, test_size=0.2, random_state=42, stratify=y_dx
)

pipeline_residual = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000))
])

evaluate_model(
    pipeline_residual,
    X_train_res,
    X_test_res,
    y_train_res,
    y_test_res,
    "ASD Logistic (Site Removed)"
)


# =========================================================
# SITE CLASSIFICATION AFTER REMOVAL
# =========================================================

print("\n===================================")
print("Site Classification AFTER Removal")
print("===================================")

X_train_s_res, X_test_s_res, y_train_s_res, y_test_s_res = train_test_split(
    X_residual, y_site, test_size=0.2, random_state=42, stratify=y_site
)

site_pipeline_res = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000))
])

evaluate_model(
    site_pipeline_res,
    X_train_s_res,
    X_test_s_res,
    y_train_s_res,
    y_test_s_res,
    "Site Logistic (After Removal)"
)


# =========================================================
# LOSO AFTER RESIDUALIZATION
# =========================================================

print("\n===================================")
print("LOSO ASD AFTER Site Removal")
print("===================================")

loso_res_acc = []
loso_res_f1 = []

for fold, (train_idx, test_idx) in enumerate(logo.split(X_residual, y_dx, groups=y_site)):

    X_train_fold, X_test_fold = X_residual[train_idx], X_residual[test_idx]
    y_train_fold, y_test_fold = y_dx[train_idx], y_dx[test_idx]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000))
    ])

    acc, f1 = evaluate_model(
        pipeline,
        X_train_fold,
        X_test_fold,
        y_train_fold,
        y_test_fold,
        f"LOSO Residual Fold {fold+1}"
    )

    loso_res_acc.append(acc)
    loso_res_f1.append(f1)

print("\n===================================")
print("Final LOSO Residual Results")
print("===================================")
print("Mean Accuracy:", round(np.mean(loso_res_acc), 4))
print("Mean F1:", round(np.mean(loso_res_f1), 4))
