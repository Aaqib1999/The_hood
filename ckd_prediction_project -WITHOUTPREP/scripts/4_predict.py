import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load data
df = pd.read_csv("data/kidney_disease.csv")
if 'id' in df.columns:
    df = df.drop('id', axis=1)
df = df.apply(lambda col: col.map(lambda x: np.nan if str(x).strip() in ['?', ''] else x))

# Separate target column before discretization
classification = df['classification']
df = df.drop('classification', axis=1)

# 2. Discretize selected numeric columns
def discretize_column(col, bins, labels):
    return pd.cut(df[col].astype(float), bins=bins, labels=labels, include_lowest=True)

discretize_map = {
    'age': ([0, 30, 60, 120], ['low', 'normal', 'high']),
    'bp': ([0, 80, 120, 300], ['low', 'normal', 'high']),
    'sg': ([1.000, 1.010, 1.020, 1.030], ['low', 'normal', 'high']),
    'al': ([0, 1, 3, 5], ['low', 'normal', 'high']),
    'su': ([0, 1, 3, 5], ['low', 'normal', 'high']),
    'bgr': ([0, 100, 140, 500], ['low', 'normal', 'high']),
    'bu': ([0, 20, 50, 400], ['low', 'normal', 'high']),
    'sc': ([0, 1, 2, 20], ['low', 'normal', 'high']),
    'sod': ([100, 135, 145, 160], ['low', 'normal', 'high']),
    'pot': ([2, 4, 6, 10], ['low', 'normal', 'high']),
    'hemo': ([0, 10, 15, 25], ['low', 'normal', 'high']),
    'pcv': ([0, 30, 45, 60], ['low', 'normal', 'high']),
    'wc': ([0, 6000, 12000, 30000], ['low', 'normal', 'high']),
    'rc': ([0, 3, 5, 10], ['low', 'normal', 'high']),
}

for col, (bins, labels) in discretize_map.items():
    if col in df.columns:
        df[col + '_disc'] = discretize_column(col, bins, labels)

# 3. Drop original numeric columns (keep only discretized and categoricals)
drop_cols = list(discretize_map.keys())
df = df.drop(columns=drop_cols)

# 4. Fill missing values with mode (all are categorical now)
for col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 5. Clean binary categorical columns to ensure only 0/1
binary_map = {
    'yes': 1, 'no': 0, '1': 1, '0': 0, 'nan': 0, '': 0
}
for col in ['dm', 'htn', 'cad', 'pe', 'ane']:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(binary_map)
        )
        df[col] = df[col].apply(lambda x: 1 if str(x) == '1' else 0)

# For 'appet' (good/poor)
if 'appet' in df.columns:
    df['appet'] = (
        df['appet']
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({'good': 1, 'poor': 0, 'nan': 0, '': 0})
    )
    df['appet'] = df['appet'].apply(lambda x: 1 if str(x) == '1' else 0)

# 6. Encode all features except target (LabelEncoder for multi-class categoricals only)
for col in df.columns:
    if col == 'classification':
        continue
    # Skip columns already forced to 0/1
    if col in ['dm', 'htn', 'cad', 'pe', 'ane', 'appet']:
        continue
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 7. Encode target as well (if not already 0/1)
classification = classification.astype(str).str.lower().str.strip()
classification_mapped = classification.map(lambda x: 0 if x == 'notckd' else (1 if x == 'ckd' else 2))
df['classification'] = classification_mapped

# 8. Save the fully processed DataFrame
df.to_csv("data/discretized_encoded_full.csv", index=False)
print("All features encoded and saved to data/discretized_encoded_full.csv")

# 9. Load preprocessed data
df = pd.read_csv("data/discretized_encoded_full.csv")

# 10. Check for missing values and fill with mode (if any)
for col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 11. Split features and target
X = df.drop(columns=["classification"])
y = df["classification"]

# Handle warning: adjust n_splits if needed
min_class_count = y.value_counts().min()
n_splits = min(5, min_class_count) if min_class_count < 5 else 5

# 12. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 13. Models to train with hyperparameter grids
model_grids = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    ),
    "decision_tree": (
        DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
    ),
    "random_forest": (
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
    )
}

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

results = {}
metrics_bar = {
    "accuracy": [],
    "f1": [],
    "recall": [],
    "precision": [],
    "model": []
}

os.makedirs("models/pictures_discrete", exist_ok=True)

for name, (base_model, param_grid) in model_grids.items():
    # Use RFE for all models
    rfe = RFE(base_model, n_features_to_select=18)
    rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_].tolist()
    print(f"\n{name} selected features by RFE:", selected_features)

    # Hyperparameter tuning with GridSearchCV
    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid.fit(X_train[selected_features], y_train)
    best_model = grid.best_estimator_
    print(f"{name} best params: {grid.best_params_}")

    # Train and evaluate
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    best_model.fit(X_train_sel, y_train)
    y_pred = best_model.predict(X_test_sel)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    test_precision = precision_score(y_test, y_pred, average='macro')
    print(f"{name} Test Accuracy: {test_acc:.4f}")
    print(f"{name} Test F1 Score: {test_f1:.4f}")
    print(f"{name} Test Recall: {test_recall:.4f}")
    print(f"{name} Test Precision: {test_precision:.4f}")
    print(f"{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Save confusion matrix image
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"models/pictures_discrete/{name}_confusion_matrix.png")
    plt.close()

    # Save tree structure image for tree models
    if name == "decision_tree":
        plt.figure(figsize=(16,8))
        plot_tree(best_model, feature_names=selected_features, filled=True, rounded=True, fontsize=10)
        plt.title("Decision Tree Structure")
        plt.savefig("models/pictures_discrete/decision_tree_structure.png")
        plt.close()
    elif name == "random_forest":
        plt.figure(figsize=(16,8))
        plot_tree(best_model.estimators_[0], feature_names=selected_features, filled=True, rounded=True, fontsize=10)
        plt.title("Random Forest (First Tree) Structure")
        plt.savefig("models/pictures_discrete/random_forest_first_tree_structure.png")
        plt.close()

    # K-Fold Cross Validation
    cv_scores = cross_val_score(best_model, X[selected_features], y, cv=skf, scoring='accuracy')
    cv_f1_scores = cross_val_score(best_model, X[selected_features], y, cv=skf, scoring='f1_macro')
    cv_acc_mean = cv_scores.mean()
    cv_acc_std = cv_scores.std()
    cv_f1_mean = cv_f1_scores.mean()
    cv_f1_std = cv_f1_scores.std()
    print(f"{name} K-Fold CV Accuracy: {cv_acc_mean:.4f} (+/- {cv_acc_std:.4f})")
    print(f"{name} K-Fold CV F1 Score: {cv_f1_mean:.4f} (+/- {cv_f1_std:.4f})")

    # Save CV accuracy graph
    plt.figure()
    plt.plot(range(1, len(cv_scores)+1), cv_scores, marker='o', linestyle='-', color='blue')
    plt.title(f"{name} Cross-Validation Fold Accuracies")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f"models/pictures_discrete/{name}_cv_accuracy.png")
    plt.close()

    # Save trained model and selected features for later prediction
    joblib.dump(best_model, f"models/pictures_discrete/{name}_discrete.pkl")
    joblib.dump(selected_features, f"models/pictures_discrete/{name}_selected_features.pkl")

    # Save results
    results[name] = {
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "test_recall": test_recall,
        "test_precision": test_precision,
        "cv_accuracy_mean": cv_acc_mean,
        "cv_accuracy_std": cv_acc_std,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std
    }
    metrics_bar["accuracy"].append(test_acc)
    metrics_bar["f1"].append(test_f1)
    metrics_bar["recall"].append(test_recall)
    metrics_bar["precision"].append(test_precision)
    metrics_bar["model"].append(name)

print("\nAll models trained, tuned, evaluated, and cross-validated.")

# Save results to CSV
pd.DataFrame(results).T.to_csv("data/model_results_discrete.csv")
print("\nModel results saved to data/model_results_discrete.csv")
print("Images saved to models/pictures_discrete/")

# Save bar plots for accuracy, f1, recall, precision
for metric in ["accuracy", "f1", "recall", "precision"]:
    plt.figure(figsize=(6,4))
    plt.bar(metrics_bar["model"], metrics_bar[metric], color=["blue", "green", "orange"])
    plt.ylabel(metric.capitalize())
    plt.title(f"Test {metric.capitalize()} by Model")
    plt.ylim(0, 1)
    plt.savefig(f"models/pictures_discrete/{metric}_bar.png")
    plt.close()