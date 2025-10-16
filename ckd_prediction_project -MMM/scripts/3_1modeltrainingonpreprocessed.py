from base64 import encode
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# # Load dataset
df = pd.read_csv("data/1kidney_disease_preprocessed.csv")
# df.drop(columns=['id'], inplace=True)
df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
df['classification'] = df['classification'].replace({'ckd\t': 'ckd'})

# Convert these columns to numeric
for col in ['pcv', 'wc', 'rc']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Features and target
X = df.drop(columns=['classification'])
y = df['classification'].map({'ckd': 1, 'notckd': 0})

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Impute missing values
num_imputer = SimpleImputer(strategy='mean')
X[num_cols] = num_imputer.fit_transform(X[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# encode categorical columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['notckd', 'ckd'], yticklabels=['notckd', 'ckd'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report Visualization
report = classification_report(y_test, y_pred, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
class_labels = [0, 1]  # 0: notckd, 1: ckd
class_names = ['notckd', 'ckd']
scores = [[report[str(c)][m] for m in metrics] for c in class_labels]

df_scores = pd.DataFrame(scores, columns=metrics, index=class_names)

# Add accuracy as a separate bar
accuracy = accuracy_score(y_test, y_pred)
df_scores.loc['accuracy'] = [accuracy, accuracy, accuracy]

df_scores.plot(kind='bar')
plt.title('Classification Report Metrics (including Accuracy)')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.show()

# Save everything
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(num_imputer, "models/num_imputer.pkl")
joblib.dump(cat_imputer, "models/cat_imputer.pkl")
joblib.dump(encoders, "models/encoder.pkl")
# Example of saving feature names when training
feature_columns = X_train.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.pkl')
