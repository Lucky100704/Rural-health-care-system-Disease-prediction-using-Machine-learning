import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle

data = pd.read_csv('synthetic_asthma_data.csv')
X = data.drop(columns=['asthma'])
y = data['asthma']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='f1_weighted')
grid.fit(X_train_resampled, y_train_resampled)

model = grid.best_estimator_
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("🔍 Model Evaluation Metrics:")
print(f"✅ Accuracy:  {accuracy * 100:.2f}%")
print(f"🎯 Precision: {precision:.2f}")
print(f"🔁 Recall:    {recall:.2f}")
print(f"⚖️ F1 Score:  {f1:.2f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

with open('asthma_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Final model saved as 'asthma_model.pkl'")
