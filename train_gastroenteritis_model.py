# train_gastroenteritis_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the dataset
data = pd.read_csv('gastroenteritis_data.csv')  # Replace with your actual path if different

# Define features and target
X = data[['fever', 'vomiting', 'diarrhea', 'abdominal_pain', 'bloody_diarrhea']]
y = data['type']  # 0 = Viral, 1 = Bacterial

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics
print("Model Evaluation Metrics:")
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")

# Save the trained model
joblib.dump(model, 'gastroenteritis_model.pkl')
print("Model saved as 'gastroenteritis_model.pkl'")
