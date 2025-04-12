import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Step 1: Create a dataset (you can replace this with a larger dataset)
data = {
    "Fever": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    "Cough": [1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    "Sore_Throat": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    "Runny_Nose": [0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    "Headache": [1, 1, 0, 0, 0, 1, 1, 0, 1, 0],
    "Cold_Type": ["Viral", "Viral", "Bacterial", "Bacterial", "Allergy", "Viral", "Allergy", "Bacterial", "Viral", "Allergy"]
}

# Step 2: Prepare the data
df = pd.DataFrame(data)
X = df.drop("Cold_Type", axis=1)  # Features (symptoms)
y = df["Cold_Type"]  # Target (cold type)

# Step 3: Encode the target variable into numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Step 5: Train a RandomForest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # for multi-class
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Step 8: Print evaluation metrics
print("Model Evaluation Metrics:")
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")

# Step 9: Save both the model and label encoder
model_and_encoder = {"model": model, "label_encoder": label_encoder}
joblib.dump(model_and_encoder, 'cold_type_model_and_encoder.pkl')
print("Model and label encoder saved as 'cold_type_model_and_encoder.pkl'")
