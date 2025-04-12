import pandas as pd
import numpy as np

# Number of synthetic samples
num_samples = 1000

# Generate random systolic and diastolic values within typical BP ranges
np.random.seed(42)
systolic = np.random.randint(90, 180, num_samples)  # Random systolic values (90-179)
diastolic = np.random.randint(60, 120, num_samples)  # Random diastolic values (60-119)

# Assign BP categories based on typical ranges
def categorize_bp(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return 'Normal'
    elif 120 <= systolic < 130 and diastolic < 80:
        return 'Elevated'
    elif 130 <= systolic < 140 or 80 <= diastolic < 90:
        return 'Hypertension Stage 1'
    elif 140 <= systolic or 90 <= diastolic:
        return 'Hypertension Stage 2'
    else:
        return 'Unknown'

# Apply the categorization function
target = [categorize_bp(s, d) for s, d in zip(systolic, diastolic)]

# Create a DataFrame
data = pd.DataFrame({
    'systolic': systolic,
    'diastolic': diastolic,
    'target': target
})

# Save the dataset as a CSV file
data.to_csv('blood_pressure_data.csv', index=False)

print("Synthetic dataset generated and saved as 'blood_pressure_data.csv'")
