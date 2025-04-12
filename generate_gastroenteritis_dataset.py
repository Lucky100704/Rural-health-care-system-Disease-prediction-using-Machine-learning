# generate_gastroenteritis_dataset.py

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
data_size = 500
data = pd.DataFrame({
    'fever': np.random.choice([0, 1], size=data_size),  # 0 = No, 1 = Yes
    'vomiting': np.random.choice([0, 1], size=data_size),  # 0 = No, 1 = Yes
    'diarrhea': np.random.choice([0, 1], size=data_size),  # 0 = No, 1 = Yes
    'abdominal_pain': np.random.choice([0, 1], size=data_size),  # 0 = No, 1 = Yes
    'bloody_diarrhea': np.random.choice([0, 1], size=data_size),  # 0 = No, 1 = Yes (for bacterial)
    'type': np.random.choice([0, 1], size=data_size, p=[0.7, 0.3])  # 0 = Viral, 1 = Bacterial
})

# Save dataset to CSV for reuse
data.to_csv('gastroenteritis_data.csv', index=False)

print("Synthetic dataset created and saved as 'gastroenteritis_data.csv'!")
