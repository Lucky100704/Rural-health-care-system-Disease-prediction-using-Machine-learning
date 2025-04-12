import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Create a synthetic binary classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    weights=[0.65, 0.35],  # simulate class imbalance
    random_state=42
)

# Convert to DataFrame
feature_names = [f'feature_{i}' for i in range(1, 11)]
data = pd.DataFrame(X, columns=feature_names)
data['asthma'] = y

# Save to CSV (optional)
data.to_csv('synthetic_asthma_data.csv', index=False)
print("✅ Synthetic asthma dataset created and saved as 'synthetic_asthma_data.csv'")
