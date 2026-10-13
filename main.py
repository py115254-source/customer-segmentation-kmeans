
"""Customer segmentation using K-Means clustering."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Project constants
DATA_FILE = Path("Mall_Customers.csv")
OUTPUT_FILE = Path("customer_segments.csv")
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42
MIN_K = 2
MAX_K = 10


def load_data(file_path):
    """Load customer data from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Add the Mall Customers dataset."
        )
    return pd.read_csv(file_path)


def find_optimal_k(scaled_features):
    """Select the best cluster count using silhouette score."""
    best_k = MIN_K
    best_score = -1

    for k in range(MIN_K, min(MAX_K, len(scaled_features) - 1) + 1):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)

        print(f"k={k}, silhouette score={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def main():
    """Run the customer segmentation pipeline."""
    customers = load_data(DATA_FILE)
    print("Dataset loaded:", customers.shape)

    missing = [feature for feature in FEATURES if feature not in customers.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features = customers[FEATURES].copy()
    features = features.fillna(features.median())

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    print("\nFinding optimal number of clusters...")
    optimal_k = find_optimal_k(scaled_features)
    print("Selected k:", optimal_k)

    model = KMeans(
        n_clusters=optimal_k,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    customers["Cluster"] = model.fit_predict(scaled_features)

    customers.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("\nCustomers in each cluster:")
    print(customers["Cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()
