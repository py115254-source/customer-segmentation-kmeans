
# Customer Segmentation using K-Means Clustering

## Project Overview
This project groups mall customers into different segments using the K-Means clustering algorithm. It uses annual income and spending score to identify customer groups.

## Objectives
- Load the Mall Customers dataset.
- Select annual income and spending score features.
- Standardize numerical features.
- Find the optimal number of clusters using silhouette score.
- Apply K-Means clustering.
- Add cluster labels to the dataset.
- Export the results to a CSV file.

## Technologies Used
- Python
- Pandas
- Scikit-learn

## Methodology
1. Load the customer dataset.
2. Select the required features.
3. Handle missing feature values.
4. Standardize the features using StandardScaler.
5. Compare cluster counts from 2 to 10 using silhouette score.
6. Train K-Means using the best-scoring cluster count.
7. Add cluster labels to the original dataframe.
8. Export the final dataframe to customer_segments.csv.

## Results
- Total customers: 200
- Selected number of clusters: 5
- Selection method: Silhouette Score
- Output file: customer_segments.csv

The selected cluster count is based on the highest silhouette score among the tested values.

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the project:

python main.py

## Output
The generated customer_segments.csv file contains the original customer data and a Cluster column with the assigned cluster labels.
