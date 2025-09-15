import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset
def load_data():
    # Assuming the file is named 'Mall_Customers.csv' in the same directory
    try:
        df = pd.read_csv('Mall_Customers.csv')
        print("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print("File not found. Please make sure 'Mall_Customers.csv' is in the current directory.")
        return None

# Exploratory Data Analysis
def explore_data(df):
    print("=" * 50)
    print("DATASET INFO:")
    print("=" * 50)
    print(df.info())
    
    print("\n" + "=" * 50)
    print("DATASET DESCRIPTION:")
    print("=" * 50)
    print(df.describe())
    
    print("\n" + "=" * 50)
    print("FIRST 5 ROWS:")
    print("=" * 50)
    print(df.head())
    
    print("\n" + "=" * 50)
    print("NULL VALUES CHECK:")
    print("=" * 50)
    print(df.isnull().sum())
    
    # Check for duplicate entries
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# Data Visualization
def visualize_data(df):
    # Distribution of Age, Annual Income, and Spending Score
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(df['Age'], kde=True, ax=axes[0])
    axes[0].set_title('Distribution of Age')
    
    sns.histplot(df['Annual Income (k$)'], kde=True, ax=axes[1])
    axes[1].set_title('Distribution of Annual Income (k$)')
    
    sns.histplot(df['Spending Score (1-100)'], kde=True, ax=axes[2])
    axes[2].set_title('Distribution of Spending Score (1-100)')
    
    plt.tight_layout()
    plt.show()
    
    # Gender distribution
    plt.figure(figsize=(6, 6))
    gender_counts = df['Genre'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Gender Distribution')
    plt.show()
    
    # Relationship between Annual Income and Spending Score
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Genre')
    plt.title('Annual Income vs Spending Score')
    plt.show()

# Data Preprocessing
def preprocess_data(df):
    # Select features for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled

# Determine optimal number of clusters using Elbow Method
def find_optimal_clusters(X_scaled):
    wcss = []  # Within-Cluster Sum of Square
    silhouette_scores = []
    cluster_range = range(2, 11)
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
    
    # Plot elbow method
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(cluster_range, wcss, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    
    plt.subplot(1, 2, 2)
    plt.plot(cluster_range, silhouette_scores, 'go-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    
    plt.tight_layout()
    plt.show()
    
    # Find the optimal number of clusters (elbow point)
    differences = [wcss[i] - wcss[i+1] for i in range(len(wcss)-1)]
    differences_ratio = [differences[i] / differences[i+1] for i in range(len(differences)-1)]
    optimal_clusters = differences_ratio.index(max(differences_ratio)) + 2  # +2 because we started from 2 clusters
    
    print(f"Optimal number of clusters based on elbow method: {optimal_clusters}")
    
    return optimal_clusters

# Apply K-Means clustering
def apply_kmeans(X, X_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the original dataframe
    X['Cluster'] = clusters
    
    return kmeans, X

# Visualize clusters
def visualize_clusters(X, kmeans):
    plt.figure(figsize=(12, 8))
    
    for cluster in range(kmeans.n_clusters):
        plt.scatter(X[X['Cluster'] == cluster]['Annual Income (k$)'], 
                    X[X['Cluster'] == cluster]['Spending Score (1-100)'], 
                    label=f'Cluster {cluster}', s=100)
    
    # Plot centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=300, c='yellow', label='Centroids', marker='X')
    
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Customer Segments')
    plt.legend()
    plt.grid(True)
    plt.show()

# Analyze clusters
def analyze_clusters(df, X):
    # Add cluster labels to the original dataframe
    df['Cluster'] = X['Cluster']
    
    print("=" * 50)
    print("CLUSTER ANALYSIS:")
    print("=" * 50)
    
    # Count customers in each cluster
    cluster_counts = df['Cluster'].value_counts().sort_index()
    print("Number of customers in each cluster:")
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} customers")
    
    # Average spending score and income per cluster
    cluster_analysis = df.groupby('Cluster').agg({
        'Annual Income (k$)': 'mean',
        'Spending Score (1-100)': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Count'})
    
    print("\nAverage values per cluster:")
    print(cluster_analysis.round(2))
    
    return df

# BONUS: Try DBSCAN clustering
def try_dbscan(X_scaled):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_clusters = dbscan.fit_predict(X_scaled)
    
    # Count unique clusters (excluding noise points labeled as -1)
    n_clusters = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
    n_noise = list(dbscan_clusters).count(-1)
    
    print(f"\nDBSCAN found {n_clusters} clusters and {n_noise} noise points.")
    
    return dbscan_clusters

# BONUS: Visualize DBSCAN results
def visualize_dbscan(X, dbscan_clusters):
    plt.figure(figsize=(12, 8))
    
    # Create a scatter plot with different colors for each cluster
    unique_labels = set(dbscan_clusters)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise
            col = [0, 0, 0, 1]
        
        class_member_mask = (dbscan_clusters == k)
        
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=8, label=f'Cluster {k}' if k != -1 else 'Noise')
    
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('DBSCAN Clustering')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    print("CUSTOMER SEGMENTATION WITH K-MEANS CLUSTERING")
    print("=" * 50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Explore data
    explore_data(df)
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess data
    X, X_scaled = preprocess_data(df)
    
    # Find optimal number of clusters
    optimal_clusters = find_optimal_clusters(X_scaled)
    
    # Apply K-Means
    kmeans, X_with_clusters = apply_kmeans(X.copy(), X_scaled, optimal_clusters)
    
    # Visualize clusters
    visualize_clusters(X_with_clusters, kmeans)
    
    # Analyze clusters
    df_with_clusters = analyze_clusters(df.copy(), X_with_clusters)
    
    # BONUS: Try DBSCAN
    print("\n" + "=" * 50)
    print("BONUS: DBSCAN CLUSTERING")
    print("=" * 50)
    
    dbscan_clusters = try_dbscan(X_scaled)
    
    # Visualize DBSCAN results
    visualize_dbscan(X_scaled, dbscan_clusters)

if __name__ == "__main__":
    main()