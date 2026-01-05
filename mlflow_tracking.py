"""
FILE: mlflow_tracking.py
RUN AS: Python script (.py)
COMMAND: python mlflow_tracking.py

This script tracks all experiments using MLflow.
After running, use 'mlflow ui' to view results.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
import os
warnings.filterwarnings('ignore')

print("="*70)
print("MLFLOW EXPERIMENT TRACKING - PatrolIQ")
print("="*70)

# ============================================================================
# 0. CHECK AND CREATE DATA IF MISSING
# ============================================================================
print("\n[0/6] Checking for required data files...")

required_files = ['features_geographic.csv', 'features_temporal.csv', 'features_combined.csv']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print(f"⚠️  Missing files: {missing_files}")
    print("Creating sample data...")
    
    # Create sample data
    np.random.seed(42)
    n = 5000
    
    # Geographic features
    df_geo = pd.DataFrame({
        'Latitude': np.random.uniform(41.6, 42.0, n),
        'Longitude': np.random.uniform(-87.9, -87.5, n),
        'Lat_Lon_Cluster': np.random.randint(0, 8, n)
    })
    df_geo.to_csv('features_geographic.csv', index=False)
    
    # Temporal features
    df_temporal = pd.DataFrame({
        'Hour': np.random.randint(0, 24, n),
        'Day_of_Week': np.random.randint(0, 7, n),
        'Month': np.random.randint(1, 13, n),
        'Is_Weekend': np.random.choice([0, 1], n),
        'Is_Night': np.random.choice([0, 1], n),
        'Hour_Sin': np.sin(2 * np.pi * np.random.randint(0, 24, n) / 24),
        'Hour_Cos': np.cos(2 * np.pi * np.random.randint(0, 24, n) / 24)
    })
    df_temporal.to_csv('features_temporal.csv', index=False)
    
    # Combined features
    crime_types = ['THEFT', 'BATTERY', 'ASSAULT', 'BURGLARY', 'ROBBERY']
    df_combined = df_temporal.copy()
    df_combined['Latitude'] = df_geo['Latitude']
    df_combined['Longitude'] = df_geo['Longitude']
    df_combined['Crime_Severity_Score'] = np.random.uniform(1, 10, n)
    df_combined['Population_Density'] = np.random.uniform(1000, 15000, n)
    df_combined['Avg_Income'] = np.random.uniform(25000, 120000, n)
    
    for crime in crime_types:
        df_combined[f'Crime_{crime}'] = np.random.choice([0, 1], n)
    
    df_combined.to_csv('features_combined.csv', index=False)
    
    print("✓ Sample data created")
else:
    print("✓ All required files found")

# ============================================================================
# 1. SETUP MLFLOW
# ============================================================================
print("\n[1/6] Setting up MLflow...")

mlflow.set_tracking_uri("file:./mlruns")
experiment_name = "PatrolIQ_Crime_Analysis"

try:
    mlflow.create_experiment(experiment_name)
    print(f"✓ Created new experiment '{experiment_name}'")
except:
    print(f"✓ Using existing experiment '{experiment_name}'")

mlflow.set_experiment(experiment_name)
print(f"✓ Tracking URI: {mlflow.get_tracking_uri()}")

# ============================================================================
# 2. LOAD DATA
# ============================================================================
print("\n[2/6] Loading data...")

df_geo = pd.read_csv('features_geographic.csv')
df_temporal = pd.read_csv('features_temporal.csv')
df_combined = pd.read_csv('features_combined.csv')

print(f"✓ Geographic features: {df_geo.shape}")
print(f"✓ Temporal features: {df_temporal.shape}")
print(f"✓ Combined features: {df_combined.shape}")

# ============================================================================
# 3. TRACK GEOGRAPHIC CLUSTERING
# ============================================================================
print("\n[3/6] Tracking geographic clustering experiments...")

X_geo = df_geo.values
X_geo = np.nan_to_num(X_geo, nan=0.0)

# K-Means with different K values
print("\n  K-Means Geographic:")
for k in [5, 7, 8, 10]:
    with mlflow.start_run(run_name=f"Geographic_KMeans_K{k}"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_geo)
        
        # Log parameters
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("cluster_type", "Geographic")
        mlflow.log_param("n_clusters", k)
        mlflow.log_param("n_samples", len(X_geo))
        mlflow.log_param("n_features", X_geo.shape[1])
        
        # Log metrics
        silhouette = silhouette_score(X_geo, labels)
        davies_bouldin = davies_bouldin_score(X_geo, labels)
        calinski = calinski_harabasz_score(X_geo, labels)
        
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("davies_bouldin_index", davies_bouldin)
        mlflow.log_metric("calinski_harabasz_score", calinski)
        
        # Log model
        mlflow.sklearn.log_model(kmeans, "model")
        
        print(f"    K={k}: Silhouette={silhouette:.4f}, Davies-Bouldin={davies_bouldin:.4f}")

# DBSCAN with different parameters
print("\n  DBSCAN Geographic:")
for eps in [0.1, 0.15, 0.2]:
    with mlflow.start_run(run_name=f"Geographic_DBSCAN_eps{eps}"):
        dbscan = DBSCAN(eps=eps, min_samples=50)
        labels = dbscan.fit_predict(X_geo)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Log parameters
        mlflow.log_param("algorithm", "DBSCAN")
        mlflow.log_param("cluster_type", "Geographic")
        mlflow.log_param("eps", eps)
        mlflow.log_param("min_samples", 50)
        
        # Log metrics
        mlflow.log_metric("n_clusters", n_clusters)
        mlflow.log_metric("n_noise_points", n_noise)
        mlflow.log_metric("noise_percentage", n_noise/len(labels)*100 if len(labels) > 0 else 0)
        
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 1:
                try:
                    silhouette = silhouette_score(X_geo[mask], labels[mask])
                    davies_bouldin = davies_bouldin_score(X_geo[mask], labels[mask])
                    calinski = calinski_harabasz_score(X_geo[mask], labels[mask])
                    
                    mlflow.log_metric("silhouette_score", silhouette)
                    mlflow.log_metric("davies_bouldin_index", davies_bouldin)
                    mlflow.log_metric("calinski_harabasz_score", calinski)
                    
                    print(f"    eps={eps}: Clusters={n_clusters}, Silhouette={silhouette:.4f}")
                except:
                    print(f"    eps={eps}: Clusters={n_clusters}, Noise={n_noise}")
            else:
                print(f"    eps={eps}: Clusters={n_clusters}, Noise={n_noise}")

# Hierarchical with different K values
print("\n  Hierarchical Geographic:")
for k in [5, 7, 8, 10]:
    with mlflow.start_run(run_name=f"Geographic_Hierarchical_K{k}"):
        hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = hierarchical.fit_predict(X_geo)
        
        # Log parameters
        mlflow.log_param("algorithm", "Hierarchical")
        mlflow.log_param("cluster_type", "Geographic")
        mlflow.log_param("n_clusters", k)
        mlflow.log_param("linkage", "ward")
        
        # Log metrics
        silhouette = silhouette_score(X_geo, labels)
        davies_bouldin = davies_bouldin_score(X_geo, labels)
        calinski = calinski_harabasz_score(X_geo, labels)
        
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("davies_bouldin_index", davies_bouldin)
        mlflow.log_metric("calinski_harabasz_score", calinski)
        
        print(f"    K={k}: Silhouette={silhouette:.4f}")

print("✓ Geographic clustering experiments tracked")

# ============================================================================
# 4. TRACK TEMPORAL CLUSTERING
# ============================================================================
print("\n[4/6] Tracking temporal clustering experiments...")

X_temporal = df_temporal.values
X_temporal = np.nan_to_num(X_temporal, nan=0.0)

# K-Means temporal
print("\n  K-Means Temporal:")
for k in [3, 5, 7]:
    with mlflow.start_run(run_name=f"Temporal_KMeans_K{k}"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_temporal)
        
        # Log parameters
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("cluster_type", "Temporal")
        mlflow.log_param("n_clusters", k)
        mlflow.log_param("n_samples", len(X_temporal))
        mlflow.log_param("n_features", X_temporal.shape[1])
        
        # Log metrics
        silhouette = silhouette_score(X_temporal, labels)
        davies_bouldin = davies_bouldin_score(X_temporal, labels)
        calinski = calinski_harabasz_score(X_temporal, labels)
        
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("davies_bouldin_index", davies_bouldin)
        mlflow.log_metric("calinski_harabasz_score", calinski)
        
        # Log model
        mlflow.sklearn.log_model(kmeans, "model")
        
        print(f"    K={k}: Silhouette={silhouette:.4f}")

# Hierarchical temporal
print("\n  Hierarchical Temporal:")
for k in [3, 5, 7]:
    with mlflow.start_run(run_name=f"Temporal_Hierarchical_K{k}"):
        hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = hierarchical.fit_predict(X_temporal)
        
        # Log parameters
        mlflow.log_param("algorithm", "Hierarchical")
        mlflow.log_param("cluster_type", "Temporal")
        mlflow.log_param("n_clusters", k)
        mlflow.log_param("linkage", "ward")
        
        # Log metrics
        silhouette = silhouette_score(X_temporal, labels)
        davies_bouldin = davies_bouldin_score(X_temporal, labels)
        calinski = calinski_harabasz_score(X_temporal, labels)
        
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("davies_bouldin_index", davies_bouldin)
        mlflow.log_metric("calinski_harabasz_score", calinski)
        
        print(f"    K={k}: Silhouette={silhouette:.4f}")

print("✓ Temporal clustering experiments tracked")

# ============================================================================
# 5. TRACK DIMENSIONALITY REDUCTION
# ============================================================================
print("\n[5/6] Tracking dimensionality reduction experiments...")

X_combined = df_combined.values
X_combined = np.nan_to_num(X_combined, nan=0.0)

# PCA with different components
print("\n  PCA:")
for n_comp in [2, 3, 5, 10, 15]:
    n_comp = min(n_comp, X_combined.shape[1])  # Ensure valid n_components
    
    with mlflow.start_run(run_name=f"PCA_n{n_comp}"):
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(X_combined)
        
        # Log parameters
        mlflow.log_param("algorithm", "PCA")
        mlflow.log_param("n_components", n_comp)
        mlflow.log_param("n_samples", X_combined.shape[0])
        mlflow.log_param("n_features", X_combined.shape[1])
        
        # Log metrics
        explained_variance = pca.explained_variance_ratio_
        total_variance = np.sum(explained_variance)
        
        mlflow.log_metric("total_explained_variance", total_variance)
        mlflow.log_metric("variance_percentage", total_variance * 100)
        
        for i, var in enumerate(explained_variance):
            mlflow.log_metric(f"PC{i+1}_variance", var)
        
        # Log model
        mlflow.sklearn.log_model(pca, "model")
        
        print(f"    n_components={n_comp}: Total Variance={total_variance*100:.2f}%")

print("✓ Dimensionality reduction experiments tracked")

# ============================================================================
# 6. RETRIEVE AND DISPLAY BEST MODELS
# ============================================================================
print("\n[6/6] Analyzing results...")
print("\n" + "="*70)
print("BEST PERFORMING MODELS")
print("="*70)

try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Best clustering algorithms
    clustering_runs = runs[runs['metrics.silhouette_score'].notna()].copy()
    
    if len(clustering_runs) > 0:
        best_runs = clustering_runs.nlargest(5, 'metrics.silhouette_score')
        
        print("\n🏆 Top 5 Clustering Algorithms (by Silhouette Score):")
        print("-" * 70)
        for idx, (i, row) in enumerate(best_runs.iterrows(), 1):
            cluster_type = row.get('params.cluster_type', 'N/A')
            algorithm = row.get('params.algorithm', 'N/A')
            silhouette = row.get('metrics.silhouette_score', 0)
            davies = row.get('metrics.davies_bouldin_index', 0)
            n_clusters = row.get('params.n_clusters', 'N/A')
            
            print(f"\n{idx}. {cluster_type} - {algorithm}")
            print(f"   Clusters: {n_clusters}")
            print(f"   Silhouette Score: {silhouette:.4f}")
            print(f"   Davies-Bouldin Index: {davies:.4f}")
    
    # Best PCA
    pca_runs = runs[runs['params.algorithm'] == 'PCA'].copy()
    if len(pca_runs) > 0:
        best_pca = pca_runs.nlargest(1, 'metrics.total_explained_variance')
        
        print("\n" + "-" * 70)
        print("\n🎯 Best PCA Configuration:")
        for idx, row in best_pca.iterrows():
            n_comp = row.get('params.n_components', 'N/A')
            variance = row.get('metrics.total_explained_variance', 0)
            print(f"   Components: {n_comp}")
            print(f"   Variance Explained: {variance*100:.2f}%")
    
    # Save all runs
    runs.to_csv('mlflow_all_experiments.csv', index=False)
    print("\n" + "-" * 70)
    print("✓ All experiments saved to 'mlflow_all_experiments.csv'")
    print(f"✓ Total runs tracked: {len(runs)}")
    
except Exception as e:
    print(f"⚠️  Error retrieving results: {str(e)}")

# ============================================================================
# FINAL INSTRUCTIONS
# ============================================================================
print("\n" + "="*70)
print("✅ MLFLOW TRACKING COMPLETED!")
print("="*70)
print("\n📊 To view results in MLflow UI:")
print("   1. Open a new terminal")
print("   2. Navigate to project directory")
print("   3. Run: mlflow ui")
print("   4. Open browser: http://localhost:5000")
print("\n💡 Tips:")
print("   - Compare runs by selecting multiple experiments")
print("   - Sort by metrics to find best models")
print("   - View model artifacts and parameters")
print("\n🚀 Next Steps:")
print("   - Analyze results in MLflow UI")
print("   - Select best models for deployment")
print("   - Run visualization scripts")
print("="*70)