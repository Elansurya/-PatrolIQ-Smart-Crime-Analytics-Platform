# 🚓 PatrolIQ – Smart Safety Analytics Platform

## 🚀 Project Overview

PatrolIQ is an end-to-end Urban Crime Intelligence Platform designed to analyze large-scale crime data using unsupervised machine learning techniques.

Built using 500,000 crime records sampled from Chicago’s 7.8 million public crime dataset, the system identifies crime hotspots, temporal patterns, and high-risk zones to optimize police resource allocation.

The platform integrates clustering algorithms, dimensionality reduction, MLflow experiment tracking, and a multi-page Streamlit web application deployed on Streamlit Cloud.

---

## 🎯 Business Problem

Urban law enforcement agencies face challenges in:

- Efficient patrol deployment  
- Identifying high-risk crime zones  
- Understanding peak crime hours  
- Making data-driven safety decisions  

Manual analysis of massive crime datasets is inefficient.

PatrolIQ transforms 500K crime records into actionable intelligence that supports:

- 60% faster patrol route optimization  
- Evidence-based resource allocation  
- Real-time risk assessment  
- Safer urban planning strategies  

---

## 📊 Dataset Overview

Source: Chicago Crime Dataset (Public Data Portal)  

- Full Dataset: 7.8 Million Records (2001–2025)  
- Sample Used: 500,000 Recent Records  
- Features: 22 Core Variables  
- Crime Types: 33 Categories  
- Geographic Coverage: Chicago districts & wards  

### Key Feature Categories:
- Crime Identification (ID, IUCR, FBI Code)  
- Crime Classification (Primary Type, Description)  
- Geographic Coordinates (Latitude, Longitude)  
- Administrative Boundaries (District, Ward, Beat)  
- Temporal Variables (Date, Year, Hour, Season)  
- Arrest & Domestic Indicators  

---

## 🧠 Feature Engineering

Created advanced engineered features:

- Hour of Day  
- Day of Week  
- Weekend Flag  
- Seasonal Classification  
- Crime Severity Score  
- Geographic normalization for clustering  
- Encoded crime types & location categories  

---

## 🤖 Unsupervised Learning – Clustering Analysis

### 1️⃣ Geographic Hotspot Detection

Implemented and compared 3 clustering algorithms:

- K-Means Clustering  
- DBSCAN (Density-Based Clustering)  
- Hierarchical Clustering  

Evaluation Metrics:
- Silhouette Score (> 0.5 target)  
- Davies-Bouldin Index  
- Elbow Method  

Results:
- Identified 5–10 high-risk crime zones  
- Generated color-coded risk heatmaps  
- Detected dense crime clusters & filtered noise  

---

### 2️⃣ Temporal Pattern Clustering

- K-Means on time-based features  
- Identified 3–5 major time-based crime patterns  
- Detected peak crime hours (e.g., late-night incidents)  
- Compared weekday vs weekend patterns  
- Generated hourly crime heatmaps  

---

## 📉 Dimensionality Reduction

### PCA (Principal Component Analysis)

- Reduced 22+ features to 2–3 principal components  
- Maintained 70%+ variance  
- Identified top drivers of crime patterns  
- Generated scree plot & component importance analysis  

### t-SNE Visualization

- Created 2D cluster visualization  
- Showed natural separation of crime types  
- Validated clustering structure  

---

## 🔍 MLflow Integration

- Tracked clustering parameters (K values, distance metrics)  
- Logged evaluation metrics  
- Compared model performance  
- Implemented experiment version control  
- Selected best-performing clustering algorithm for deployment  

---

## 🖥️ Streamlit Application

Multi-page interactive dashboard including:

- 🗺 Geographic Crime Heatmap  
- ⏰ Temporal Pattern Dashboard  
- 📊 Cluster Comparison View  
- 📉 PCA & t-SNE Visualization Page  
- 📈 Model Performance Monitoring  
- MLflow Experiment Viewer  

---

## ☁️ Deployment

- Fully deployed on Streamlit Cloud  
- GitHub-integrated CI/CD pipeline  
- Responsive & production-ready interface  
- Error handling & optimized performance  

---

## 🏗️ Architecture

Chicago Crime Dataset (7.8M)  
↓  
Sampling (500K Records)  
↓  
Data Cleaning & Feature Engineering  
↓  
Clustering (K-Means / DBSCAN / Hierarchical)  
↓  
Dimensionality Reduction (PCA + t-SNE)  
↓  
MLflow Experiment Tracking  
↓  
Streamlit Application  
↓  
Cloud Deployment  

---

## ⚙️ Tech Stack

Python  
Pandas & NumPy  
Scikit-learn  
K-Means, DBSCAN, Hierarchical Clustering  
PCA, t-SNE  
MLflow  
Streamlit  
Geographic Data Analysis  

Domain: Public Safety & Urban Analytics  

---

## 📈 Impact & Applications

- Identified actionable crime hotspots  
- Enabled evidence-based patrol deployment  
- Visualized complex crime patterns clearly  
- Improved situational awareness for decision-makers  
- Applicable to police departments, city planning, and emergency response systems  

---

## 📌 Key Learnings

- Large-scale dataset handling (500K records)  
- Unsupervised learning evaluation techniques  
- Spatial clustering strategies  
- Dimensionality reduction for visualization  
- MLflow experiment management  
- Production-level dashboard deployment  

---

## 🔮 Future Enhancements

- Crime prediction using supervised ML  
- Real-time streaming crime analytics  
- GIS map integration (Leaflet / Folium)  
- REST API for external integration  
- Cloud database optimization  

---

## 👨‍💻 Author
Elansurya K  
Data Scientist | Machine Learning | NLP | SQL
