# 📘 Kickstarter Project Success Prediction and Clustering

This repository contains a comprehensive analysis of Kickstarter projects using **machine learning** techniques. The goal is twofold:

- **Classification models** to predict project success or failure before launch.  
- **Clustering methods** to uncover hidden patterns and group similar projects for deeper insights.  

The project is designed to assist **creators** in strategizing their campaigns and **platform managers** in understanding project dynamics.  

---

## 📂 Project Structure

- **`Individual_Assignment_Complete_Code-Julien_Palummo.py`** → Main Python script containing the full pipeline: data cleaning, feature engineering, classification, clustering, and visualization.  
- **`Palummo_Julien_IndividualProject.pdf`** → Detailed project report with methodology, results, and interpretations.  

---

## ⚙️ Features and Workflow  

### 1. Data Preparation & Feature Engineering  
- Removed irrelevant and post-launch variables (e.g., pledges, backers).  
- Filtered dataset to only include *successful* and *failed* projects.  
- Handled missing values in categorical features.  
- Removed outliers using **Isolation Forest**.  
- Created new variables such as `goal_usd` (standardized project goal).  
- Reduced multicollinearity by dropping highly correlated features.  
- Grouped countries into **US, GB, CA, AU, and Others**.  
- Encoded categorical variables and normalized numeric ones.  

### 2. Classification Models  
Tested three supervised learning algorithms:  
- **Random Forest** → Accuracy ≈ 74.6%  
- **KNN** → Accuracy ≈ 67.9%  
- **Gradient Boosting** → Accuracy ≈ 75.3%  

Hyperparameter tuning with **GridSearchCV** improved performance:  
- Random Forest (**best consistency**) → ~75.1%  
- Gradient Boosting (**best single performance**) → ~75.7%  

### 3. Clustering Analysis  
Explored unsupervised techniques to find project groupings:  
- **K-Means** (selected method, k=6) – best silhouette score.  
- **Hierarchical Clustering** and **DBSCAN** tested but less effective.  

**Cluster Insights:**  
- Typical/average projects.  
- Projects launched early in the month.  
- Longer preparation before launch.  
- Higher funding goals (ambitious projects).  
- Early-day launches to capture traffic.  
- Longer campaign durations.  

---

## 📊 Visualizations  
- Correlation heatmaps  
- Elbow method for optimal k  
- PCA 2D projections of clusters  
- Cluster feature deviations (bar plots)  
- Category and country distributions across clusters  
- Dendrograms for hierarchical clustering  

---

## 📈 Results & Key Findings  
- **Gradient Boosting** is the strongest classifier, but **Random Forest** offers stable results across random states.  
- Clustering reveals subtle but important behavioral differences (e.g., timing of launches, campaign duration).  
- Project success is influenced more by **pre-launch decisions** than by post-launch activity.  

---

## 🛠️ Tech Stack  
- **Languages:** Python  
- **Libraries:** `pandas`, `scikit-learn`, `seaborn`, `matplotlib`, `scipy`  
- **Techniques:** Classification, Clustering, Outlier Removal, Feature Engineering, Hyperparameter Tuning  
