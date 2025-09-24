import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble import IsolationForest
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN



# Load the dataset
file_path = 'C:/Users/julie/Downloads/Kickstarter (1).xlsx'   ### Please replace this with the testing dataset
kickstarter_data = pd.read_excel(file_path)

#Remove useless columns
kickstarter_data = kickstarter_data.drop(columns=["id","deadline", "created_at","launched_at", "currency"])

# Remove post-launch variables
post_launch_vars = ['pledged', 'usd_pledged','staff_pick','spotlight','disable_communication', 'backers_count', 'state_changed_at', 'state_changed_at_month',
                    'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr', 
                    'launch_to_state_change_days','state_changed_at_weekday']

kickstarter_data = kickstarter_data.drop(columns=post_launch_vars)

# Filter observations where state is either 'successful' or 'failure'
kickstarter_data = kickstarter_data[kickstarter_data['state'].isin(['successful', 'failed'])]

# Remove rows with missing 'category'
kickstarter_data = kickstarter_data.dropna(subset=['category'])

#Remove Name
kickstarter_data = kickstarter_data.drop(columns=["name"])

## Removing Outliers
def remove_outliers_isolation_forest(df, contamination_factor=0.1):
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Initialize Isolation Forest
    iso_forest = IsolationForest(contamination=contamination_factor, random_state=42)

    # Fit and predict
    outliers = iso_forest.fit_predict(df[numeric_cols])

    # Filter out the outliers
    df_filtered = df[outliers != -1]
    return df_filtered

# Applying the function with a default contamination factor of 10%
kickstarter_data = remove_outliers_isolation_forest(kickstarter_data)

# Feature Engineering
kickstarter_data['goal_usd'] = kickstarter_data['goal'] * kickstarter_data['static_usd_rate']
kickstarter_data = kickstarter_data.drop(columns=['goal', 'static_usd_rate'])

## Correlation Heatmap:
# Calculating the correlation matrix
correlation_matrix = kickstarter_data.corr()

# Setting up the matplotlib figure
plt.figure(figsize=(12, 10))

# Drawing the heatmap
sns.heatmap(correlation_matrix, 
            cmap='coolwarm', 
            annot=False, 
            fmt=".2f", 
            linewidths=.5)

# Adding title
plt.title('Correlation Heatmap of Kickstarter Dataset')

# Showing the plot
plt.show()

# Variables to remove: We choose one variable from each highly correlated pair
variables_to_remove = ['name_len','blurb_len']

# Dropping these variables
kickstarter_data = kickstarter_data.drop(columns=variables_to_remove)

# Counting the occurrences of each country in the dataset
country_counts = kickstarter_data['country'].value_counts()
country_counts_dict = country_counts.to_dict()
print(country_counts_dict)

# Grouping the countries in the dataset
kickstarter_data['country_group'] = kickstarter_data['country'].apply(lambda x: x if x in ['US', 'GB', 'CA', 'AU'] else 'Others')

# Dropping the original 'country' column
kickstarter_data = kickstarter_data.drop(columns=['country'])


# Encoding Categorical Variables
categorical_columns = kickstarter_data.select_dtypes(include=['object', 'category']).columns
categorical_columns = categorical_columns[categorical_columns != 'state']  # Exclude 'state' from encoding
# Convert 'state' column to binary
kickstarter_data['state'] = (kickstarter_data['state'] == 'successful').astype(int)

kickstarter_data_encoded = pd.get_dummies(kickstarter_data, columns=categorical_columns, drop_first=True)

# Normalizing/Standardizing Numeric Variables
numeric_columns_for_scaling = kickstarter_data_encoded.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
kickstarter_data_encoded[numeric_columns_for_scaling] = scaler.fit_transform(kickstarter_data_encoded[numeric_columns_for_scaling])


##########################################
######### Step 1 #########################
##########################################

X = kickstarter_data_encoded.drop('state', axis=1)
y = kickstarter_data_encoded['state']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)

# Apply StandardScaler for feature scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy for {name}: {accuracy}\n")
    print(f"Classification Report for {name}:\n {report}\n")

# Random Forest: 0.7457158651188502
# KNN: 0.6788280818131565
# Gradient Boosting: 0.7531785516860143

# Update the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': list(range(300, 801, 100)),  # n_estimators from 300 to 800, stepping by 100
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Define parameter grid for KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

# Update the parameter grid for Gradient Boosting
param_grid_gb = {
    'n_estimators': list(range(300, 801, 100)),  # n_estimators from 300 to 800, stepping by 100
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Initialize GridSearchCV for each model
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, cv=5, n_jobs=-1, verbose=2)
grid_search_gb = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), param_grid=param_grid_gb, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
# Assuming X_train and y_train are already defined
grid_search_rf.fit(X_train, y_train)
grid_search_knn.fit(X_train, y_train)
grid_search_gb.fit(X_train, y_train)

# Extract best parameters and accuracy
best_params_rf = grid_search_rf.best_params_
best_accuracy_rf = grid_search_rf.best_score_
best_params_knn = grid_search_knn.best_params_
best_accuracy_knn = grid_search_knn.best_score_
best_params_gb = grid_search_gb.best_params_
best_accuracy_gb = grid_search_gb.best_score_

# Print results
print(f"Best Parameters for Random Forest: {best_params_rf}") #max_depth': 20, 'min_samples_split': 5, 'n_estimators': 600
print(f"Best Accuracy for Random Forest: {best_accuracy_rf}")
print(f"Best Parameters for KNN: {best_params_knn}") #'n_neighbors': 7, 'weights': 'uniform'
print(f"Best Accuracy for KNN: {best_accuracy_knn}")
print(f"Best Parameters for Gradient Boosting: {best_params_gb}") #'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300
print(f"Best Accuracy for Gradient Boosting: {best_accuracy_gb}")

# Random Forest: 0.7509870660313138
# KNN: 0.6837304288631721
# Gradient Boosting: 0.7567052416609938

##########################################
######### Step 2 #########################
##########################################

# Prepare the dataset
X = kickstarter_data_encoded.drop('state', axis=1)
y = kickstarter_data_encoded['state']

# Initialize a Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)

# Initialize RFECV
rfecv = RFECV(estimator=rf_clf, step=1, cv=5, scoring='accuracy')

# Fit RFECV
rfecv.fit(X, y)

# Get the features selected by RFECV - For time saving I have hardcoded the selected features 
#                                      but these were given by the RFCEV
selected_features = ['name_len_clean', 'deadline_day', 'deadline_hr', 'created_at_day',
       'created_at_hr', 'launched_at_day', 'launched_at_hr',
       'create_to_launch_days', 'launch_to_deadline_days', 'goal_usd']

print(selected_features)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X[selected_features])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

## Kmean clustering:
optimal_k = 6 
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X[selected_features])

kickstarter_data['cluster'] = y_kmeans

## Clusters analyis:
# For numerical features
cluster_descriptive_stats = kickstarter_data.groupby('cluster').describe()

# For a specific categorical feature
cluster_category_counts = kickstarter_data.groupby(['cluster', 'category']).size().unstack(fill_value=0)


# Bar chart for a categorical feature across clusters
sns.countplot(x='category', hue='cluster', data=kickstarter_data)
plt.title('Category Group Distribution per Cluster')
plt.show()

# Countplot for countries by cluster
plt.figure(figsize=(12, 8))
sns.countplot(x='country_group', hue='cluster', data=kickstarter_data)
plt.title('Country Group Distribution per Cluster')
plt.xticks(rotation=45)  # Rotate x-axis labels if they overlap
plt.legend(title='Cluster')
plt.show()


pca = PCA(n_components=2)
principal_components = pca.fit_transform(X[selected_features])
plt.figure(figsize=(10, 8))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=kickstarter_data['cluster'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Kickstarter Projects')
plt.colorbar()
plt.show()


# Extract centroids
centroids = kmeans.cluster_centers_

# Convert centroids into a DataFrame for easier interpretation
centroids_df = pd.DataFrame(centroids, columns=selected_features)

# Calculate the mean of the features for the overall dataset
overall_mean = X[selected_features].mean()

# Subtract the overall mean from the centroids to see the deviation
centroid_deviations = centroids_df - overall_mean

# Add a cluster label column to the centroid deviations for easy plotting
centroid_deviations['cluster'] = [f'Cluster {i+1}' for i in range(optimal_k)]

# Melt the centroid deviations for easier seaborn plotting
centroid_deviations_melted = centroid_deviations.melt(id_vars='cluster', var_name='Feature', value_name='Deviation from Mean')

# Plotting the deviation of each cluster centroid from the overall mean for each feature
plt.figure(figsize=(14, 8))
sns.barplot(x='Feature', y='Deviation from Mean', hue='cluster', data=centroid_deviations_melted)
plt.xticks(rotation=90)  # Rotate feature names for better readability
plt.title('Feature Deviation from Mean for Each Cluster')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend out of the plot
plt.tight_layout()  # Adjust layout to fit feature names
plt.show()


## Model Evaluation:
score = silhouette_score(X[selected_features], y_kmeans)
print("Silhouette Score: ", score)  ## 0.13185

for k in range(2, 10):  # Trying different k values
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X[selected_features])
    score = silhouette_score(X[selected_features], y_kmeans)
    print(f"Silhouette Score for {k} clusters: {score}")

## DBSCAN

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X[selected_features])

# Add DBSCAN cluster labels to the DataFrame
kickstarter_data['dbscan_cluster'] = y_dbscan

score = silhouette_score(X[selected_features], y_dbscan)
print("Silhouette Score: ", score) ## -0.4533

## Hierarchical Clustering

Z = linkage(X[selected_features], method='ward')

# Plotting the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

n_clusters = 5  
cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
cluster_labels = cluster.fit_predict(X[selected_features])

# Adding the cluster labels to your original DataFrame
kickstarter_data['hierarchical_cluster'] = cluster_labels

silhouette_avg = silhouette_score(X[selected_features], cluster_labels)
print("The average silhouette_score is :", silhouette_avg)  # 0.09934
