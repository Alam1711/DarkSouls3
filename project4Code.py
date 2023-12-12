###########################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load data into frame
df = pd.read_csv('DS3_weapon.csv')

# Extract the first value from the 'Physical' column and convert it to numeric
df['Damage'] = df['Damage'].apply(lambda x: int(x.split('/')[0]))

# Select relevant features (base damage output and weight)
X = df[['Damage', 'Weight']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a categorical target variable based on base damage output
df['damage_category'] = pd.cut(df['Damage'], bins=[-float('inf'), 80, 120, float('inf')], labels=['Less than 80', '80-120', 'More than 120'])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the clusters with decision boundaries
plt.figure(figsize=(14, 8))

# Scatter plot for weapons with base damage <= 80
plt.scatter(df['Damage'][df['damage_category'] == 'Less than 80'], df['Weight'][df['damage_category'] == 'Less than 80'], c='blue', label='Base Damage <= 80')

# Scatter plot for weapons with base damage between 80 and 120
plt.scatter(df['Damage'][df['damage_category'] == '80-120'], df['Weight'][df['damage_category'] == '80-120'], c='green', label='80 < Base Damage <= 120')

# Scatter plot for weapons with base damage > 120 (marked with a red circle)
red_dots = df[df['damage_category'] == 'More than 120']
plt.scatter(red_dots['Damage'], red_dots['Weight'], c='red', label='Base Damage > 120')

plt.title('KMeans Clustering of Weapons based on Base Damage Output, Weight, and Damage Category')
plt.xlabel('Base Damage')
plt.ylabel('Weight')
plt.legend()

plt.show()



