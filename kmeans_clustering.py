import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("data/lung_cancer.csv")

X = data.drop("LUNG_CANCER", axis=1)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# BEFORE OPTIMIZATION (poor initialization)
kmeans_before = KMeans(n_clusters=3, init="random", n_init=1, random_state=42)
kmeans_before.fit(X_scaled)

before_inertia = kmeans_before.inertia_

print("Before Optimization:", before_inertia)

# AFTER OPTIMIZATION (better initialization)
kmeans_after = KMeans(n_clusters=3, init="k-means++", n_init=20, random_state=42)
kmeans_after.fit(X_scaled)

after_inertia = kmeans_after.inertia_

print("After Optimization:", after_inertia)

# Comparison graph
labels = ["Before Optimization", "After Optimization"]
values = [before_inertia, after_inertia]

plt.bar(labels, values)
plt.title("K-Means Objective Function Comparison")
plt.ylabel("Inertia (Objective Function)")

plt.savefig("results/kmeans_optimization.png")
plt.show()
