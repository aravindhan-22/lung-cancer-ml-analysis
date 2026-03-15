import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

# Load dataset
data = pd.read_csv("data/lung_cancer.csv")

# Split features and target
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Baseline Logistic Regression
baseline = LogisticRegression()
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_pred)

# SGD Model
sgd = SGDClassifier(loss="log_loss", max_iter=1000)
sgd.fit(X_train, y_train)

sgd_pred = sgd.predict(X_test)

sgd_accuracy = accuracy_score(y_test, sgd_pred)

print("Baseline Accuracy:", baseline_accuracy)
print("SGD Accuracy:", sgd_accuracy)

# Convergence analysis
epochs = 50
loss_values = []

sgd_conv = SGDClassifier(loss="log_loss")
classes = np.unique(y_train)

for epoch in range(epochs):
    sgd_conv.partial_fit(X_train, y_train, classes=classes)
    probs = sgd_conv.predict_proba(X_train)
    loss = log_loss(y_train, probs)
    loss_values.append(loss)

plt.plot(loss_values, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.title("SGD Convergence Analysis")
plt.grid()

plt.savefig("results/sgd_convergence.png")
plt.show()

# Comparison chart
models = ["Logistic", "SGD"]
scores = [baseline_accuracy, sgd_accuracy]

plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.savefig("results/model_comparison.png")
plt.show()