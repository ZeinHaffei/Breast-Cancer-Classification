"""
@author: Zein Al Haffei
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import ListedColormap


print("Task 1: Data Preprocessing ...")
print("Working on ...")


# **Task 1: Data Preprocessing**
# 1. Load and examine the Breast Cancer Wisconsin dataset attributes, target labels, and feature descriptions.
# Load the Breast Cancer Wisconsin dataset

data = load_breast_cancer()

# Create a DataFrame for easier data manipulation
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the target (0 for benign, 1 for malignant) to the DataFrame
df['target'] = data.target

# 2. Check for any missing values, and if present, apply appropriate handling techniques.
# Check for missing values
missing_values = df.isnull().sum()

# If missing values are found, you can apply handling techniques
if missing_values.sum() > 0:
    # For example, you can choose to drop rows with missing values:
    df.dropna(inplace=True)


# 3. Normalize or standardize the features to ensure fair comparison across algorithms.
# Separate the features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the features using StandardScaler
X_scaled = scaler.fit_transform(X)

# Create a new DataFrame with scaled features
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Concatenate the scaled features with the target variable
df_scaled['target'] = y

print("Task 1: done.\n")


# -------------------------------------------------------------------------#
# **Task 2: Implementation of Classification Algorithms**
print("Task 2: Implementation of Classification Algorithms ...")
print("Working on ...")

# Initialize the classifiers
logistic_regression = LogisticRegression(max_iter=1000000)  # Increase max_iter
naive_bayes = GaussianNB()
decision_tree = DecisionTreeClassifier()
svm_classifier = SVC(probability=True)
knn = KNeighborsClassifier()

print("Task 2: done.\n")

# -------------------------------------------------------------------------#
# **Task 3: Model Training and Evaluation**
print("Task 3: Model Training and Evaluation ...")
print("Working on ...")

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


models = [logistic_regression, naive_bayes, decision_tree, svm_classifier, knn]
model_names = ["Logistic Regression", "Naive Bayes", "Decision Tree", "SVM", "KNN"]
results = {}

# train the classifiers and evaluate the models
for model, name in zip(models, model_names):
    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Calculate ROC curve and AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
    else:
        fpr, tpr, auc = None, None, None

        # Store results in a dictionary
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": cm,
        "ROC Curve (FPR)": fpr,
        "ROC Curve (TPR)": tpr,
        "AUC": auc,
    }

# Print the evaluation results for each model
for name, metrics in results.items():
    print(f"Model: {name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")
    print("\n")

print("Task 3: done.\n")

# -------------------------------------------------------------------------#
# **Task 4: Visualization of Results**

print("Task 4: Visualization of Results ...")
print("Working on ...")

# 1. Plot ROC curves for each classification algorithm on the same graph.
plt.figure(figsize=(8, 6))

# Plot ROC curves for each model
for name, metrics in results.items():
    fpr = metrics["ROC Curve (FPR)"]
    tpr = metrics["ROC Curve (TPR)"]
    auc = metrics["AUC"]

    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f}')

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--')

# Set axis labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')

# Add a legend
plt.legend(loc="lower right")

# Show the plot
plt.show()


# 2. Create a heatmap to visualize the confusion matrix for each algorithm.
# Initialize a figure to hold the heatmaps
plt.figure(figsize=(12, 8))

# Plot confusion matrix heatmaps for each model
for i, (name, metrics) in enumerate(results.items(), start=1):
    cm = metrics["Confusion Matrix"]

    plt.subplot(2, 3, i)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# 3. Compare the decision boundaries of different algorithms using a scatter plot with two selected features.
# Select two features (change these feature names as needed)
feature1_name = 'mean radius'
feature2_name = 'mean texture'

# Extract the selected features
X_subset = df_scaled[[feature1_name, feature2_name]].values

# Mesh grid creation for the feature space
x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Create a scatter plot of the data points
plt.figure(figsize=(15, 6))


for i, (model, name) in enumerate(zip(models, model_names), start=1):
    plt.subplot(2, 3, i)

    # # Fit the model to the selected features
    model.fit(X_subset, y)

    # Predict the class labels for the points in the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create a contour plot to visualize the decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y, cmap=plt.cm.RdYlBu, s=20, edgecolor='k', label=name)

    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    plt.title(f'Decision Boundaries - {name}')
    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
print("Task 4: done.\n")

# -------------------------------------------------------------------------#
# Task 5: Interpretation and Analysis
# Done in documentation

# -------------------------------------------------------------------------#
# Task 6: Model Tuning and Improvement

print("Task 6: Model Tuning and Improvement ...")
print("Working on ...")

# 1. Experiment with hyperparameter tuning for each classification algorithm to optimize their performance
# Hyperparameter tuning for each model


for model, name in zip(models, model_names):
    # Define hyperparameter grid for grid search
    param_grid = {}

    if isinstance(model, LogisticRegression):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2']
        }
    elif isinstance(model, DecisionTreeClassifier):
        param_grid = {
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif isinstance(model, SVC):
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
    elif isinstance(model, KNeighborsClassifier):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }

    # Perform grid search cross-validation
    # grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=1)
    grid_search = GridSearchCV(model, param_grid, n_jobs=1)  # Use only one CPU core

    # grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=5, n_jobs=1)

    grid_search.fit(X_train, y_train)

    # Get the best model with tuned hyperparameters
    best_model = grid_search.best_estimator_

    # Predict on the test set
    y_pred = best_model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Calculate ROC curve and AUC
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
    else:
        fpr, tpr, auc = None, None, None

    # Store results in a dictionary
    results[name] = {
        "Best Hyperparameters": grid_search.best_params_,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": cm,
        "ROC Curve (FPR)": fpr,
        "ROC Curve (TPR)": tpr,
        "AUC": auc,
    }

# Print the evaluation results for each model with tuned hyperparameters
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"Best Hyperparameters: {metrics['Best Hyperparameters']}")
    for metric_name, value in metrics.items():
        if metric_name != 'Best Hyperparameters':
            print(f"{metric_name}: {value}")
    print("\n")

print("Task 6: done.\n")

# -------------------------------------------------------------------------#
# **Task 7: Visualizing Decision Boundaries **
print("Task 7: Visualizing Decision Boundaries ...")
print("Working on ...")

# Create a meshgrid of feature values
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Visualize decision boundaries for each model in the same figure
plt.figure(figsize=(12, 8))

models = {
    'Logistic Regression': logistic_regression,
    'Naive Bayes': naive_bayes,
    'Decision Tree': decision_tree,
    'SVM': svm_classifier,
    'KNN': knn
}

cmaps = ['#FFAAAA', '#AAAAFF']  # Color maps for the background

for i, (model_name, model) in enumerate(models.items(), start=1):
    plt.subplot(2, 3, i)

    # Make predictions on the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and data points
    plt.contourf(xx, yy, Z, cmap=ListedColormap(cmaps), alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(['#FF0000', '#0000FF']), s=20,
                edgecolor='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'Decision Boundary - {model_name}')

plt.tight_layout()
plt.show()

print("Task 7: done.\n")