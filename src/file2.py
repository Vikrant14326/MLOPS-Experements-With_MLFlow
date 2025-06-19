import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='Vikrant14326', repo_name='MLOPS-Experements-With_MLFlow', mlflow=True)

# Set MLflow to use DagsHub
mlflow.set_tracking_uri("https://dagshub.com/Vikrant14326/MLOPS-Experements-With_MLFlow.mlflow")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define RF model params
max_depth = 8
n_estimators = 5

# Set experiment
mlflow.set_experiment('YT-MLOPS-Exp2')

with mlflow.start_run(run_name="Random Forest Classifier"):
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save and log confusion matrix plot
    cm_plot_path = "Confusion-matrix.png"
    plt.savefig(cm_plot_path)
    mlflow.log_artifact(cm_plot_path)

    # Log current script
    mlflow.log_artifact(__file__)

    # Set tags
    mlflow.set_tags({"Author": "Vikrant Kumar", "Project": "Wine Classification"})

    # 🔧 Save and log the model manually (DagsHub does not support log_model API fully)
    model_dir = "rf_model"
    mlflow.sklearn.save_model(rf, model_dir)
    mlflow.log_artifacts(model_dir, artifact_path="Random-Forest-Model")

    print(f"✅ Accuracy: {accuracy}")
