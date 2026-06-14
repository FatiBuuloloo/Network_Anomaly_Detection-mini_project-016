import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import mlflow, mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_experiment("Network_Anomaly_Detection-Isolation_Forest")
with mlflow.start_run(run_name="Isolation_forest_Model"):
    if not os.path.exists("preprocessed_data.csv"):
        print("File dataset ditemukan, melanjutkan proses training...")
    df = pd.read_csv("preprocessed_data.csv")
# Splitting Data
    print("Splitting Data")
    data_normal = df[df['label']==0]
    data_not_normal = df[df['label']==1]
    data_train, data_train2 = train_test_split(data_normal, test_size=0.15, random_state=42)
    data_train.drop(columns=['attack_category','label'],axis=1,inplace=True)
    data_test = pd.concat([data_train2,data_not_normal],axis=0)
    ground_truth = data_test['label']
    data_test.drop(columns=['attack_category','label'],axis=1,inplace=True)
    print()    
    n_estimators = 100
    contamination = 0.1
    random_state = 42

    mlflow.log_param("Model_Name", "Isolation_Forest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("contamination", 0.1)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("Model_Name", "Isolation_Forest")
    print('training model')
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
    model.fit(data_train)
    print('model trained successfully')
    print()
    print('evaluating model')
    test_preds = model.predict(data_test)
    y_pred = [1 if p == -1 else 0 for p in test_preds]

    precission_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        ground_truth, y_pred, average=None
    )

    mlflow.log_metric("Precission_Normal_Class", precission_per_class[0])
    mlflow.log_metric("Recall_Normal_Class", recall_per_class[0])
    mlflow.log_metric("F1_Score_Normal_Class", f1_per_class[0])
    mlflow.log_metric("Precission_Anomali_Class", precission_per_class[1])
    mlflow.log_metric("Recall_Anomali_Class", recall_per_class[1])
    mlflow.log_metric("F1_Score_Anomali_Class", f1_per_class[1])

    print("========== METRICS PER CLASS LOGGED TO MLFLOW ==========")
    print(f"Kelas [Normal]  -> F1: {f1_per_class[0]:.4f} | Recall: {recall_per_class[0]:.4f}")
    print(f"Kelas [Anomali] -> F1: {f1_per_class[1]:.4f} | Recall: {recall_per_class[1]:.4f}\n")

    cm = confusion_matrix(ground_truth, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=['Pred Normal (0)', 'Pred Anomali (1)'], 
                yticklabels=['Aktual Normal (0)', 'Aktual Anomali (1)'])
    plt.title('Confusion Matrix - Isolation Forest')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plot_path = 'confusion_matrix.png'
    plt.savefig(plot_path)
    plt.close()

    
    mlflow.log_artifact(plot_path)
    mlflow.sklearn.log_model(model, artifact_path='ISO_Model')
    print("Done")