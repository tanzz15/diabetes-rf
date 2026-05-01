import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("processed.csv")

X = df.drop("Outcome", axis=1) #data input (glukosa, BMI, dll)
y = df["Outcome"] #diabetes/tidak

# =====================
# MLFLOW SETUP
# =====================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Diabetes_RF_Final")

# =====================
# LOOP TRAINING
# =====================
from imblearn.over_sampling import SMOTE

strategies = [
    {"name": "baseline", "class_weight": None, "use_smote": False},
    {"name": "class_weight", "class_weight": "balanced", "use_smote": False},
    {"name": "smote", "class_weight": None, "use_smote": True}
]

params_list = [
    {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },

    {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt"
    },

    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "max_features": "log2"
    },

    {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt"
    }
]

for strat in strategies:
    for params in params_list:

        run_name = f"{strat['name']}_{params['n_estimators']}"

        with mlflow.start_run(run_name=f"{strat['name']}_{params['n_estimators']}"):   

            # SPLIT
            X_train, X_test, y_train, y_test = train_test_split( 
                X, y, test_size=0.3, stratify=y, random_state=42
            )

            mlflow.log_param("test_size", 0.3) 
            mlflow.log_param("n_estimators", params["n_estimators"]) 
            mlflow.log_param("max_depth", params["max_depth"]) 
            mlflow.log_param("min_samples_split", params["min_samples_split"]) 
            mlflow.log_param("min_samples_leaf", params["min_samples_leaf"]) 
            mlflow.log_param("max_features", params["max_features"]) 
            mlflow.log_param("class_weight", strat["class_weight"])

            # PIPELINE
            if strat["use_smote"]:
                pipeline = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('smote', SMOTE(random_state=42)),
                    ('model', RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        min_samples_leaf=params["min_samples_leaf"],
                        max_features=params["max_features"],
                        class_weight=strat["class_weight"],
                        random_state=42
                    ))
                ])
            else:
                pipeline = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('model', RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        min_samples_leaf=params["min_samples_leaf"],
                        max_features=params["max_features"],
                        class_weight=strat["class_weight"],
                        random_state=42
                    ))
                ])
                
            # TRAIN
            pipeline.fit(X_train, y_train) 

            # Cross Validation
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=5,
                scoring='f1'
            )

            cv_mean = cv_scores.mean()

            mlflow.log_metric("cv_f1_mean", cv_mean)

            print("CV F1:", cv_mean)

            # PREDICT
            y_prob = pipeline.predict_proba(X_test)[:,1]

            threshold = 0.4

            y_pred = (y_prob >= threshold).astype(int) 

            # METRICS
            acc = accuracy_score(y_test, y_pred) 
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            mlflow.log_metric("accuracy", acc) #Seberapa banyak benar
            mlflow.log_metric("precision", precision) #Seberapa tepat
            mlflow.log_metric("recall", recall)  #Seberapa data yg positif
            mlflow.log_metric("f1_score", f1) #Precision & recall
            mlflow.log_metric("auc", auc) #Area Under the Curve

            # CONFUSION MATRIX
            cm = confusion_matrix(y_test, y_pred) 
            plt.figure()
            sns.heatmap(cm, annot=True, fmt='d')
            plt.title("Confusion Matrix")
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png") 
            plt.close()

            # CLASSIFICATION REPORT
            report = classification_report(y_test, y_pred) 
            print(report)

            with open("classification_report.txt", "w") as f:
                f.write(report) 

            mlflow.log_artifact("classification_report.txt")

            # MODEL LOGGING
            mlflow.sklearn.log_model(
                pipeline,
                "model",
                input_example=X_train.iloc[:5] 
            )

            joblib.dump(pipeline, f"model_{run_name}.pkl")

        print(f"{strat['name']} | n={params['n_estimators']} | Acc={acc:.3f} | F1={f1:.3f}")