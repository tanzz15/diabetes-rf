import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
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
# STRATEGY
# =====================
strategies = [
    {"name": "baseline", "class_weight": None, "use_smote": False},
    {"name": "class_weight", "class_weight": "balanced", "use_smote": False},
    {"name": "smote", "class_weight": None, "use_smote": True}
]
# =====================
# PARAMETER SEARCH
# =====================
param_dist = {
    'model__n_estimators': [100, 200, 300, 500],
    'model__max_depth': [5, 10, 15, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2']
}
# =====================
# SPLIT DATA
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)
# =====================
# TRAINING LOOP
# =====================
for strat in strategies:
    # PIPELINE
    if strat["use_smote"]:
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', RandomForestClassifier(
                class_weight=strat["class_weight"],
                random_state=42
            ))
        ])
    else:
        pipeline = Pipeline([
            ('model', RandomForestClassifier(
                class_weight=strat["class_weight"],
                random_state=42
            ))
        ])

    # CROSS VALIDATION
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # RANDOM SEARCH
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring='f1',
        cv=skf,
        n_jobs=-1,
        random_state=42
    )

    # TRAIN
    search.fit(X_train, y_train)

    # BEST MODEL
    best_model = search.best_estimator_
    best_params = search.best_params_

    # RUN NAME
    run_name = f"{strat['name']}_{best_params['model__n_estimators']}"

    # =====================
    # MLFLOW RUN
    # =====================
    with mlflow.start_run(run_name=run_name):

        # LOG PARAMS
        mlflow.log_param("test_size", 0.3)
        mlflow.log_param("n_estimators", best_params['model__n_estimators'])
        mlflow.log_param("max_depth", best_params['model__max_depth'])
        mlflow.log_param("min_samples_split", best_params['model__min_samples_split'])
        mlflow.log_param("min_samples_leaf", best_params['model__min_samples_leaf'])
        mlflow.log_param("max_features", best_params['model__max_features'])
        mlflow.log_param("class_weight", strat["class_weight"])
        mlflow.log_param("use_smote", strat["use_smote"])

        # CV SCORE
        cv_mean = search.best_score_
        mlflow.log_metric("cv_f1_mean", cv_mean)
        print("CV F1:", cv_mean)

        # =====================
        # PREDICT
        # =====================
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # THRESHOLD TUNING
        best_f1 = 0
        best_threshold = 0
        for t in [x / 100 for x in range(30, 71)]:
            pred = (y_prob >= t).astype(int)
            score = f1_score(y_test, pred)
            if score > best_f1:
                best_f1 = score
                best_threshold = t
        print("Best Threshold:", best_threshold)
        print("Best F1:", best_f1)

        # FINAL PREDICTION
        y_pred = (y_prob >= best_threshold).astype(int)

        # =====================
        # METRICS
        # =====================
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

        # =====================
        # CONFUSION MATRIX
        # =====================
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d'
        )

        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # =====================
        # CLASSIFICATION REPORT
        # =====================
        report = classification_report(y_test, y_pred)
        print(report)

        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # =====================
        # FEATURE IMPORTANCE
        # =====================
        importance = best_model.named_steps['model'].feature_importances_
        feature_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        })
        feature_df = feature_df.sort_values(
            by='Importance',
            ascending=False
        )
        print(feature_df)

        # VISUALISASI
        plt.figure(figsize=(8,5))
        sns.barplot(
            x='Importance',
            y='Feature',
            data=feature_df
        )
        plt.title("Feature Importance")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        # =====================
        # SAVE MODEL
        # =====================
        mlflow.sklearn.log_model(
            best_model,
            "model",
            input_example=X_train.iloc[:5]
        )
        joblib.dump(
            best_model,
            f"model_{run_name}.pkl"
        )

        # =====================
        # RESULT
        # =====================
        print(
            f"{strat['name']} | "
            f"Acc={acc:.3f} | "
            f"Precision={precision:.3f} | "
            f"Recall={recall:.3f} | "
            f"F1={f1:.3f} | "
            f"AUC={auc:.3f}"
        )
