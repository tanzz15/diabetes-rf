import pandas as pd
import mlflow.pyfunc


class DiabetesPyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, preprocessor, model, threshold):
        self.preprocessor = preprocessor
        self.model = model
        self.threshold = threshold

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        X_processed = self.preprocessor.transform(model_input)
        proba = self.model.predict_proba(X_processed)[:, 1]
        pred = (proba >= self.threshold).astype(int)

        return pd.DataFrame({
            "prediction": pred,
            "probability_diabetes": proba,
            "threshold": self.threshold
        })