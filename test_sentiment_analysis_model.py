# test_sentiment_analysis_model.py
import unittest
import mlflow.pyfunc

class TestSentimentAnalysisModel(unittest.TestCase):
    def test_model_prediction(self):
        # Configure MLflow to connect to Databricks
        mlflow.set_tracking_uri('databricks')
        mlflow.set_registry_uri('databricks')

        # Load the model from the Databricks registry
        model_name = "sentiment_analysis_model"
        stage = "Production"  # Or use a specific version if preferred
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

        # Example test data (simple input for demonstration)
        test_input = [[0.1, 0.1, 0.1, 0.7]]  # Adjust based on your actual input features for the model
        prediction = model.predict(test_input)
        self.assertIsNotNone(prediction)  # Check if the prediction is not None

        # Optionally, you can add more checks here to validate prediction accuracy or response format

if __name__ == '__main__':
    unittest.main()
