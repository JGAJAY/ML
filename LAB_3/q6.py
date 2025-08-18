import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def evaluate_knn_model(model_path, test_csv):
    model = joblib.load(model_path)
    test_data = pd.read_csv(test_csv)
    features = test_data.drop('Label', axis=1)
    labels = test_data['Label']
    predictions = model.predict(features)
    return accuracy_score(labels, predictions)

# Main
score = evaluate_knn_model('knn_model.pkl', 'groundwater_test.csv')
print(f"Model accuracy on test set: {score * 100:.2f}%")
