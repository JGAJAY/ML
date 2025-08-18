import pandas as pd
import joblib

def get_predictions(model_file, test_file):
    model = joblib.load(model_file)
    df = pd.read_csv(test_file)
    X = df.drop(columns='Label')
    y = df['Label']
    full_predictions = model.predict(X)
    return full_predictions, y.values, X.iloc[0]

# Main
y_pred, y_true, first_sample = get_predictions('knn_model.pkl', 'groundwater_test.csv')

print("Sample Predictions (first 10):")
for idx in range(10):
    print(f"Sample {idx+1} => Predicted: {y_pred[idx]}, Actual: {y_true[idx]}")

single_pred = joblib.load('knn_model.pkl').predict([first_sample.values])[0]
print("\nSingle vector prediction:", single_pred, "| Actual:", y_true[0])
