import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

def knn_train_and_save(input_file, output_model, k=3):
    dataset = pd.read_csv(input_file)
    features = dataset.drop('Label', axis=1)
    target = dataset['Label']
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(features, target)
    joblib.dump(model, output_model)

# Main
knn_train_and_save('groundwater_train.csv', 'knn_model.pkl')
print("Model has been trained and saved to 'knn_model.pkl'")
