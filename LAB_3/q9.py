import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def get_classification_metrics(train_csv, test_csv, k=3):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    X_tr, y_tr = train.drop('Label', axis=1), train['Label']
    X_te, y_te = test.drop('Label', axis=1), test['Label']

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_tr, y_tr)

    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)

    results = {
        'Train Accuracy': accuracy_score(y_tr, y_tr_pred),
        'Test Accuracy': accuracy_score(y_te, y_te_pred),
        'Train CM': confusion_matrix(y_tr, y_tr_pred),
        'Test CM': confusion_matrix(y_te, y_te_pred),
        'Train Report': classification_report(y_tr, y_tr_pred),
        'Test Report': classification_report(y_te, y_te_pred)
    }
    return results

# Main
metrics = get_classification_metrics('groundwater_train.csv', 'groundwater_test.csv')

print(f"Train Accuracy: {metrics['Train Accuracy']*100:.2f}%")
print(f"Test Accuracy: {metrics['Test Accuracy']*100:.2f}%\n")
print("Train Confusion Matrix:\n", metrics['Train CM'])
print("Test Confusion Matrix:\n", metrics['Test CM'])
print("\nTrain Classification Report:\n", metrics['Train Report'])
print("Test Classification Report:\n", metrics['Test Report'])
