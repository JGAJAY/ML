import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def accuracy_vs_k_plot(train_csv, test_csv, k_max=11):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    X_train, y_train = train.drop('Label', axis=1), train['Label']
    X_test, y_test = test.drop('Label', axis=1), test['Label']

    results = []
    for k in range(1, k_max + 1):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        results.append((k, acc))
        print(f"k = {k}, Accuracy = {acc * 100:.2f}%")

    ks, accs = zip(*results)
    plt.plot(ks, [v * 100 for v in accs], marker='x', linestyle='-')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("k-NN Classification Accuracy")
    plt.grid()
    plt.savefig("knn_k_accuracy_variation.png")
    plt.show()

# Main
accuracy_vs_k_plot('groundwater_train.csv', 'groundwater_test.csv')
