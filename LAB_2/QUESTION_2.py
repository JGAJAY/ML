import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

data = {
    'Candies': [20, 16, 27, 19, 24, 22, 15, 18, 21, 16],
    'Mangoes': [6, 3, 6, 1, 4, 1, 4, 4, 1, 2],
    'Milk': [2, 6, 2, 2, 2, 5, 2, 2, 4, 4],
    'Payment': [386, 289, 393, 110, 280, 167, 271, 274, 148, 198]
}

df = pd.DataFrame(data)
df['Label'] = ['RICH' if p > 200 else 'POOR' for p in df['Payment']]
X = df[['Candies', 'Mangoes', 'Milk']].values
y = np.array([1 if label == 'RICH' else 0 for label in df['Label']])

model = LogisticRegression()
model.fit(X, y)
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

print("---- Classification Report ----")
print(classification_report(y, predictions, target_names=['POOR', 'RICH']))
print(f"Accuracy: {accuracy:.2f}")

df['Predicted Label'] = ['RICH' if p == 1 else 'POOR' for p in predictions]
print("\n---- Customer Classification ----")
print(df[['Candies', 'Mangoes', 'Milk', 'Payment', 'Label', 'Predicted Label']])
