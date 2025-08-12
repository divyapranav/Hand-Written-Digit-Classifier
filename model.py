import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

digits = datasets.load_digits()

x = digits.data
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

svm_model = SVC(kernel='rbf', gamma=0.001, C=100)

svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

with open('model.pkl','wb') as f:
    pickle.dump(svm_model, f)
