from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features:", iris.feature_names)
print("Flower types:", iris.target_names)
print("Number of samples:", len(iris.data))
print("Predictions:", predictions)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy * 100, "%")
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

joblib.dump(model, "outputs/iris_model.pkl")
print("Model saved successfully!")