import joblib
from sklearn.datasets import load_iris

model = joblib.load("outputs/iris_model.pkl")
iris = load_iris()

new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)
print("Predicted flower:", iris.target_names[prediction[0]])