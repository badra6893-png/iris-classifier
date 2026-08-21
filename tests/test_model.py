import joblib

model = joblib.load("outputs/iris_model.pkl")
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

assert prediction[0] == 0
print("Test passed successfully!")