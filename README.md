# Iris Flower Classifier

This project is a simple machine learning application that classifies Iris flowers into three species: Setosa, Versicolor, and Virginica.

## Machine Learning Model
The project uses the K-Nearest Neighbors (KNN) algorithm from Scikit-learn.

## Dataset
The Iris dataset contains 150 samples with four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

## Project Structure
- `src/load_data.py` - Loads the Iris dataset.
- `src/train.py` - Trains and evaluates the model and saves the trained model and confusion matrix.
- `src/predict.py` - Loads the saved model and makes predictions.
- `tests/test_model.py` - Tests that the trained model works correctly.
- `outputs/iris_model.pkl` - Saved trained machine learning model.
- `requirements.txt` - Required Python packages.

## Run the Project

Train and save the model:

python3 src/train.py

Make a prediction:

python3 src/predict.py

Run the test:

python3 tests/test_model.py

## Result
The model successfully classifies Iris flowers and achieved high accuracy on the test dataset.