import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor

# Load your dataset
data = pd.read_csv("Dataset/final_dataset.csv")  # adjust path if needed

# Choose your features and target (update as per your dataset)
X = data.drop(columns=["DO", "Station", "Year"])  # input features
y = data["DO"]  # target column to predict

# Train a basic model
model = DecisionTreeRegressor()
model.fit(X, y)

# Save the model and feature names
joblib.dump(model, "pollution_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
