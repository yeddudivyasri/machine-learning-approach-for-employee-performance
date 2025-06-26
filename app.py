from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
import joblib
import os

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv("garments_worker_productivity.csv")

# Clean and convert 'quarter' (e.g., 'Quarter1') to numeric
df["quarter"] = df["quarter"].str.extract(r'(\d+)')
df["quarter"] = pd.to_numeric(df["quarter"], errors='coerce')
df = df.dropna(subset=["quarter"])
df["quarter"] = df["quarter"].astype(int)

# Target and features
y = df["actual_productivity"]
X = df.drop(columns=["actual_productivity", "date"], errors="ignore")

# One-hot encode categorical columns
categorical_cols = ["department", "day"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(X[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Merge encoded and numeric features
X_final = pd.concat([X.drop(columns=categorical_cols).reset_index(drop=True), encoded_df], axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train the LightGBM model
model = LGBMRegressor()
model.fit(X_train, y_train)

# Save the model, encoder, and feature names
joblib.dump((model, encoder, X_final.columns.tolist()), "model.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Load the model and encoder
        model, encoder, feature_names = joblib.load("model.joblib")

        # One-hot encode input
        encoded_input = encoder.transform(input_df[["department", "day"]])
        encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(["department", "day"]))

        # Merge encoded and numeric
        input_final = pd.concat([input_df.drop(columns=["department", "day"]), encoded_df], axis=1)
        input_final = input_final.reindex(columns=feature_names, fill_value=0)

        # Predict
        prediction = model.predict(input_final)[0]
        return jsonify({"predicted_productivity": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
