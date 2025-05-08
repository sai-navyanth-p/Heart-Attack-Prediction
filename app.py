from flask import Flask, render_template, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel

app = Flask(__name__)

# Start Spark session (required for PySpark models)
spark = SparkSession.builder.appName("HeartAttackWebApp").getOrCreate()

# Load pipeline and model
pipeline = PipelineModel.load("saved_models/preprocessing_pipeline")
rf_model = RandomForestClassificationModel.load("saved_models/random_forest")

# List of all features (must match your form and model)
all_features = [
    "Age", "Gender", "Cholesterol", "BloodPressure", "HeartRate", "BMI", "Smoker", "Diabetes", "Hypertension",
    "FamilyHistory", "PhysicalActivity", "AlcoholConsumption", "Diet", "StressLevel", "Ethnicity", "Income",
    "EducationLevel", "Medication", "ChestPainType", "ECGResults", "MaxHeartRate", "ST_Depression",
    "ExerciseInducedAngina", "Slope", "NumberOfMajorVessels", "Thalassemia", "PreviousHeartAttack",
    "StrokeHistory", "Residence", "EmploymentStatus", "MaritalStatus"
]

numeric_features = [
    "Age", "Cholesterol", "BloodPressure", "HeartRate", "BMI", "Smoker", "Diabetes", "Hypertension",
    "FamilyHistory", "PhysicalActivity", "AlcoholConsumption", "StressLevel", "Income", "MaxHeartRate",
    "ST_Depression", "NumberOfMajorVessels", "PreviousHeartAttack", "StrokeHistory"
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Prepare input dict for Spark DataFrame
        input_dict = {}
        for feat in all_features:
            val = data.get(feat)
            # Convert numeric fields
            if feat in numeric_features:
                try:
                    input_dict[feat] = float(val)
                except Exception:
                    input_dict[feat] = 0.0
            else:
                input_dict[feat] = val
        # Create Spark DataFrame
        new_df = spark.createDataFrame([input_dict])
        # Preprocess and predict
        new_df_prep = pipeline.transform(new_df)
        pred_row = rf_model.transform(new_df_prep).select("prediction", "probability").collect()[0]
        pred = int(pred_row["prediction"])
        prob = pred_row["probability"][int(pred)]
        prediction = "Heart Attack" if pred == 1 else "No Heart Attack"
        confidence = f"{prob*100:.2f}%"
        disclaimer = "This prediction is for demonstration only and not for medical use."
        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "disclaimer": disclaimer
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
