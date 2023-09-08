from flask import Flask, request, jsonify
import joblib
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

@app.route("/", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        year = int(data["year"])
        townCode = int(data["town_code"])
        flatTypeCode = int(data["flat_type_code"])
        storeyRangeCode = int(data["storey_range_code"])
        floorArea = int(data["floor_area_sqm"])
        flatModelCode = int(data["flat_model_code"])
        remainingLeaseYear = int(data["remaining_lease_year"])

        model = joblib.load("randomforest.joblib")
        pred = model.predict([[year, townCode, flatTypeCode, storeyRangeCode, floorArea, flatModelCode, remainingLeaseYear]])

        result = {
            "message": "Prediction successful",
            "year": year,
            "floorAreaSqm": floorArea,
            "predictedValue": float(pred[0])
        }

        return jsonify(result), 200
    except Exception as e:
        error_result = {
            "message": "Error occurred",
            "error": str(e)
        }
        return jsonify(error_result), 500

if __name__ == "__main__":
    app.run(debug=True)