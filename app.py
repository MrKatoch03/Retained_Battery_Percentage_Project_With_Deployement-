from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import BatteryData, PredictPipeline

application = Flask(__name__)
app = application


## Route for the home page
@app.route('/')
def index():
    return render_template('index.html')


## Route for prediction input form and result
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collecting all input features from form
        data = BatteryData(
            manufacturer=request.form.get('manufacturer'),
            chemistry=request.form.get('chemistry'),
            capacity_kWh=float(request.form.get('capacity_kWh')),
            charge_cycles=float(request.form.get('charge_cycles')),
            avg_temp_celsius=float(request.form.get('avg_temp_celsius')),
            discharge_rate_c=float(request.form.get('discharge_rate_c')),
            charge_rate_c=float(request.form.get('charge_rate_c')),
            avg_soc_percent=float(request.form.get('avg_soc_percent')),
            storage_time_months=float(request.form.get('storage_time_months')),
            fast_charge_ratio=float(request.form.get('fast_charge_ratio')),
            calendar_age_years=float(request.form.get('calendar_age_years'))
        )

        # Converting data to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:\n", pred_df)

        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Rendering the result
        return render_template('home.html', results=round(results[0], 2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080)
