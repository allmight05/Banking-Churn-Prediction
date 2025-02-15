import sys
from flask import Flask, render_template, request
import pandas as pd
from src.exceptions.exception import ProjectException
from src.utils.util import load_object

# Import your pipeline classes (ensure the module path is correct)
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve data from form
        age = int(request.form.get("age"))
        gender = request.form.get("gender")
        region_category = request.form.get("region_category")
        membership_category = request.form.get("membership_category")
        joining_date = request.form.get("joining_date")
        joined_through_referral = request.form.get("joined_through_referral")
        preferred_offer_types = request.form.get("preferred_offer_types")
        medium_of_operation = request.form.get("medium_of_operation")
        internet_option = request.form.get("internet_option")
        last_visit_time = request.form.get("last_visit_time")
        days_since_last_login = int(request.form.get("days_since_last_login"))
        avg_time_spent = float(request.form.get("avg_time_spent"))
        avg_transaction_value = float(request.form.get("avg_transaction_value"))
        avg_frequency_login_days = float(request.form.get("avg_frequency_login_days"))
        points_in_wallet = float(request.form.get("points_in_wallet"))
        used_special_discount = request.form.get("used_special_discount")
        offer_application_preference = request.form.get("offer_application_preference")
        past_complaint = request.form.get("past_complaint")
        complaint_status = request.form.get("complaint_status")
        feedback = request.form.get("feedback")
        
        # Create an instance of CustomData without churn risk score
        data = CustomData(
            age=age,
            gender=gender,
            region_category=region_category,
            membership_category=membership_category,
            joining_date=joining_date,
            joined_through_referral=joined_through_referral,
            preferred_offer_types=preferred_offer_types,
            medium_of_operation=medium_of_operation,
            internet_option=internet_option,
            last_visit_time=last_visit_time,
            days_since_last_login=days_since_last_login,
            avg_time_spent=avg_time_spent,
            avg_transaction_value=avg_transaction_value,
            avg_frequency_login_days=avg_frequency_login_days,
            points_in_wallet=points_in_wallet,
            used_special_discount=used_special_discount,
            offer_application_preference=offer_application_preference,
            past_complaint=past_complaint,
            complaint_status=complaint_status,
            feedback=feedback
        )
        
        # Convert input data to a DataFrame
        input_df = data.get_data_as_data_frame()
        
        # Create a prediction pipeline and get the churn risk prediction
        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)
        
        # Render the result page with the prediction (churn risk score)
        return render_template('result.html', prediction=prediction[0])
    
    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
